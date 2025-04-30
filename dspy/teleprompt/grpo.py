import logging
import random
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.lm import LM
from dspy.clients.utils_finetune import TrainDataFormat, GRPOGroup
from dspy.dsp.utils.settings import settings
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.example import Example
from dspy.primitives.program import Program
from dspy.teleprompt.bootstrap_finetune import FinetuneTeleprompter, all_predictors_have_lms, prepare_teacher, bootstrap_trace_data


logger = logging.getLogger(__name__)


class GRPO(FinetuneTeleprompter):
    def __init__(
        self,
        metric: Optional[Callable] = None,
        multitask: bool = True,
        train_kwargs: Optional[Union[Dict[str, Any], Dict[LM, Dict[str, Any]]]] = None,
        adapter: Optional[Union[Adapter, Dict[LM, Adapter]]] = None,
        exclude_demos: bool = False,
        num_threads: int = 6,
        num_train_steps: int = 100,
        seed: int = 0,
        num_dspy_examples_per_grpo_step: int = 1,
        num_samples_per_input: int = 1,
        use_train_as_val: bool = False,
        num_steps_for_val: int = 5,
        variably_invoked_predictor_grouping_mode: Union[Literal['truncate'], Literal['fill'], Literal['ragged']] = 'truncate',
        variably_invoked_predictor_fill_strategy: Optional[Union[Literal['randint'], Literal['max']]] = None,
    ):
        super().__init__(train_kwargs=train_kwargs)
        self.metric = metric
        self.multitask = multitask
        self.adapter: Dict[LM, Adapter] = self.convert_to_lm_dict(adapter)
        self.exclude_demos = exclude_demos
        self.num_threads = num_threads
        self.num_train_steps = num_train_steps
        self.rng = random.Random(seed)
        self.num_dspy_examples_per_grpo_step = num_dspy_examples_per_grpo_step
        self.num_samples_per_input = num_samples_per_input
        self.use_train_as_val = use_train_as_val
        self.num_steps_for_val = num_steps_for_val

        assert exclude_demos, "exclude_demos==False is not supported yet. Please set it to True."
        assert multitask, "independent GRPO training jobs for each predictor in the student program is not supported yet. Please set multitask=True."

        # The backend will be called with a batch of (num_dspy_examples_per_grpo_step * (num_samples_per_input * len(teachers)) * num_predictors) per training set if multitask is True
        # If multitask is False, the backend will be called with a batch of (num_dspy_examples_per_grpo_step * (num_samples_per_input * len(teachers))) per training job
        self.variably_invoked_predictor_grouping_mode = variably_invoked_predictor_grouping_mode
        if variably_invoked_predictor_grouping_mode == 'fill':
            assert variably_invoked_predictor_fill_strategy is not None, "variably_invoked_predictor_fill_strategy must be set when variably_invoked_predictor_grouping_mode is 'fill'"
            assert variably_invoked_predictor_fill_strategy in ['randint', 'max'], "variably_invoked_predictor_fill_strategy must be either 'randint' or 'max'"
        self.variably_invoked_predictor_fill_strategy = variably_invoked_predictor_fill_strategy

    def compile(
        self,
        student: Program,
        trainset: List[Example],
        valset: Optional[List[Example]] = None,
        **kwargs,
    ) -> Program:
        logger.info("Validating the inputs...")

        # TODO(GRPO Team): Following checks are for unimplemented features.
        # Consider if we want to eventually implement them or remove. We don't
        # yet support:
        # * teacher programs
        # * multitask == False
        # * student program with multiple predictor LMs
        # The main reason for these is that we update the LMs in place. If these
        # LMs are shared between the different predictors of the student
        # program and we have multitask == False, we need to decide which steps
        # will use new LM copies and we need to ensure our decision is
        # consistent with any teacher LMs that share the same LMs.
        teacher = None

        if not self.multitask:
            raise ValueError(
                "Independent GRPO training jobs for each predictor in the student program "
                "are not supported yet. Please set multitask=True."
            )

        student_lms = {id(pred.lm) for pred in student.predictors()}
        assert len(student_lms) == 1, (
            f"Student program has multiple LMs: {student_lms}. "
            "GRPO only supports student programs with a single LM."
            "You can set the LM for a program with `program.set_lm(...)`"
        )

        # Our regular input validation starts here
        if self.use_train_as_val:
            assert valset is None, "If use_train_as_val is True, valset must be None."

        logger.info("Preparing the student program...")
        all_predictors_have_lms(student)
        initial_caches_student = disable_lm_cache(student)
        pred_signature_hash_to_ind = {hash(pred.signature): ind for ind, pred in enumerate(student.predictors())}
        num_student_predictors = len(student.predictors())

        logging.info("Preparing the teacher program(s)...")
        teachers = teacher if isinstance(teacher, list) else [teacher]
        initial_caches_teachers = []
        for ind, t in enumerate(teachers):
            teachers[ind] = prepare_teacher(student=student, teacher=t)
            initial_caches = disable_lm_cache(teachers[ind])
            initial_caches_teachers.append(initial_caches)

        # We generate num_samples_per_input trace per example per teacher
        # This way, we will have num_generations trace data for each example
        num_generations = self.num_samples_per_input * len(teachers)

        # Update train_kwargs
        for pred in student.predictors():
            train_kwargs = self.train_kwargs[pred.lm]
            train_kwargs = {} if train_kwargs is None else train_kwargs
            train_kwargs["num_generations"] = num_generations
            self.train_kwargs[pred.lm] = train_kwargs

        # We need to have a separate job for each unique LM x the data
        # collection strategy. This properly handles all combinations of
        # multitask and predictor LMs
        logger.info("Preparing the GRPO training job(s)...")
        grpo_training_jobs = {}
        for pred_ind, pred in enumerate(student.predictors()):
            data_key = None if self.multitask else pred_ind
            job_key = (pred.lm, data_key)
            if job_key not in grpo_training_jobs:
                train_kwargs = self.train_kwargs[pred.lm]
                job = pred.lm.reinforce(train_kwargs=train_kwargs)
                grpo_training_jobs[job_key] = job

        if valset is None and self.use_train_as_val:
            logger.info("Using the training set as the validation set.")
            valset = trainset
        elif valset is not None:
            logger.info("Using the user provided validation set.")
        else:
            logger.info("Not using any validation set.")

        if valset:
            assert isinstance(self.num_steps_for_val, int) and self.num_steps_for_val > 0, "num_steps_for_val must be a positive integer."
            valset_evaluator = Evaluate(
                devset=valset,
                num_threads=self.num_threads,
                display_progress=True,
                return_outputs=False,
                provide_traceback=True,  # TODO(check with team)
            )

            logger.info("Evaluating the student program on the validation set before training loop...")
            valset_evaluation = valset_evaluator(student, metric=self.metric)
            logger.info(f"Student program validation set score before training loop: {valset_evaluation}")


        logger.info("Starting the GRPO training loop...")
        for train_step_idx in range(self.num_train_steps):
            logger.info(f"GRPO training step {train_step_idx + 1}/{self.num_train_steps}...")
            subsample_training_dataset = self.rng.sample(trainset, self.num_dspy_examples_per_grpo_step)

            logger.info("Bootstrapping data...")
            trace_data = [[[] for _ in range(len(teachers))] for _ in range(len(subsample_training_dataset))]
            for tind, teacher in enumerate(teachers):
                for _ in range(self.num_samples_per_input):
                    # We rely on disabled caches to ensure that we get different
                    # traces
                    round_data = bootstrap_trace_data(
                        program=teacher,
                        dataset=subsample_training_dataset,
                        metric=self.metric,
                        num_threads=self.num_threads,
                    )
                    for data_dict in round_data:
                        trace_data[data_dict['example_ind']][tind].append(data_dict)

            # At this point, trace_data: List[example_idx -> List[teacher_idx -> [num_samples_per_input * Dict(example, prediction, trace, example_ind, score)]]]
            # Shape of trace is: [dspy_module_invocation_idx -> Tuple[Predictor, PredictorInputs, Prediction]]
            assert len(trace_data) == len(subsample_training_dataset), f"Trace data length {len(trace_data)} does not match the number of examples {len(subsample_training_dataset)}"
            assert len(trace_data[0]) == len(teachers), f"Trace data length {len(trace_data[0])} does not match the number of teachers {len(teachers)}"
            assert len(trace_data[0][0]) == self.num_samples_per_input, f"Trace data length {len(trace_data[0][0])} does not match the expected number of samples per input {self.num_samples_per_input}"
            assert "trace" in trace_data[0][0][0], "Trace data does not contain the 'trace' key"
            assert len(trace_data[0][0][0]["trace"]) > 0, "Trace data is empty"
            assert len(trace_data[0][0][0]["trace"][0]) == 3, f"Trace tuple length {len(trace_data[0][0][0]['trace'][0])} does not match the expected length 3"

            logger.info("Preparing the training data batch from bootstrapped examples for GRPO...")
            # Now, we need to prepare batches of data to be sent for training
            # Shape of train_batch_per_predictor: List[num_student_predictors -> List[ ]]
            train_batch_per_predictor: List[List[GRPOGroup]] = [[] for _ in range(num_student_predictors)]
            for pred_id in range(num_student_predictors):
                for example_ind, example_data in enumerate(trace_data):
                    # Each example_data is a list of teacher_idx -> [num_samples_per_input * Dict(example, prediction, trace, example_ind, score)]
                    # We need to flatten this list and create a batch for each predictor

                    # TODO(Lakshya, Omar, Noah): Discuss what to do with the same module being invoked multiple times within a single dspy.Example
                    predictor_example_invocations: List[List[Tuple]] = []

                    for teacher_idx, teacher_data in enumerate(example_data):
                        for sample in teacher_data:
                            # Each sample is a Dict(example, prediction, trace, example_ind, score)
                            # sample['prediction'] is module_level prediction
                            assert sample["example_ind"] == example_ind, f"Example index {sample['example_ind']} does not match the expected index {example_ind}"

                            trace_instances_for_current_pred = [(*t, sample["score"]) for t in sample["trace"] if hash(t[0].signature) == hash(student.predictors()[pred_id].signature)]

                            predictor_example_invocations.append(trace_instances_for_current_pred)

                    assert len(predictor_example_invocations) == num_generations, f"Number of predictor example invocations {len(predictor_example_invocations)} does not match the expected batch size {num_generations}"

                    min_len = min([len(predictor_example_invocations[i]) for i in range(len(predictor_example_invocations))])
                    max_len = max([len(predictor_example_invocations[i]) for i in range(len(predictor_example_invocations))])
                    if min_len == 0:
                        logger.warning(f"Skipping example {example_ind} for predictor {pred_id} as it has no invocations.")
                        continue

                    if self.variably_invoked_predictor_grouping_mode == 'truncate':
                        predictor_example_invocations = [invocation[:min_len] for invocation in predictor_example_invocations]
                    elif self.variably_invoked_predictor_grouping_mode == 'fill':
                        if self.variably_invoked_predictor_fill_strategy == 'randint':
                            selector = lambda l: self.rng.choice(l)
                        else:
                            selector = lambda l: l[-1]
                        predictor_example_invocations = [
                            invocation + [selector(invocation) for _ in range(max_len - len(invocation))]
                            for invocation in predictor_example_invocations
                        ]
                    else:
                        assert self.variably_invoked_predictor_grouping_mode == 'ragged', f"Unknown variably invoked predictor grouping mode {self.variably_invoked_predictor_grouping_mode}"

                    max_len = max([len(predictor_example_invocations[i]) for i in range(len(predictor_example_invocations))])

                    example_training_data: List[GRPOGroup] = [[] for _ in range(max_len)]

                    for group_idx in range(max_len):
                        for rollout_idx in range(len(predictor_example_invocations)):
                            trace_instance = predictor_example_invocations[rollout_idx][group_idx]
                            score = trace_instance[3]
                            # for module_invocation_idx, trace_instance in enumerate(trace_instances_for_current_pred):
                            # Each trace is a tuple of (Predictor, PredictorInputs, Prediction)
                            trace_pred_id = pred_signature_hash_to_ind[hash(trace_instance[0].signature)]
                            assert trace_pred_id == pred_id

                            predictor = trace_instance[0]
                            pred_lm = predictor.lm
                            adapter = self.adapter[pred_lm] or settings.adapter or ChatAdapter()
                            assert isinstance(adapter, ChatAdapter), f"Adapter {adapter} is not a ChatAdapter. GRPO training is not supported for this adapter."
                            # TODO(Lakshya): Currently we exclude demos from the training data
                            # TODO(GRPO Team): Use build_call_data_from_trace (from bootstrap_finetune) instead of
                            # dealing with the message formatting ourselves.
                            inp_messages = adapter.format(
                                signature=trace_instance[0].signature, 
                                inputs=trace_instance[1], 
                                demos=[] # TODO: Add support for demos
                            )
                            all_messages = adapter.format_finetune_data(
                                signature=trace_instance[0].signature,
                                inputs=trace_instance[1],
                                outputs=trace_instance[2],
                                demos=[] # TODO: Add support for demos
                            )['messages']

                            assert all_messages[:-1] == inp_messages, f"Input messages {inp_messages} do not match the expected messages {all_messages[:-1]}"

                            # if len(example_training_data[group_idx]["input"]["messages"]) == 0:
                            #     example_training_data[group_idx]["input"]["messages"] = [{"role": msg['role'], 'content': msg['content']} for msg in inp_messages]
                            # elif example_training_data[group_idx]["input"]["messages"] != inp_messages:
                            #     logger.info(f"Input messages {inp_messages} do not match the expected messages {example_training_data[group_idx]['input']['messages']}")

                            # response_msg = all_messages[-1]
                            # assert 'role' in response_msg and 'content' in response_msg, f"Response message {response_msg} does not contain the expected keys 'role' and 'content'"
                            # example_training_data[group_idx]["completions"].append({
                            #     "role": response_msg["role"],
                            #     "content": response_msg["content"],
                            #     "reward": score,
                            # })

                            example_training_data[group_idx].append({
                                "messages": inp_messages,
                                "completion": {
                                    "role": all_messages[-1]["role"],
                                    "content": all_messages[-1]["content"],
                                },
                                "reward": score,
                            })

                    train_batch_per_predictor[pred_id].extend(example_training_data)

            for predictor_train_batch in train_batch_per_predictor:
                for grpo_train_group in predictor_train_batch:
                    assert len(grpo_train_group) == num_generations, f"Number of completions {len(example_training_data['completions'])} does not match the expected number num_samples_per_input*len(teachers)={num_generations}"
                    if len(set(map(repr, grpo_train_group))) < 2:
                        # TODO(GRPO Team): How can we avoid this warning?
                        logger.warning(f"GRPOGroup has no diversity. This could be due to low temperature, or low number of rollouts, or the cache could be enabled inadvertently. The GRPOGroup is {grpo_train_group}.")

            # We now run the GRPO step. Notes:
            # * The job here has a reference to a particular M that's attached
            #   to the student program. We update the .model field of this LM
            #   inside the job, which also updates the LM in the student program
            #   since these point to the same reference (along with any teacher
            #   program that shares the same LM).
            # * TODO(GRPO Team): This is inconsistent with how
            #   BootstrapFinetune works, which creates new LM instances post
            #   training. We should decide whether the LMs should be updated in
            #   place or new LMs should be created, and standardize our approach
            #   for both. If we decide to create new LMs, we should find a way
            #   to update self.adapter and self.train_kwargs accordingly, in
            #   addition to updating any teacher programs that share the same
            #   LM.
            logger.info("Invoking GRPO training step...")
            for (lm, data_key), job in grpo_training_jobs.items():
                train_data: List[GRPOGroup] = sum(train_batch_per_predictor, []) if data_key is None else train_batch_per_predictor[data_key]
                job.step(train_data=train_data, train_data_format=TrainDataFormat.GRPO_CHAT)
            
            for (lm, data_key), job in grpo_training_jobs.items():
                if (train_step_idx + 1) % lm.kwargs["update_interval"] == 0 and train_step_idx != 0:
                    logger.info(f"Current train step is {train_step_idx + 1}. Updating the model...")
                    job.update_model()

            logger.info(f"GRPO training step {train_step_idx + 1}/{self.num_train_steps} completed.")
            if valset and ((train_step_idx + 1) % self.num_steps_for_val == 0 or train_step_idx + 1 == self.num_train_steps):
                logger.info(f"Evaluating the student program on the validation set after training step {train_step_idx + 1}/{self.num_train_steps}")
                valset_evaluation = valset_evaluator(student, metric=self.metric)
                logger.info(f"Student program validation set score after training step {train_step_idx + 1}/{self.num_train_steps}: {valset_evaluation}")

        logger.info("Done with the iterations! Retrieving the final model(s)...")
        for (lm, data_key), job in grpo_training_jobs.items():
            job.terminate()

        # Revert cache states to their initial values. Note that the student
        # program might have the same LM as one of the teachers. We first modify
        # the caches of the student LMs, then those of the teacher LMs. We
        # follow the opposite order when reverting.
        for tind, t in enumerate(teachers):
            for pind, pred in enumerate(t.predictors()):
                pred.lm.cache = initial_caches_teachers[tind][pind]

        for pind, pred in enumerate(student.predictors()):
            pred.lm.cache = initial_caches_student[pind]

        logger.info("GRPO compiler has finished compiling the student program")
        student._compiled = True
        return student


def disable_lm_cache(program: Program):
    """Disable the LM cache for all predictors in the program."""
    initial_caches = []
    for pred in program.predictors():
        if not pred.lm:
            raise ValueError(f"Cannot disable cache: predictor {pred} does not have an LM set.")
        initial_caches.append(pred.lm.cache)
        pred.lm.cache = False
    return initial_caches