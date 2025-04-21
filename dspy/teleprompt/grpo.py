import logging
from collections import defaultdict
import random
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from copy import deepcopy

import dspy
from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.lm import LM
from dspy.dsp.utils.settings import settings
from dspy.evaluate.evaluate import Evaluate
from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.primitives.program import Program
from dspy.teleprompt.teleprompt import Teleprompter

# TODO (Lakshya, Noah): This entire file should be appropriately refactored into an arbor LM and arbor provider
from dspy.clients.arbor.arbor import ArborGRPOTrainer, GRPOGroupMember, GRPOGroup, GRPOBatch

logger = logging.getLogger(__name__)

class FinetuneTeleprompter(Teleprompter):
    def __init__(
        self,
        train_kwargs: Optional[Union[Dict[str, Any], Dict[LM, Dict[str, Any]]]] = None,
    ):
        self.train_kwargs: Dict[LM, Any] = self.convert_to_lm_dict(train_kwargs or {})

    @staticmethod
    def convert_to_lm_dict(arg) -> Dict[LM, Any]:
        non_empty_dict = arg and isinstance(arg, dict)
        if non_empty_dict and all(isinstance(k, LM) for k in arg.keys()):
            return arg
        # Default to using the same value for all LMs
        return defaultdict(lambda: arg)

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
        num_rollouts_per_dspy_example_per_step: int = 1,
        use_train_as_val: bool = False,
        num_steps_for_val: int = 5,
        variably_invoked_predictor_grouping_mode: Union[Literal['truncate'], Literal['fill'], Literal['ragged']] = 'truncate',
        variably_invoked_predictor_fill_strategy: Optional[Union[Literal['randint'], Literal['max']]] = None,
        grpo_beta: float = 0.04,
        arbor_inference_model_update_interval: int = 25,
    ):
        # TODO(feature): Inputs train_kwargs (a dict with string keys) and
        # adapter (Adapter) can depend on the LM they are used with. We are
        # takingthese as parameters for the time being. However, they can be
        # attached to LMs themselves -- an LM could know which adapter it should
        # be used with along with the train_kwargs. This will lead the only
        # required argument for LM.finetune() to be the train dataset.

        super().__init__(train_kwargs=train_kwargs)
        self.metric = metric
        self.multitask = multitask
        self.adapter: Dict[LM, Adapter] = self.convert_to_lm_dict(adapter)
        self.exclude_demos = exclude_demos
        assert exclude_demos, "exclude_demos==False is not supported yet. Please set it to True." # TODO(Lakshya): Remove this when exclude_demos is supported
        self.num_threads = num_threads
        self.num_train_steps = num_train_steps
        self.rng = random.Random(seed)
        self.num_dspy_examples_per_grpo_step = num_dspy_examples_per_grpo_step
        self.num_rollouts_per_dspy_example_per_step = num_rollouts_per_dspy_example_per_step
        self.use_train_as_val = use_train_as_val
        self.num_steps_for_val = num_steps_for_val

        # The backend will be called with a batch of (num_dspy_examples_per_grpo_step * num_rollouts_per_dspy_example_per_step * num_predictors) per training set if multitask is True
        # If multitask is False, the backend will be called with a batch of (num_dspy_examples_per_grpo_step * num_rollouts_per_dspy_example_per_step) per training job

        self.per_predictor_batch_size = num_dspy_examples_per_grpo_step * num_rollouts_per_dspy_example_per_step

        self.variably_invoked_predictor_grouping_mode = variably_invoked_predictor_grouping_mode
        if variably_invoked_predictor_grouping_mode == 'fill':
            assert variably_invoked_predictor_fill_strategy is not None, "variably_invoked_predictor_fill_strategy must be set when variably_invoked_predictor_grouping_mode is 'fill'"
            assert variably_invoked_predictor_fill_strategy in ['randint', 'max'], "variably_invoked_predictor_fill_strategy must be either 'randint' or 'max'"
        self.variably_invoked_predictor_fill_strategy = variably_invoked_predictor_fill_strategy

        self.grpo_beta = grpo_beta
        self.arbor_inference_model_update_interval = arbor_inference_model_update_interval

    def compile(
        self, student: Program, trainset: List[Example], teacher: Optional[Union[Program, List[Program]]] = None, valset: Optional[List[Example]] = None, **kwargs
    ) -> Program:
        # TODO: Print statements can be converted to logger.info if we ensure
        # that the default DSPy logger logs info level messages in notebook
        # environments.
        logger.info("Preparing the student program...")
        num_student_predictors = len(student.predictors())
        pred_signature_has_to_pred_index = {hash(pred.signature): pred_id for pred_id, pred in enumerate(student.predictors())}

        logger.info("Preparing the student program by setting default lm for all predictors, and disabling cache for all used lms...")
        set_missing_predictor_lms(student)

        logger.info("Preparing the teacher program(s) by setting default lm for all predictors, and disabling cache for all used lms...")
        teachers = teacher if isinstance(teacher, list) else [teacher]
        teachers = [prepare_teacher(student, t) for t in teachers]
        for t in teachers:
            set_missing_predictor_lms(t)
        
        # We ensure that self.num_dspy_examples_per_grpo_step is perfectly divisible by num(teachers)
        assert self.num_rollouts_per_dspy_example_per_step % len(teachers) == 0, f"num_dspy_examples_per_grpo_step {self.num_dspy_examples_per_grpo_step} must be divisible by the number of teachers {len(teachers)}"
        
        # We generate self.num_rollouts_per_dspy_example_per_step / num(teachers) trace per example per teacher
        # This way, we will have self.num_rollouts_per_dspy_example_per_step trace data for each example
        num_samples_per_input = self.num_rollouts_per_dspy_example_per_step // len(teachers)

        # Initialize GRPO training jobs in the backend
        logger.info("Preparing the GRPO training job(s)...")
        grpo_training_jobs = [] # This maps each predictor_idx to its GRPO training job
        if self.multitask:
            # In this, there will be only one GRPO training job for the context LM
            model_names = {pred.lm.model for pred in student.predictors()}
            assert len(model_names) == 1, "The student program must have only one context LM."
            model_name = list(model_names)[0]
            lm_being_trained = student.get_lm()
            grpo_training_job = ArborGRPOTrainer(
                lm=lm_being_trained,
                suffix="grpo",
                beta=self.grpo_beta,
                num_generations=self.num_rollouts_per_dspy_example_per_step,
                update_interval=self.arbor_inference_model_update_interval
            )
            grpo_training_job.initialize()
            grpo_training_jobs = [grpo_training_job]
        else:
            # In this, there will be one GRPO training job for each predictor in the student program
            # TODO(Lakshya): Implement individual task training jobs for each predictor
            assert False, "Independent GRPO training jobs for each predictor in the student program are not supported yet."
        
        
        if self.use_train_as_val:
            assert valset is None, "If use_train_as_val is True, valset must be None."

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
            trace_data = [[] for _ in range(len(subsample_training_dataset))]
            for t in teachers:
                collected_trace_data = bootstrap_trace_data(
                    program=t, dataset=subsample_training_dataset, metric=self.metric, num_threads=self.num_threads, num_samples_per_input=num_samples_per_input
                )
                for example_ind, example_data in enumerate(collected_trace_data):
                    trace_data[example_ind].append(example_data)

            # At this point, trace_data: List[example_idx -> List[teacher_idx -> [num_samples_per_input * Dict(example, prediction, trace, example_ind, score)]]]
            # Shape of trace is: [dspy_module_invocation_idx -> Tuple[Predictor, PredictorInputs, Prediction]]
            assert len(trace_data) == len(subsample_training_dataset), f"Trace data length {len(trace_data)} does not match the number of examples {len(subsample_training_dataset)}"
            assert len(trace_data[0]) == len(teachers), f"Trace data length {len(trace_data[0])} does not match the number of teachers {len(teachers)}"
            assert len(trace_data[0][0]) == num_samples_per_input, f"Trace data length {len(trace_data[0][0])} does not match the expected number of samples per input {num_samples_per_input}"
            assert "trace" in trace_data[0][0][0], "Trace data does not contain the 'trace' key"
            assert len(trace_data[0][0][0]["trace"]) > 0, "Trace data is empty"
            assert len(trace_data[0][0][0]["trace"][0]) == 3, f"Trace tuple length {len(trace_data[0][0][0]['trace'][0])} does not match the expected length 3"

            logger.info("Preparing the training data batch from bootstrapped examples for GRPO...")
            # Now, we need to prepare batches of data to be sent for training
            # Shape of train_batch_per_predictor: List[num_student_predictors -> List[ ]]
            train_batch_per_predictor: List[GRPOBatch] = [[] for _ in range(num_student_predictors)]
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

                    assert len(predictor_example_invocations) == self.num_rollouts_per_dspy_example_per_step, f"Number of predictor example invocations {len(predictor_example_invocations)} does not match the expected batch size {self.per_predictor_batch_size}"

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
                            trace_pred_id = pred_signature_has_to_pred_index[hash(trace_instance[0].signature)]
                            assert trace_pred_id == pred_id

                            predictor = trace_instance[0]
                            pred_lm = predictor.lm
                            adapter = self.adapter[pred_lm] or settings.adapter or ChatAdapter()
                            assert isinstance(adapter, ChatAdapter), f"Adapter {adapter} is not a ChatAdapter. GRPO training is not supported for this adapter."
                            # TODO(Lakshya): Currently we exclude demos from the training data
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
                # assert len(predictor_train_batch) == self.per_predictor_batch_size, f"Batch size {len(predictor_train_batch)} does not match the expected batch size {self.per_predictor_batch_size}"
                for grpo_train_group in predictor_train_batch:
                    assert len(grpo_train_group) == self.num_rollouts_per_dspy_example_per_step, f"Number of completions {len(example_training_data['completions'])} does not match the expected number self.num_rollouts_per_dspy_example_per_step={self.num_rollouts_per_dspy_example_per_step}"
                    if len(set(map(repr, grpo_train_group))) < 2:
                        logger.warning(f"GRPOGroup has no diversity. This could be due to low temperature, or low number of rollouts, or the cache could be enabled inadvertently. The GRPOGroup is {grpo_train_group}.")

            logger.info("Invoking GRPO training step on the arbor backend...")
            if self.multitask:
                train_batch: GRPOBatch = sum(train_batch_per_predictor, [])
                grpo_training_job.run_grpo_step(train_batch)
            else:
                # TODO (Lakshya): Implement multitask==False
                for pred_id, grpo_training_job in enumerate(grpo_training_jobs):
                    train_batch: GRPOBatch = train_batch_per_predictor[pred_id]
                    grpo_training_job.run_grpo_step(train_batch)
            
            logger.info(f"GRPO training step {train_step_idx + 1}/{self.num_train_steps} completed.")
            
            if valset and ((train_step_idx + 1) % self.num_steps_for_val == 0 or train_step_idx + 1 == self.num_train_steps):
                logger.info(f"Evaluating the student program on the validation set after training step {train_step_idx + 1}/{self.num_train_steps}")
                valset_evaluation = valset_evaluator(student, metric=self.metric)
                logger.info(f"Student program validation set score after training step {train_step_idx + 1}/{self.num_train_steps}: {valset_evaluation}")
        
        logger.info("GRPO compiler has finished compiling the student program")
        student._compiled = True
        return student

def bootstrap_trace_data(
    program: Program,
    dataset: List[Example],
    metric: Optional[Callable] = None,
    num_threads=6,
    num_samples_per_input: int = 1,
) -> List[Dict[str, Any]]:
    # Return a list of dicts with the following keys: example_ind, example, prediction, trace, and score
    # (if metric != None)

    dataset_len = len(dataset)

    # TODO: deepcopy is used because copy does not set input keys
    dataset = [deepcopy(example) for example in dataset for _ in range(num_samples_per_input)]

    evaluator = Evaluate(
        devset=dataset,
        num_threads=num_threads,
        display_progress=True,
        return_outputs=True,
        provide_traceback=True,  # TODO(check with team)
    )

    def wrapped_metric(example, prediction, trace=None):
        prediction, _ = prediction
        return metric(example, prediction, trace) if metric else True

    def wrapped_program(**kwargs):
        with dspy.context(trace=[]):
            return program(**kwargs), dspy.settings.trace.copy()

    # TODO(Lakshya): How to ensure that the num_samples_per_input generations from the same example are different? Set lm temp, or disable cache.
    _, outputs = evaluator(wrapped_program, metric=wrapped_metric)

    data = [[] for _ in range(dataset_len)]
    for example_ind, (example, prediction, score) in enumerate(outputs):
        example_ind = example_ind % dataset_len
        prediction, trace = prediction
        data_dict = dict(example=example, prediction=prediction, trace=trace, example_ind=example_ind)
        if metric:
            data_dict["score"] = score
        data[example_ind].append(data_dict)

    return data

# Note: Shared below are useful functions for preparing student/teacher programs
# Similar methods are implemented separately and used by other DSPy
# teleprompters. These can be moved to shared locations.
def set_missing_predictor_lms(program: Program) -> Program:
    # If the predictors do not have LMs, set them to the global LM
    for pred in program.predictors():
        if not pred.lm:
            pred.lm = dspy.settings.lm
        
        # TODO(Lakshya, Omar): Check if this is the right place to disable cache
        pred.lm.cache = False

    return program

def prepare_student(student: Program) -> Program:
    if getattr(student, "_compiled", False):
        raise ValueError("The student program should not be compiled.")

    # TODO: Should we use reset_copy here? How would it affect the student
    # program's predictor LMs, if they are set?
    student = student.deepcopy()
    return student


def prepare_teacher(student: Program, teacher: Program = None) -> Program:
    if teacher is None:
        return student.deepcopy()
    else:
        teacher = teacher.deepcopy()

    # Ensuring that the student and teacher are are structurally equivalent
    assert_structural_equivalency(student, teacher)

    # Ensuring that the student and teacher programs do not share predictors
    assert_no_shared_predictor(student, teacher)

    return teacher


def assert_structural_equivalency(program1: object, program2: object):
    assert isinstance(program1, Program)
    assert isinstance(program2, Program)

    num1 = len(program1.predictors())
    num2 = len(program2.predictors())
    err = f"Structurally equivalent programs must have the the number of predictors. The number of predictors for the two modules do not match: {num1} != {num2}"
    assert num1 == num2, err

    pzip = zip(program1.named_predictors(), program2.named_predictors())
    for ind, ((name1, pred1), (name2, pred2)) in enumerate(pzip):
        err = f"Program predictor names must match at  corresponding indices for structural equivalency. The predictor names for the programs do not match at index {ind}: '{name1}' != '{name2}'"
        assert name1 == name2, err
        assert isinstance(pred1, Predict)
        assert isinstance(pred2, Predict)


def assert_no_shared_predictor(program1: Program, program2: Program):
    id_to_name1 = {id(p): n for n, p in program1.named_predictors()}
    id_to_name2 = {id(p): n for n, p in program2.named_predictors()}
    shared_ids = set(id_to_name1.keys()) & set(id_to_name2.keys())

    pred_names = ", ".join(id_to_name1[id] for id in shared_ids)
    err = f"The programs share the following predictor(s) with each other: {pred_names}"
    assert not shared_ids, err


def get_unique_lms(program: Program) -> List[LM]:
    lms = [pred.lm for pred in program.predictors()]
    return list(set(lms))


def launch_lms(program: Program):
    lms = get_unique_lms(program)
    for lm in lms:
        lm.launch()


def kill_lms(program: Program):
    lms = get_unique_lms(program)
    for lm in lms:
        lm.kill()
