import logging
import random
import time
from collections import Counter, deque
from typing import Any, Callable, Literal

from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.xml_adapter import XMLAdapter
from dspy.clients.lm import LM
from dspy.clients.utils_finetune import GRPOGroup, GRPOStatus, TrainDataFormat
from dspy.dsp.utils.settings import settings
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.teleprompt.bootstrap_finetune import (
    FinetuneTeleprompter,
    all_predictors_have_lms,
    assert_structural_equivalency,
)
from dspy.teleprompt.bootstrap_trace import FailedPrediction, bootstrap_trace_data

logger = logging.getLogger(__name__)


class GRPO(FinetuneTeleprompter):
    def __init__(
        self,
        metric: Callable | None = None,
        multitask: bool = True,
        train_kwargs: dict[str, Any] | dict[LM, dict[str, Any]] | None = None,
        adapter: Adapter | dict[LM, Adapter] | None = None,
        exclude_demos: bool = False,
        num_threads: int = 6,
        num_train_steps: int = 100,
        seed: int = 0,
        num_dspy_examples_per_grpo_step: int = 1,
        num_rollouts_per_grpo_step: int = 1,
        use_train_as_val: bool = False,
        num_steps_for_val: int = 5,
        report_train_scores: bool = False,
        failure_score: float = 0,
        format_failure_score: float = -1,
        variably_invoked_predictor_grouping_mode: Literal["truncate"] | Literal["fill"] | Literal["ragged"] = "truncate",
        variably_invoked_predictor_fill_strategy: Literal["randint"] | Literal["max"] | None = None,
    ):
        super().__init__(train_kwargs=train_kwargs)
        self.metric = metric
        self.multitask = multitask
        self.adapter: dict[LM, Adapter] = self.convert_to_lm_dict(adapter)
        self.exclude_demos = exclude_demos
        self.num_threads = num_threads
        self.num_train_steps = num_train_steps
        self.rng = random.Random(seed)
        self.num_dspy_examples_per_grpo_step = num_dspy_examples_per_grpo_step
        self.num_rollouts_per_grpo_step = num_rollouts_per_grpo_step
        self.use_train_as_val = use_train_as_val
        self.num_steps_for_val = num_steps_for_val
        self.report_train_scores = report_train_scores
        self.failure_score = failure_score
        self.format_failure_score = format_failure_score

        assert failure_score > format_failure_score, "failure_score must be greater than format_failure_score since the range [format_failure_score, failure_score] is used to provide dspy formatting rewards"

        if self.use_train_as_val:
            assert report_train_scores, "If use_train_as_val is True, report_train_scores must be True."

        assert exclude_demos, "exclude_demos==False is not supported yet. Please set it to True."
        assert multitask, "independent GRPO training jobs for each predictor in the student program is not supported yet. Please set multitask=True."

        # The backend will be called with a batch of (num_dspy_examples_per_grpo_step * num_rollouts_per_grpo_step * num_predictors) per training set if multitask is True
        # If multitask is False, the backend will be called with a batch of (num_dspy_examples_per_grpo_step * num_rollouts_per_grpo_step) per training job
        self.variably_invoked_predictor_grouping_mode = variably_invoked_predictor_grouping_mode
        if variably_invoked_predictor_grouping_mode == "fill":
            assert variably_invoked_predictor_fill_strategy is not None, "variably_invoked_predictor_fill_strategy must be set when variably_invoked_predictor_grouping_mode is 'fill'"
            assert variably_invoked_predictor_fill_strategy in ["randint", "max"], "variably_invoked_predictor_fill_strategy must be either 'randint' or 'max'"
        self.variably_invoked_predictor_fill_strategy = variably_invoked_predictor_fill_strategy

        self.shuffled_trainset_ids = []
        self.epoch = -1
        self.id_freqs = Counter()
        self.fulfilled_batch_ids = []
        self.pending_batch_ids = []

    def validate_trace_data_and_log_issues(
        self,
        trace_data: list[list[list[dict[str, Any]]]],
        subsample_training_dataset: list[Example],
        num_teachers: int,
        num_samples_per_input: int,
        pred_signature_hash_to_ind: dict[int, int],
    ):
        # At this point, trace_data: list[example_idx -> list[teacher_idx -> [num_samples_per_input * Dict(example, prediction, trace, example_ind, score)]]]
        # Shape of trace is: [dspy_module_invocation_idx -> Tuple[Predictor, PredictorInputs, Prediction]]
        assert len(trace_data) == len(subsample_training_dataset), f"Trace data length {len(trace_data)} does not match the number of examples {len(subsample_training_dataset)}"
        assert len(trace_data[0]) == num_teachers, f"Trace data length {len(trace_data[0])} does not match the number of teachers {num_teachers}"
        # TODO(GRPO Team): Ideally, once the dspy format issue is fixed, this change should be reverted back to being a normal assert.
        if len(trace_data[0][0]) == 0:
            logger.warning(f"Trace data for example {0} and teacher {0} is empty. This is likely due to all examples in the training set input, resulting in the model generating output not following the dspy response format.")
        elif len(trace_data[0][0]) != num_samples_per_input:
            logger.warning(f"Trace data length {len(trace_data[0][0])} does not match the expected number of samples per input {num_samples_per_input}")
            assert "trace" in trace_data[0][0][0], "Trace data does not contain the 'trace' key"
            assert len(trace_data[0][0][0]["trace"]) > 0, "Trace data is empty"
            assert len(trace_data[0][0][0]["trace"][0]) == 3, f"Trace tuple length {len(trace_data[0][0][0]['trace'][0])} does not match the expected length 3"

        for example_data in trace_data:
            for teacher_data in example_data:
                for sample in teacher_data:
                    for t in sample["trace"]:
                        assert hash(t[0].signature) in pred_signature_hash_to_ind

    def report_validation_metrics(self, student, trainset, valset, logger, step_idx=-1):
        if step_idx == -1 or step_idx == self.num_train_steps - 1 or (step_idx + 1) % self.num_steps_for_val == 0:
            pass
        else:
            return

        if valset is not None:
            # Validation set provided by user
            assert not self.use_train_as_val, "If valset is provided, use_train_as_val must be False."
            assert isinstance(self.num_steps_for_val, int) and self.num_steps_for_val > 0, "num_steps_for_val must be a positive integer."
            if self.report_train_scores:
                if step_idx == -1:
                    logger.info("Using user provided validation set and reporting train scores for every validation step in addition.")
                valset_evaluator = Evaluate(
                    devset=valset + trainset,
                    num_threads=self.num_threads,
                    display_progress=True,
                    provide_traceback=False,  # TODO(check with team)
                    max_errors=len(valset)*10,  # TODO(check with team)
                    failure_score=self.failure_score
                )
                if step_idx == -1:
                    logger.info("Evaluating the student program on the train+validation set before training loop...")
                else:
                    logger.info(f"Evaluating the student program on the validation set after training step {step_idx + 1}/{self.num_train_steps}")
                valset_evaluation = valset_evaluator(student, metric=self.metric)
                trainset_scores = [r[-1] for r in valset_evaluation.results[len(valset):]]
                valset_scores = [r[-1] for r in valset_evaluation.results[:len(valset)]]
                trainset_agg = sum(trainset_scores) / len(trainset_scores)
                valset_agg = sum(valset_scores) / len(valset_scores)
                if step_idx == -1:
                    logger.info(f"Student program training set score before training loop: {trainset_agg}")
                    logger.info(f"Student program validation set score before training loop: {valset_agg}")
                else:
                    logger.info(f"Student program training set score after training step {step_idx + 1}/{self.num_train_steps}: {trainset_agg}")
                    logger.info(f"Student program validation set score after training step {step_idx + 1}/{self.num_train_steps}: {valset_agg}")
            else:
                if step_idx == -1:
                    logger.info("Using user provided validation set and not reporting train scores.")
                valset_evaluator = Evaluate(
                    devset=valset,
                    num_threads=self.num_threads,
                    display_progress=True,
                    provide_traceback=False,  # TODO(check with team)
                    max_errors=len(valset)*10,  # TODO(check with team)
                    failure_score=self.failure_score
                )
                if step_idx == -1:
                    logger.info("Evaluating the student program on the validation set before training loop...")
                else:
                    logger.info(f"Evaluating the student program on the validation set after training step {step_idx + 1}/{self.num_train_steps}")
                valset_evaluation = valset_evaluator(student, metric=self.metric)
                if step_idx == -1:
                    logger.info(f"Student program validation set score before training loop: {valset_evaluation.score}")
                else:
                    logger.info(f"Student program validation set score after training step {step_idx + 1}/{self.num_train_steps}: {valset_evaluation.score}")
        else:
            # No validation set provided by user
            if self.report_train_scores:
                assert self.use_train_as_val, "If report_train_scores is True, use_train_as_val must be True when valset is not provided explicitly."
                assert isinstance(self.num_steps_for_val, int) and self.num_steps_for_val > 0, "num_steps_for_val must be a positive integer."
                if step_idx == -1:
                    logger.info("Using trainset as validation set.")
                valset_evaluator = Evaluate(
                    devset=trainset,
                    num_threads=self.num_threads,
                    display_progress=True,
                    provide_traceback=False,  # TODO(check with team)
                    max_errors=len(trainset)*10,  # TODO(check with team)
                    failure_score=self.failure_score
                )
                if step_idx == -1:
                    logger.info("Evaluating the student program on the validation set before training loop...")
                else:
                    logger.info(f"Evaluating the student program on the validation set after training step {step_idx + 1}/{self.num_train_steps}")
                valset_evaluation = valset_evaluator(student, metric=self.metric)
                if step_idx == -1:
                    logger.info(f"Student program training set score before training loop: {valset_evaluation.score}")
                else:
                    logger.info(f"Student program training set score after training step {step_idx + 1}/{self.num_train_steps}: {valset_evaluation.score}")
            else:
                # No valset provided, and not using train as val
                assert not self.use_train_as_val, "If report_train_scores is False, use_train_as_val must be False."
                if step_idx == -1:
                    logger.info("Not using any validation set and not reporting train scores.")

    def update_shuffled_trainset(self, original_trainset):
        self.shuffled_trainset_ids = list(range(len(original_trainset)))
        self.rng.shuffle(self.shuffled_trainset_ids)
        for id in self.shuffled_trainset_ids:
            self.id_freqs[id] += 1

        num_to_pad = self.num_dspy_examples_per_grpo_step - (len(original_trainset) % self.num_dspy_examples_per_grpo_step)
        if num_to_pad > 0:
            # Select ids based on least frequent ids
            for _ in range(num_to_pad):
                selected_id = self.id_freqs.most_common()[::-1][0][0]
                self.shuffled_trainset_ids.append(selected_id)
                self.id_freqs[selected_id] += 1

    def select_training_sample_and_update_shuffled_trainset(
        self,
        original_trainset: list[Example],
        train_step_idx: int,
    ) -> list[Example]:
        base_idx = train_step_idx * self.num_dspy_examples_per_grpo_step
        if self.epoch == -1:
            curr_epoch = 0
        else:
            curr_epoch = base_idx // len(self.shuffled_trainset_ids)
        if curr_epoch > self.epoch:
            logger.info(f"Updating shuffled trainset for epoch {curr_epoch}...")
            self.epoch = curr_epoch
            self.update_shuffled_trainset(original_trainset)

        assert len(self.shuffled_trainset_ids) >= self.num_dspy_examples_per_grpo_step, f"Shuffled trainset length {len(self.shuffled_trainset_ids)} is less than num_dspy_examples_per_grpo_step {self.num_dspy_examples_per_grpo_step}"
        assert len(self.shuffled_trainset_ids) % self.num_dspy_examples_per_grpo_step == 0, f"Shuffled trainset length {len(self.shuffled_trainset_ids)} is not divisible by num_dspy_examples_per_grpo_step {self.num_dspy_examples_per_grpo_step}"

        base_idx = base_idx % len(self.shuffled_trainset_ids)
        end_idx = base_idx + self.num_dspy_examples_per_grpo_step
        assert end_idx <= len(self.shuffled_trainset_ids), f"End index {end_idx} is out of bounds for shuffled trainset length {len(self.shuffled_trainset_ids)}"
        selected_ids = self.shuffled_trainset_ids[base_idx:end_idx]
        selected_trainset = [original_trainset[i] for i in selected_ids]
        return selected_trainset

    def compile(
        self,
        student: Module,
        trainset: list[Example],
        teacher: Module | list[Module] | None = None,
        valset: list[Example] | None = None,
        **kwargs,
    ) -> Module:
        logger.info("Starting the GRPO compilation process... The LM(s) for the student program will be updated in place at the end of the training.")
        logger.info("Validating the inputs...")

        assert len(trainset) > 0, "Training set is empty. Please provide a non-empty training set."

        if len(trainset) < self.num_dspy_examples_per_grpo_step:
            logger.warning(
            f"Number of training examples {len(trainset)} is less than the number of examples per GRPO step {self.num_dspy_examples_per_grpo_step}. "
                "Repeating the training set to fill the GRPO step. This could lead to overfitting and training instability."
            )
            multiplier = (self.num_dspy_examples_per_grpo_step + len(trainset) - 1) // len(trainset)
            if multiplier > 1:
                logger.warning(
                    f"Repeating the training set {multiplier} times to fill the GRPO step. This could lead to overfitting and training instability."
                )
                trainset = trainset * multiplier

        # TODO(GRPO Team): Following checks are for unimplemented features.
        # Consider if we want to eventually implement them or remove. We don't
        # yet support:
        # * multitask == False
        # * student program with multiple predictor LMs
        # The main reason for these is that we update the LMs in place. If these
        # LMs are shared between the different predictors of the student
        # program and we have multitask == False, we need to decide which steps
        # will use new LM copies and we need to ensure our decision is
        # consistent with any teacher LMs that share the same LMs.
        # TODO(GRPO Team): We want to make it possible to continue GRPO runs in
        # the future by saving the state of the GRPO run in the event of a
        # process failure.
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
        pred_signature_hash_to_ind = {hash(pred.signature): ind for ind, pred in enumerate(student.predictors())}
        num_student_predictors = len(student.predictors())

        logging.info("Preparing the teacher program(s)... We will ensure that the provided programs have the same program structure as the student program.")
        if (isinstance(teacher, list) and len(teacher) == 0) or teacher is None:
            teacher = student
        teachers = teacher if isinstance(teacher, list) else [teacher]
        for t in teachers:
            assert_structural_equivalency(student, t)
            all_predictors_have_lms(t)

        # Ensure that the teachers list contain the student program
        assert student in teachers, f"Student program {student} is not in the list of teachers {teachers}. Please provide the student program as one of the teachers. Alternatively, you can leave the teacher argument as None, and the student program will be used as the teacher program."
        assert self.num_rollouts_per_grpo_step % len(teachers) == 0, (
            f"The GRPO group size (num_rollouts_per_grpo_step) {self.num_rollouts_per_grpo_step} is not divisible by the number of teachers {len(teachers)}. "
            "This is required to ensure that each teacher gets the same number of examples."
            "Please provide a number of examples that is divisible by the number of teachers."
        )
        num_samples_per_input = self.num_rollouts_per_grpo_step // len(teachers)

        # We will disable the LM cache for all programs (student and teachers)
        # These will be reverted to their original state at the end of the
        # training
        lm_cache_dict = {}
        disable_lm_cache(program=student, lm_cache_dict=lm_cache_dict)
        for t in teachers:
            disable_lm_cache(program=t, lm_cache_dict=lm_cache_dict)

        # Update train_kwargs
        for pred in student.predictors():
            train_kwargs = self.train_kwargs[pred.lm]
            train_kwargs = {} if train_kwargs is None else train_kwargs
            train_kwargs["num_generations"] = self.num_rollouts_per_grpo_step
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

        self.report_validation_metrics(
            student=student,
            trainset=trainset,
            valset=valset,
            logger=logger,
            step_idx=-1,
        )

        # Queue of GRPO groups per training job; key is (LM, data_key)
        group_queues = {}
        logger.info("Starting the GRPO training loop...")
        for train_step_idx in range(self.num_train_steps):
            logger.info(f"GRPO training step {train_step_idx + 1}/{self.num_train_steps}...")

            subsample_training_dataset = self.select_training_sample_and_update_shuffled_trainset(
                original_trainset=trainset,
                train_step_idx=train_step_idx,
            )
            def _any_available_for_step():
                for _, job in grpo_training_jobs.items():
                    grpo_status: GRPOStatus = job.get_status()
                    pending_batch_ids = grpo_status["pending_batch_ids"]
                    available = set(pending_batch_ids) - set(self.fulfilled_batch_ids)
                    if available:
                        return True
                return False

            while not _any_available_for_step():
                time.sleep(1)

            logger.info("Bootstrapping data...")
            trace_data = [[[] for _ in range(len(teachers))] for _ in range(len(subsample_training_dataset))]
            for tind, teacher in enumerate(teachers):
                subsample_training_dataset_repeated = [example for _ in range(num_samples_per_input) for example in subsample_training_dataset]
                round_data = bootstrap_trace_data(
                    program=teacher,
                    dataset=subsample_training_dataset_repeated,
                    metric=self.metric,
                    num_threads=self.num_threads,
                    raise_on_error=False, # TODO(GRPO Team): This should be True, once the dspy format issue is fixed
                    capture_failed_parses=True,
                    failure_score=self.failure_score,
                    format_failure_score=self.format_failure_score,
                    log_format_failures=True,
                )
                for data_dict in round_data:
                    example_ind_in_subsample = data_dict["example_ind"] % len(subsample_training_dataset)
                    data_dict["example_ind"] = example_ind_in_subsample
                    trace_data[example_ind_in_subsample][tind].append(data_dict)

            # The trace_data for examples with FailedPrediction cases will have the signature at index 0, instead of the predictor
            # We need to replace the signature with the predictor

            # At this point, trace_data: list[example_idx -> list[teacher_idx -> [num_samples_per_input * Dict(example, prediction, trace, example_ind, score)]]]
            # Shape of trace is: [dspy_module_invocation_idx -> Tuple[Predictor, PredictorInputs, Prediction]]
            self.validate_trace_data_and_log_issues(
                trace_data=trace_data,
                subsample_training_dataset=subsample_training_dataset,
                num_teachers=len(teachers),
                num_samples_per_input=num_samples_per_input,
                pred_signature_hash_to_ind=pred_signature_hash_to_ind,
            )

            logger.info("Preparing the training data batch from bootstrapped examples for GRPO...")
            # Now, we need to prepare batches of data to be sent for training
            # Shape of train_batch_per_predictor: list[num_student_predictors -> list[ ]]
            train_batch_per_predictor: list[list[GRPOGroup]] = [[] for _ in range(num_student_predictors)]
            for pred_id in range(num_student_predictors):
                for example_ind, example_data in enumerate(trace_data):
                    # Each example_data is a list of teacher_idx -> [num_samples_per_input * Dict(example, prediction, trace, example_ind, score)]
                    # We need to flatten this list and create a batch for each predictor

                    # TODO(Lakshya, Omar, Noah): Discuss what to do with the same module being invoked multiple times within a single dspy.Example
                    predictor_example_invocations: list[list[tuple]] = []

                    for teacher_data in example_data:
                        for sample in teacher_data:
                            # Each sample is a Dict(example, prediction, trace, example_ind, score)
                            # sample['prediction'] is module_level prediction
                            assert sample["example_ind"] == example_ind, f"Example index {sample['example_ind']} does not match the expected index {example_ind}"

                            trace_instances_for_current_pred = [(*t, sample["score"]) for t in sample["trace"] if hash(t[0].signature) == hash(student.predictors()[pred_id].signature)]

                            predictor_example_invocations.append(trace_instances_for_current_pred)

                    if len(predictor_example_invocations) == 0:
                        logger.warning(f"Skipping example {example_ind} for predictor {pred_id} as it has no invocations. This is likely due to all examples in the training set input, resulting in the model generating output not following the dspy response format.")
                        continue
                    elif len(predictor_example_invocations) != self.num_rollouts_per_grpo_step:
                        logger.warning(f"Number of predictor example invocations {len(predictor_example_invocations)} does not match the expected batch size {self.num_rollouts_per_grpo_step}. This is likely due to all examples in the training set input, resulting in the model generating output not following the dspy response format.")

                    min_len = min([len(predictor_example_invocations[i]) for i in range(len(predictor_example_invocations))])
                    max_len = max([len(predictor_example_invocations[i]) for i in range(len(predictor_example_invocations))])
                    if min_len == 0:
                        logger.warning(f"Skipping example {example_ind} for predictor {pred_id} as it has no invocations.")
                        continue

                    if self.variably_invoked_predictor_grouping_mode == "truncate":
                        predictor_example_invocations = [invocation[:min_len] for invocation in predictor_example_invocations]
                    elif self.variably_invoked_predictor_grouping_mode == "fill":
                        if self.variably_invoked_predictor_fill_strategy == "randint":
                            selector = lambda l: self.rng.choice(l) # noqa: E731, E741
                        else:
                            selector = lambda l: l[-1] # noqa: E731, E741
                        predictor_example_invocations = [
                            invocation + [selector(invocation) for _ in range(max_len - len(invocation))]
                            for invocation in predictor_example_invocations
                        ]
                    else:
                        assert self.variably_invoked_predictor_grouping_mode == "ragged", f"Unknown variably invoked predictor grouping mode {self.variably_invoked_predictor_grouping_mode}"
                    max_len = max([len(predictor_example_invocations[i]) for i in range(len(predictor_example_invocations))])

                    example_training_data: list[GRPOGroup] = [[] for _ in range(max_len)]

                    for group_idx in range(max_len):
                        for rollout_idx in range(len(predictor_example_invocations)):
                            trace_instance = predictor_example_invocations[rollout_idx][group_idx]
                            score = trace_instance[3]
                            # for module_invocation_idx, trace_instance in enumerate(trace_instances_for_current_pred):
                            # Each trace is a tuple of (Predictor, PredictorInputs, Prediction)
                            trace_pred_id = pred_signature_hash_to_ind.get(hash(trace_instance[0].signature))
                            assert trace_pred_id == pred_id

                            predictor = trace_instance[0]
                            pred_lm = predictor.lm
                            adapter = self.adapter[pred_lm] or settings.adapter or XMLAdapter()
                            assert isinstance(adapter, ChatAdapter), f"Adapter {adapter} is not a ChatAdapter. GRPO training is not supported for this adapter."
                            # TODO(Lakshya): Currently we exclude demos from the training data
                            # TODO(GRPO Team): Use build_call_data_from_trace (from bootstrap_finetune) instead of
                            # dealing with the message formatting ourselves.
                            inp_messages = adapter.format(
                                signature=trace_instance[0].signature,
                                inputs=trace_instance[1],
                                demos=[] # TODO: Add support for demos
                            )

                            if isinstance(trace_instance[2], FailedPrediction):
                                score = trace_instance[2].format_reward or self.format_failure_score
                                example_training_data[group_idx].append({
                                    "messages": inp_messages,
                                    "completion": {
                                        "role": "assistant",
                                        "content": trace_instance[2].completion_text,
                                    },
                                    "reward": float(score),
                                })
                                logger.warning(f"Adding a format failure example to the training data for predictor {pred_id} and example {example_ind}.")
                            else:
                                all_messages = adapter.format_finetune_data(
                                    signature=trace_instance[0].signature,
                                    inputs=trace_instance[1],
                                    outputs=trace_instance[2],
                                    demos=[] # TODO: Add support for demos
                                )["messages"]

                                assert all_messages[:-1] == inp_messages, f"Input messages {inp_messages} do not match the expected messages {all_messages[:-1]}"

                                example_training_data[group_idx].append({
                                    "messages": inp_messages,
                                    "completion": {
                                        "role": all_messages[-1]["role"],
                                        "content": all_messages[-1]["content"],
                                    },
                                    "reward": float(score),
                                })

                    train_batch_per_predictor[pred_id].extend(example_training_data)

            if not any(train_batch_per_predictor):
                logger.warning("No training data found for this training step. This means that the model did not generate valid formatted responses for any of the examples in the training set. This is a critical error. Please check the model and the training set.")
                continue

            for predictor_train_batch in train_batch_per_predictor:
                for grpo_train_group in predictor_train_batch:
                    if len(grpo_train_group) != self.num_rollouts_per_grpo_step:
                        logger.warning(f"Number of completions {len(grpo_train_group)} does not match the expected number num_rollouts_per_grpo_step={self.num_rollouts_per_grpo_step}")
                        assert len(grpo_train_group) <= self.num_rollouts_per_grpo_step, f"Number of completions {len(grpo_train_group)} is greater than the expected number num_rollouts_per_grpo_step={self.num_rollouts_per_grpo_step}"
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
            for (lm_for_job, data_key), job in grpo_training_jobs.items():
                train_data: list[GRPOGroup] = sum(train_batch_per_predictor, []) if data_key is None else train_batch_per_predictor[data_key] #noqa: RUF017
                for group in train_data:
                    if len(group) != self.num_rollouts_per_grpo_step:
                        # TODO(GRPO Team): This is very undesirable. This occurs only because in some of the generations, the model does not follow the correct dspy format.
                        # The ideal solution is to identify the full response string in that predictor's group, and then assign a high-negative (user-configurable) reward to that group.
                        # Pad the group to the expected number of generations by repeating the whole group, might require multiple iterations
                        while len(group) < self.num_rollouts_per_grpo_step:
                            group.extend(group[:min(self.num_rollouts_per_grpo_step - len(group), len(group))])
                    assert len(group) == self.num_rollouts_per_grpo_step, f"Number of completions {len(group)} does not match the expected number self.num_rollouts_per_grpo_step={self.num_rollouts_per_grpo_step}"

                # Determine available batch IDs for this specific job
                grpo_status: GRPOStatus = job.get_status()
                pending_batch_ids = grpo_status["pending_batch_ids"]
                available_batch_ids = list(set(pending_batch_ids) - set(self.fulfilled_batch_ids))
                if not available_batch_ids:
                    continue

                # Initialize and (re)fill the queue for this job as needed
                job_key = (lm_for_job, data_key)
                q = group_queues.setdefault(job_key, deque())

                # Refill strategy: add randomized copies of current train_data until we can satisfy all batch_ids
                if len(q) < len(available_batch_ids) and len(train_data) > 0:
                    need = len(available_batch_ids) - len(q)
                    while need > 0:
                        # Shuffle by sampling without replacement
                        shuffled = self.rng.sample(train_data, k=len(train_data))
                        q.extend(shuffled)
                        need -= len(shuffled)

                # Build GRPOGroup items by popping from the queue; fallback to random selection if needed
                final_train_data: list[GRPOGroup] = []
                for bid in available_batch_ids:
                    if q:
                        grp = q.popleft()
                    else:
                        # Fallback: choose randomly from current train_data (or flattened pool) if queue underflows
                        fallback_pool = train_data if len(train_data) > 0 else sum(train_batch_per_predictor, [])
                        if len(fallback_pool) == 0:
                            # Nothing to send for this job
                            continue
                        grp = self.rng.choice(fallback_pool)
                    final_train_data.append({"batch_id": bid, "group": grp})

                if not final_train_data:
                    continue

                # Track fulfilled IDs to avoid reuse
                self.fulfilled_batch_ids.extend([item["batch_id"] for item in final_train_data])

                job.step(train_data=final_train_data, train_data_format=TrainDataFormat.GRPO_CHAT)

            logger.info(f"GRPO training step {train_step_idx + 1}/{self.num_train_steps} completed.")

            self.report_validation_metrics(
                student=student,
                trainset=trainset,
                valset=valset,
                logger=logger,
                step_idx=train_step_idx,
            )

        logger.info("Done with the iterations! Retrieving the final model(s)...")
        for _, job in grpo_training_jobs.items():
            job.terminate()

        # Revert cache states to their initial values
        recover_lm_cache(program=student, lm_cache_dict=lm_cache_dict)
        for t in teachers:
            recover_lm_cache(program=t, lm_cache_dict=lm_cache_dict)

        logger.info("GRPO compiler has finished compiling the student program")
        student._compiled = True
        return student


def disable_lm_cache(program: Module, lm_cache_dict: dict):
    """Disable the LM cache for all predictors in the program."""
    for pred in program.predictors():
        if not pred.lm:
            raise ValueError(f"Cannot disable cache: predictor {pred} does not have an LM set.")
        if pred.lm not in lm_cache_dict:  # Check to avoid overwriting the cache
            lm_cache_dict[pred.lm] = pred.lm.cache
        pred.lm.cache = False


def recover_lm_cache(program: Module, lm_cache_dict: dict):
    """Recover the LM caches for all predictors in the program to their original state."""
    for pred in program.predictors():
        if pred.lm in lm_cache_dict:
            pred.lm.cache = lm_cache_dict[pred.lm]
        else:
            # We do not expect this branch to execute at all since all the LMs
            # are modified in place and no new LMs are created during training.
            # However, we do not complain if this happens since this is a
            # relatively minor feature. We default the LM cache to True.
            pred.lm.cache = True
