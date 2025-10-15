from __future__ import annotations

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


class ArborGRPO(FinetuneTeleprompter):
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

    # The rest mirrors GRPO, but remains independent to avoid tight coupling
    def validate_trace_data_and_log_issues(
        self,
        trace_data: list[list[list[dict[str, Any]]]],
        subsample_training_dataset: list[Example],
        num_teachers: int,
        num_samples_per_input: int,
        pred_signature_hash_to_ind: dict[int, int],
    ):
        assert len(trace_data) == len(subsample_training_dataset)
        assert len(trace_data[0]) == num_teachers
        if len(trace_data[0][0]) == 0:
            logger.warning(
                f"Trace data for example {0} and teacher {0} is empty. This is likely due to all examples in the training set input, resulting in the model generating output not following the dspy response format."
            )
        elif len(trace_data[0][0]) != num_samples_per_input:
            logger.warning(
                f"Trace data length {len(trace_data[0][0])} does not match the expected number of samples per input {num_samples_per_input}"
            )
            assert "trace" in trace_data[0][0][0]
            assert len(trace_data[0][0][0]["trace"]) > 0
            assert len(trace_data[0][0][0]["trace"][0]) == 3

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
            assert not self.use_train_as_val
            assert isinstance(self.num_steps_for_val, int) and self.num_steps_for_val > 0
            if self.report_train_scores:
                if step_idx == -1:
                    logger.info("Using user provided validation set and reporting train scores for every validation step in addition.")
                valset_evaluator = Evaluate(
                    devset=valset + trainset,
                    num_threads=self.num_threads,
                    display_progress=True,
                    provide_traceback=False,
                    max_errors=len(valset) * 10,
                    failure_score=self.failure_score,
                )
                if step_idx == -1:
                    logger.info("Evaluating the student program on the train+validation set before training loop...")
                else:
                    logger.info(
                        f"Evaluating the student program on the validation set after training step {step_idx + 1}/{self.num_train_steps}"
                    )
                valset_evaluation = valset_evaluator(student, metric=self.metric)
                trainset_scores = [r[-1] for r in valset_evaluation.results[len(valset) :]]
                valset_scores = [r[-1] for r in valset_evaluation.results[: len(valset)]]
                trainset_agg = sum(trainset_scores) / len(trainset_scores)
                valset_agg = sum(valset_scores) / len(valset_scores)
                if step_idx == -1:
                    logger.info(f"Student program training set score before training loop: {trainset_agg}")
                    logger.info(f"Student program validation set score before training loop: {valset_agg}")
                else:
                    logger.info(
                        f"Student program training set score after training step {step_idx + 1}/{self.num_train_steps}: {trainset_agg}"
                    )
                    logger.info(
                        f"Student program validation set score after training step {step_idx + 1}/{self.num_train_steps}: {valset_agg}"
                    )
            else:
                if step_idx == -1:
                    logger.info("Using user provided validation set and not reporting train scores.")
                valset_evaluator = Evaluate(
                    devset=valset,
                    num_threads=self.num_threads,
                    display_progress=True,
                    provide_traceback=False,
                    max_errors=len(valset) * 10,
                    failure_score=self.failure_score,
                )
                if step_idx == -1:
                    logger.info("Evaluating the student program on the validation set before training loop...")
                else:
                    logger.info(
                        f"Evaluating the student program on the validation set after training step {step_idx + 1}/{self.num_train_steps}"
                    )
                valset_evaluation = valset_evaluator(student, metric=self.metric)
                if step_idx == -1:
                    logger.info(
                        f"Student program validation set score before training loop: {valset_evaluation.score}"
                    )
                else:
                    logger.info(
                        f"Student program validation set score after training step {step_idx + 1}/{self.num_train_steps}: {valset_evaluation.score}"
                    )
        else:
            if self.report_train_scores:
                assert self.use_train_as_val
                assert isinstance(self.num_steps_for_val, int) and self.num_steps_for_val > 0
                if step_idx == -1:
                    logger.info("Using trainset as validation set.")
                valset_evaluator = Evaluate(
                    devset=trainset,
                    num_threads=self.num_threads,
                    display_progress=True,
                    provide_traceback=False,
                    max_errors=len(trainset) * 10,
                    failure_score=self.failure_score,
                )
                if step_idx == -1:
                    logger.info("Evaluating the student program on the validation set before training loop...")
                else:
                    logger.info(
                        f"Evaluating the student program on the validation set after training step {step_idx + 1}/{self.num_train_steps}"
                    )
                valset_evaluation = valset_evaluator(student, metric=self.metric)
                if step_idx == -1:
                    logger.info(
                        f"Student program training set score before training loop: {valset_evaluation.score}"
                    )
                else:
                    logger.info(
                        f"Student program training set score after training step {step_idx + 1}/{self.num_train_steps}: {valset_evaluation.score}"
                    )
            else:
                assert not self.use_train_as_val
                if step_idx == -1:
                    logger.info("Not using any validation set and not reporting train scores.")

    def update_shuffled_trainset(self, original_trainset):
        self.shuffled_trainset_ids = list(range(len(original_trainset)))
        self.rng.shuffle(self.shuffled_trainset_ids)
        for id in self.shuffled_trainset_ids:
            self.id_freqs[id] += 1

        num_to_pad = self.num_dspy_examples_per_grpo_step - (
            len(original_trainset) % self.num_dspy_examples_per_grpo_step
        )
        if num_to_pad > 0:
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

        assert len(self.shuffled_trainset_ids) >= self.num_dspy_examples_per_grpo_step
        assert (
            len(self.shuffled_trainset_ids) % self.num_dspy_examples_per_grpo_step == 0
        )

        base_idx = base_idx % len(self.shuffled_trainset_ids)
        end_idx = base_idx + self.num_dspy_examples_per_grpo_step
        assert end_idx <= len(self.shuffled_trainset_ids)
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
        logger.info(
            "Starting the GRPO compilation process... The LM(s) for the student program will be updated in place at the end of the training."
        )
        logger.info("Validating the inputs...")

        assert len(trainset) > 0

        if len(trainset) < self.num_dspy_examples_per_grpo_step:
            logger.warning(
                f"Number of training examples {len(trainset)} is less than the number of examples per GRPO step {self.num_dspy_examples_per_grpo_step}. Repeating the training set to fill the GRPO step. This could lead to overfitting and training instability."
            )
            multiplier = (self.num_dspy_examples_per_grpo_step + len(trainset) - 1) // len(trainset)
            if multiplier > 1:
                logger.warning(
                    f"Repeating the training set {multiplier} times to fill the GRPO step. This could lead to overfitting and training instability."
                )
                trainset = trainset * multiplier

        if not self.multitask:
            raise ValueError(
                "Independent GRPO training jobs for each predictor in the student program are not supported yet. Please set multitask=True."
            )

        student_lms = {id(pred.lm) for pred in student.predictors()}
        assert len(student_lms) == 1, (
            f"Student program has multiple LMs: {student_lms}. "
            "GRPO only supports student programs with a single LM."
            "You can set the LM for a program with `program.set_lm(...)`"
        )

        if self.use_train_as_val:
            assert valset is None

        logger.info("Preparing the student program...")
        all_predictors_have_lms(student)
        pred_signature_hash_to_ind = {hash(pred.signature): ind for ind, pred in enumerate(student.predictors())}
        num_student_predictors = len(student.predictors())

        logging.info(
            "Preparing the teacher program(s)... We will ensure that the provided programs have the same program structure as the student program."
        )
        if (isinstance(teacher, list) and len(teacher) == 0) or teacher is None:
            teacher = student
        teachers = teacher if isinstance(teacher, list) else [teacher]
        for t in teachers:
            assert_structural_equivalency(student, t)
            all_predictors_have_lms(t)

        assert student in teachers
        assert self.num_rollouts_per_grpo_step % len(teachers) == 0
        num_samples_per_input = self.num_rollouts_per_grpo_step // len(teachers)

        lm_cache_dict = {}
        disable_lm_cache(program=student, lm_cache_dict=lm_cache_dict)
        for t in teachers:
            disable_lm_cache(program=t, lm_cache_dict=lm_cache_dict)

        for pred in student.predictors():
            train_kwargs = self.train_kwargs[pred.lm]
            train_kwargs = {} if train_kwargs is None else train_kwargs
            train_kwargs["num_generations"] = self.num_rollouts_per_grpo_step
            self.train_kwargs[pred.lm] = train_kwargs

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
                subsample_training_dataset_repeated = [
                    example for _ in range(num_samples_per_input) for example in subsample_training_dataset
                ]
                round_data = bootstrap_trace_data(
                    program=teacher,
                    dataset=subsample_training_dataset_repeated,
                    metric=self.metric,
                    num_threads=self.num_threads,
                    raise_on_error=False,
                    capture_failed_parses=True,
                    failure_score=self.failure_score,
                    format_failure_score=self.format_failure_score,
                    log_format_failures=True,
                )
                for data_dict in round_data:
                    example_ind_in_subsample = data_dict["example_ind"] % len(subsample_training_dataset)
                    data_dict["example_ind"] = example_ind_in_subsample
                    trace_data[example_ind_in_subsample][tind].append(data_dict)

            self.validate_trace_data_and_log_issues(
                trace_data=trace_data,
                subsample_training_dataset=subsample_training_dataset,
                num_teachers=len(teachers),
                num_samples_per_input=num_samples_per_input,
                pred_signature_hash_to_ind=pred_signature_hash_to_ind,
            )

            logger.info("Preparing the training data batch from bootstrapped examples for GRPO...")
            train_batch_per_predictor: list[list[GRPOGroup]] = [[] for _ in range(num_student_predictors)]
            for pred_id in range(num_student_predictors):
                for example_ind, example_data in enumerate(trace_data):
                    predictor_example_invocations: list[list[tuple]] = []

                    for teacher_data in example_data:
                        for sample in teacher_data:
                            assert sample["example_ind"] == example_ind

                            trace_instances_for_current_pred = [
                                (*t, sample["score"]) for t in sample["trace"] if hash(t[0].signature) == hash(student.predictors()[pred_id].signature)
                            ]

                            predictor_example_invocations.append(trace_instances_for_current_pred)

                    if len(predictor_example_invocations) == 0:
                        logger.warning(
                            f"Skipping example {example_ind} for predictor {pred_id} as it has no invocations. This is likely due to all examples in the training set input, resulting in the model generating output not following the dspy response format."
                        )
                        continue
                    elif len(predictor_example_invocations) != self.num_rollouts_per_grpo_step:
                        logger.warning(
                            f"Number of predictor example invocations {len(predictor_example_invocations)} does not match the expected batch size {self.num_rollouts_per_grpo_step}. This is likely due to all examples in the training set input, resulting in the model generating output not following the dspy response format."
                        )

                    min_len = min([len(predictor_example_invocations[i]) for i in range(len(predictor_example_invocations))])
                    max_len = max([len(predictor_example_invocations[i]) for i in range(len(predictor_example_invocations))])
                    if min_len == 0:
                        logger.warning(
                            f"Skipping example {example_ind} for predictor {pred_id} as it has no invocations."
                        )
                        continue

                    if self.variably_invoked_predictor_grouping_mode == "truncate":
                        predictor_example_invocations = [invocation[:min_len] for invocation in predictor_example_invocations]
                    elif self.variably_invoked_predictor_grouping_mode == "fill":
                        if self.variably_invoked_predictor_fill_strategy == "randint":
                            selector = lambda l: self.rng.choice(l)  # noqa: E731, E741
                        else:
                            selector = lambda l: l[-1]  # noqa: E731, E741
                        predictor_example_invocations = [
                            invocation
                            + [selector(invocation) for _ in range(max_len - len(invocation))]
                            for invocation in predictor_example_invocations
                        ]
                    else:
                        assert (
                            self.variably_invoked_predictor_grouping_mode == "ragged"
                        ), f"Unknown variably invoked predictor grouping mode {self.variably_invoked_predictor_grouping_mode}"
                    max_len = max([len(predictor_example_invocations[i]) for i in range(len(predictor_example_invocations))])

                    example_training_data: list[GRPOGroup] = [[] for _ in range(max_len)]

                    for group_idx in range(max_len):
                        for rollout_idx in range(len(predictor_example_invocations)):
                            trace_instance = predictor_example_invocations[rollout_idx][group_idx]
                            score = trace_instance[3]
                            trace_pred_id = pred_signature_hash_to_ind.get(hash(trace_instance[0].signature))
                            assert trace_pred_id == pred_id

                            predictor = trace_instance[0]
                            pred_lm = predictor.lm
                            adapter = self.adapter[pred_lm] or settings.adapter or XMLAdapter()
                            assert isinstance(adapter, ChatAdapter)
                            inp_messages = adapter.format(
                                signature=trace_instance[0].signature,
                                inputs=trace_instance[1],
                                demos=[],
                            )

                            if isinstance(trace_instance[2], FailedPrediction):
                                score = trace_instance[2].format_reward or self.format_failure_score
                                example_training_data[group_idx].append(
                                    {
                                        "messages": inp_messages,
                                        "completion": {
                                            "role": "assistant",
                                            "content": trace_instance[2].completion_text,
                                        },
                                        "reward": float(score),
                                    }
                                )
                                logger.warning(
                                    f"Adding a format failure example to the training data for predictor {pred_id} and example {example_ind}."
                                )
                            else:
                                all_messages = adapter.format_finetune_data(
                                    signature=trace_instance[0].signature,
                                    inputs=trace_instance[1],
                                    outputs=trace_instance[2],
                                    demos=[],
                                )["messages"]

                                assert all_messages[:-1] == inp_messages

                                example_training_data[group_idx].append(
                                    {
                                        "messages": inp_messages,
                                        "completion": {
                                            "role": all_messages[-1]["role"],
                                            "content": all_messages[-1]["content"],
                                        },
                                        "reward": float(score),
                                    }
                                )

                    train_batch_per_predictor[pred_id].extend(example_training_data)

            if not any(train_batch_per_predictor):
                logger.warning(
                    "No training data found for this training step. This means that the model did not generate valid formatted responses for any of the examples in the training set. This is a critical error. Please check the model and the training set."
                )
                continue

            for predictor_train_batch in train_batch_per_predictor:
                for grpo_train_group in predictor_train_batch:
                    if len(grpo_train_group) != self.num_rollouts_per_grpo_step:
                        logger.warning(
                            f"Number of completions {len(grpo_train_group)} does not match the expected number num_rollouts_per_grpo_step={self.num_rollouts_per_grpo_step}"
                        )
                        assert (
                            len(grpo_train_group) <= self.num_rollouts_per_grpo_step
                        )
                    if len(set(map(repr, grpo_train_group))) < 2:
                        logger.warning(
                            f"GRPOGroup has no diversity. This could be due to low temperature, or low number of rollouts, or the cache could be enabled inadvertently. The GRPOGroup is {grpo_train_group}."
                        )

            logger.info("Invoking GRPO training step...")
            for (lm_for_job, data_key), job in grpo_training_jobs.items():
                train_data: list[GRPOGroup] = (
                    sum(train_batch_per_predictor, []) if data_key is None else train_batch_per_predictor[data_key]
                )
                for group in train_data:
                    if len(group) != self.num_rollouts_per_grpo_step:
                        while len(group) < self.num_rollouts_per_grpo_step:
                            group.extend(group[: min(self.num_rollouts_per_grpo_step - len(group), len(group))])
                    assert (
                        len(group) == self.num_rollouts_per_grpo_step
                    )

                grpo_status: GRPOStatus = job.get_status()
                pending_batch_ids = grpo_status["pending_batch_ids"]
                available_batch_ids = list(set(pending_batch_ids) - set(self.fulfilled_batch_ids))
                if not available_batch_ids:
                    continue

                job_key = (lm_for_job, data_key)
                q = group_queues.setdefault(job_key, deque())

                if len(q) < len(available_batch_ids) and len(train_data) > 0:
                    need = len(available_batch_ids) - len(q)
                    while need > 0:
                        shuffled = self.rng.sample(train_data, k=len(train_data))
                        q.extend(shuffled)
                        need -= len(shuffled)

                final_train_data: list[GRPOGroup] = []
                for bid in available_batch_ids:
                    if q:
                        grp = q.popleft()
                    else:
                        fallback_pool = train_data if len(train_data) > 0 else sum(train_batch_per_predictor, [])
                        if len(fallback_pool) == 0:
                            continue
                        grp = self.rng.choice(fallback_pool)
                    final_train_data.append({"batch_id": bid, "group": grp})

                if not final_train_data:
                    continue

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

        recover_lm_cache(program=student, lm_cache_dict=lm_cache_dict)
        for t in teachers:
            recover_lm_cache(program=t, lm_cache_dict=lm_cache_dict)

        logger.info("GRPO compiler has finished compiling the student program")
        student._compiled = True
        return student


def disable_lm_cache(program: Module, lm_cache_dict: dict):
    for pred in program.predictors():
        if not pred.lm:
            raise ValueError(f"Cannot disable cache: predictor {pred} does not have an LM set.")
        if pred.lm not in lm_cache_dict:
            lm_cache_dict[pred.lm] = pred.lm.cache
        pred.lm.cache = False


def recover_lm_cache(program: Module, lm_cache_dict: dict):
    for pred in program.predictors():
        if pred.lm in lm_cache_dict:
            pred.lm.cache = lm_cache_dict[pred.lm]
        else:
            pred.lm.cache = True


