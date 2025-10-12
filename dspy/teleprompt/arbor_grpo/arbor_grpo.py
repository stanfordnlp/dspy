from __future__ import annotations

from collections import defaultdict
from typing import Any

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.lm import LM
from dspy.clients.utils_finetune import GRPOGroup, MultiGPUConfig, TrainDataFormat
from dspy.dsp.utils.settings import settings
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt.bootstrap_finetune import (
    all_predictors_have_lms,
    assert_structural_equivalency,
)
from dspy.teleprompt.bootstrap_trace import FailedPrediction, bootstrap_trace_data
from dspy.teleprompt.grpo import GRPO, disable_lm_cache, recover_lm_cache

from .arbor_grpo_config import ArborGRPOConfig


class ArborGRPO(GRPO):
    """GRPO variant that accepts ArborGRPOConfig directly per-LM and uses it for reinforcement.

    Backwards-compatible with GRPO's public methods and compile loop.
    """

    def __init__(
        self,
        *args,
        config: ArborGRPOConfig | dict[LM, ArborGRPOConfig] | None = None,
        gpu_config: MultiGPUConfig = MultiGPUConfig(num_inference_gpus=1, num_training_gpus=1),
        **kwargs,
    ):

        super().__init__(*args, gpu_config=gpu_config, **kwargs)
        self.gpu_config = gpu_config
        self.grpo_configs: dict[LM, ArborGRPOConfig] = self._convert_to_grpo_config_dict(config)

    def _create_default_config_dict(self) -> dict[LM, ArborGRPOConfig]:
        """Create a default config dict with ArborGRPOConfig(num_generations=1)."""
        return defaultdict(lambda: ArborGRPOConfig(num_generations=1))

    def _create_single_config_dict(self, config: ArborGRPOConfig) -> dict[LM, ArborGRPOConfig]:
        """Create a config dict from a single ArborGRPOConfig."""
        return defaultdict(lambda: config)

    def _validate_defaultdict_config(self, config: dict) -> dict[LM, ArborGRPOConfig]:
        """Validate and return a defaultdict config."""
        default_value = config.default_factory()  # type: ignore[attr-defined]
        if isinstance(default_value, ArborGRPOConfig):
            return config
        raise ValueError(
            f"defaultdict default factory must return ArborGRPOConfig, got {type(default_value)}"
        )

    def _validate_lm_dict_config(self, config: dict[LM, ArborGRPOConfig]) -> dict[LM, ArborGRPOConfig]:
        """Validate and convert an LM dict config."""
        if not config:
            raise ValueError("Dictionary config must have LM keys, got keys of type empty dict")

        if not all(isinstance(k, LM) for k in config.keys()):
            bad_type = type(next(iter(config.keys())))
            raise ValueError(f"Dictionary config must have LM keys, got keys of type {bad_type}")

        result: dict[LM, ArborGRPOConfig] = {}
        for lm, cfg in config.items():
            if not isinstance(cfg, ArborGRPOConfig):
                raise ValueError(
                    f"All values in LM dict must be ArborGRPOConfig instances, got {type(cfg)}"
                )
            result[lm] = cfg
        return result

    def _convert_to_grpo_config_dict(
        self, config: ArborGRPOConfig | dict[LM, ArborGRPOConfig] | None
    ) -> dict[LM, ArborGRPOConfig]:
        """Convert user-supplied config into a normalized dict[LM, ArborGRPOConfig]."""
        if config is None:
            return self._create_default_config_dict()

        if isinstance(config, ArborGRPOConfig):
            return self._create_single_config_dict(config)

        if isinstance(config, dict):
            if hasattr(config, "default_factory"):
                return self._validate_defaultdict_config(config)
            return self._validate_lm_dict_config(config)

        raise ValueError(
            f"config must be ArborGRPOConfig, dict[LM, ArborGRPOConfig], or None, got {type(config)}"
        )

    # Override the places GRPO uses train_kwargs to instead use GRPOConfig
    def _set_num_generations_on_training_params(self, student) -> None:
        # Called early in compile to update generation count per LM
        for pred in student.predictors():
            cfg = self.grpo_configs[pred.lm]
            cfg.num_generations = self.num_rollouts_per_grpo_step

    def _create_or_get_job(self, grpo_training_jobs: dict, job_key: tuple[Any, Any], pred) -> None:
        if job_key not in grpo_training_jobs:
            cfg = self.grpo_configs[pred.lm]
            job = pred.lm.reinforce(config=cfg, gpu_config=self.gpu_config)
            grpo_training_jobs[job_key] = job

    # Monkey-patch-compatible hooks into GRPO.compile by overriding compile and reusing base logic
    def compile(self, *args, **kwargs):  # type: ignore[override]
        # Run the original compile up to the injection points using base logic.
        # We reuse GRPO.compile entirely and rely on the two helper methods above by
        # shadowing names the base compile uses just before its usages.
        # To achieve this without modifying the base file further, we duplicate the few
        # lines where train_kwargs/config is manipulated by calling our hooks immediately
        # after the base has prepared teachers, etc. Minimal override strategy:

        # Call into GRPO.compile but temporarily monkey-in our behavior by
        # delegating to the same algorithm while replacing the two spots.
        return self._compile_with_overrides(*args, **kwargs)

    def _setup_and_validate_inputs(self, student, trainset, teacher, valset):
        """Set up and validate the compilation inputs."""

        assert len(trainset) > 0, "Training set is empty. Please provide a non-empty training set."

        if len(trainset) < self.num_dspy_examples_per_grpo_step:
            multiplier = (self.num_dspy_examples_per_grpo_step + len(trainset) - 1) // len(trainset)
            if multiplier > 1:
                trainset = trainset * multiplier

        if not self.multitask:
            raise ValueError(
                "Independent GRPO training jobs for each predictor in the student program are not supported yet. Please set multitask=True."
            )

        student_lms = {id(pred.lm) for pred in student.predictors()}
        assert len(student_lms) == 1, (
            f"Student program has multiple LMs: {student_lms}. GRPO only supports a single LM."
        )

        if self.use_train_as_val:
            assert valset is None, "If use_train_as_val is True, valset must be None."

        all_predictors_have_lms(student)
        pred_signature_hash_to_ind = {hash(pred.signature): ind for ind, pred in enumerate(student.predictors())}
        num_student_predictors = len(student.predictors())

        if (isinstance(teacher, list) and len(teacher) == 0) or teacher is None:
            teacher = student
        teachers = teacher if isinstance(teacher, list) else [teacher]
        for t in teachers:
            assert_structural_equivalency(student, t)
            all_predictors_have_lms(t)
        assert student in teachers
        assert self.num_rollouts_per_grpo_step % len(teachers) == 0
        num_samples_per_input = self.num_rollouts_per_grpo_step // len(teachers)

        return teachers, num_samples_per_input, pred_signature_hash_to_ind, num_student_predictors

    def _setup_teachers_and_cache(self, student, teachers):
        """Set up LM cache for student and teachers."""

        lm_cache_dict: dict = {}
        disable_lm_cache(program=student, lm_cache_dict=lm_cache_dict)
        for t in teachers:
            disable_lm_cache(program=t, lm_cache_dict=lm_cache_dict)

        return lm_cache_dict

    def _setup_training_jobs(self, student, grpo_training_jobs):
        """Set up GRPO training jobs with our config override."""
        self._set_num_generations_on_training_params(student)

        for pred_ind, pred in enumerate(student.predictors()):
            data_key = None if self.multitask else pred_ind
            job_key = (pred.lm, data_key)
            self._create_or_get_job(grpo_training_jobs, job_key, pred)

    def _create_validation_reporter(self, valset, trainset, student):
        """Create a validation reporting function."""

        def report_validation_metrics(step_idx=-1):
            if valset is None:
                return
            evaluator = Evaluate(
                devset=(valset + trainset) if self.report_train_scores else valset,
                num_threads=self.num_threads,
                display_progress=True,
                provide_traceback=False,
                max_errors=len(valset) * 10,
                failure_score=self.failure_score,
            )
            evaluator(student, metric=self.metric)

        return report_validation_metrics

    def _collect_trace_data(self, subsample_training_dataset, teachers, num_samples_per_input):
        """Collect trace data from teachers."""

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
            )
            for data_dict in round_data:
                example_ind_in_subsample = data_dict["example_ind"] % len(subsample_training_dataset)
                data_dict["example_ind"] = example_ind_in_subsample
                trace_data[example_ind_in_subsample][tind].append(data_dict)

        # Minimal validation
        if trace_data and trace_data[0] and trace_data[0][0] and "trace" in trace_data[0][0][0]:
            assert len(trace_data[0][0][0]["trace"]) > 0

        return trace_data

    def _prepare_training_batches(self, trace_data, student, num_student_predictors):
        """Prepare training batches from trace data."""

        train_batch_per_predictor: list[list[GRPOGroup]] = [[] for _ in range(num_student_predictors)]
        for pred_id in range(num_student_predictors):
            for example_ind, example_data in enumerate(trace_data):
                predictor_example_invocations: list[list[tuple]] = []
                for teacher_data in example_data:
                    for sample in teacher_data:
                        assert sample["example_ind"] == example_ind
                        trace_instances_for_current_pred = [
                            (*t, sample["score"]) for t in sample["trace"]
                            if hash(t[0].signature) == hash(student.predictors()[pred_id].signature)
                        ]
                        predictor_example_invocations.append(trace_instances_for_current_pred)

                if not predictor_example_invocations:
                    continue

                min_len = min(len(x) for x in predictor_example_invocations)
                example_training_data: list[GRPOGroup] = [[] for _ in range(min_len)]
                for group_idx in range(min_len):
                    for rollout_idx in range(len(predictor_example_invocations)):
                        trace_instance = predictor_example_invocations[rollout_idx][group_idx]
                        score = trace_instance[3]
                        adapter = self.adapter[student.predictors()[pred_id].lm] or settings.adapter or ChatAdapter()
                        inp_messages = adapter.format(
                            signature=trace_instance[0].signature, inputs=trace_instance[1], demos=[]
                        )
                        if isinstance(trace_instance[2], FailedPrediction):
                            example_training_data[group_idx].append(
                                {
                                    "messages": inp_messages,
                                    "completion": {"role": "assistant", "content": trace_instance[2].completion_text},
                                    "reward": float(score),
                                }
                            )
                        else:
                            all_messages = adapter.format_finetune_data(
                                signature=trace_instance[0].signature,
                                inputs=trace_instance[1],
                                outputs=trace_instance[2],
                                demos=[],
                            )["messages"]
                            example_training_data[group_idx].append(
                                {
                                    "messages": inp_messages,
                                    "completion": {"role": all_messages[-1]["role"], "content": all_messages[-1]["content"]},
                                    "reward": float(score),
                                }
                            )
                train_batch_per_predictor[pred_id].extend(example_training_data)

        return train_batch_per_predictor

    def _run_training_step(self, grpo_training_jobs, train_batch_per_predictor):
        """Execute a single training step."""

        for (_, data_key), job in grpo_training_jobs.items():
            train_data = sum(train_batch_per_predictor, []) if data_key is None else train_batch_per_predictor[data_key]
            job.step(train_data=train_data, train_data_format=TrainDataFormat.GRPO_CHAT)

    def _cleanup_training(self, grpo_training_jobs, student, teachers, lm_cache_dict):
        """Clean up training jobs and restore LM cache."""

        for _, job in grpo_training_jobs.items():
            job.terminate()

        recover_lm_cache(program=student, lm_cache_dict=lm_cache_dict)
        for t in teachers:
            recover_lm_cache(program=t, lm_cache_dict=lm_cache_dict)

    def _compile_with_overrides(self, *args, **kwargs):
        # Unpack args according to GRPO.compile signature
        student = kwargs.pop("student") if "student" in kwargs else args[0]
        trainset = kwargs.pop("trainset") if "trainset" in kwargs else args[1]
        teacher = kwargs.pop("teacher") if "teacher" in kwargs else (args[2] if len(args) > 2 else None)
        valset = kwargs.pop("valset") if "valset" in kwargs else (args[3] if len(args) > 3 else None)

        teachers, num_samples_per_input, pred_signature_hash_to_ind, num_student_predictors = self._setup_and_validate_inputs(
            student, trainset, teacher, valset
        )

        lm_cache_dict = self._setup_teachers_and_cache(student, teachers)

        grpo_training_jobs: dict = {}
        self._setup_training_jobs(student, grpo_training_jobs)

        report_validation_metrics = self._create_validation_reporter(valset, trainset, student)
        report_validation_metrics(step_idx=-1)

        for train_step_idx in range(self.num_train_steps):
            subsample_training_dataset = self.select_training_sample_and_update_shuffled_trainset(
                original_trainset=trainset, train_step_idx=train_step_idx
            )

            trace_data = self._collect_trace_data(subsample_training_dataset, teachers, num_samples_per_input)

            train_batch_per_predictor = self._prepare_training_batches(trace_data, student, num_student_predictors)

            self._run_training_step(grpo_training_jobs, train_batch_per_predictor)

            # Periodic validation
            if (train_step_idx + 1) % max(1, self.num_steps_for_val) == 0:
                report_validation_metrics(step_idx=train_step_idx)

        self._cleanup_training(grpo_training_jobs, student, teachers, lm_cache_dict)

        student._compiled = True
        return student


