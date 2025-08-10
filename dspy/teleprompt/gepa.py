import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

from dspy.dsp.utils.settings import settings
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types import History
from dspy.clients.lm import LM
from dspy.evaluate import Evaluate
from dspy.primitives import Example, Module

from gepa.gepa import EvaluationBatch, GEPAAdapter, GEPAResult, optimize
from .teleprompt import Teleprompter

logger = logging.getLogger(__name__)

class LoggerAdapter:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log(self, x: str):
        self.logger.info(x)

@dataclass(frozen=True)
class DspyGEPAResult:
    """
    Additional data related to the GEPA run.

    Fields:
    - candidates: list of proposed candidates (component_name -> component_text)
    - parents: lineage info; for each candidate i, parents[i] is a list of parent indices or None
    - val_aggregate_scores: per-candidate aggregate score on the validation set (higher is better)
    - val_subscores: per-candidate per-instance scores on the validation set (len == num_val_instances)
    - per_val_instance_best_candidates: for each val instance t, a set of candidate indices achieving the current best score on t
    - discovery_eval_counts: number of metric calls accumulated up to the discovery of each candidate

    - total_metric_calls: total number of metric calls made across the run
    - num_full_val_evals: number of full validation evaluations performed
    - run_dir: where artifacts were written (if any)
    - seed: RNG seed for reproducibility (if known)

    - best_idx: candidate index with the highest val_aggregate_scores
    - best_candidate: the program text mapping for best_idx
    """
    # Core data
    candidates: List[Module]
    parents: List[List[Optional[int]]]
    val_aggregate_scores: List[float]
    val_subscores: List[List[float]]
    per_val_instance_best_candidates: List[Set[int]]
    discovery_eval_counts: List[int]

    # Run metadata (optional)
    total_metric_calls: Optional[int] = None
    num_full_val_evals: Optional[int] = None
    run_dir: Optional[str] = None
    seed: Optional[int] = None

    @property
    def best_idx(self) -> int:
        scores = self.val_aggregate_scores
        return max(range(len(scores)), key=lambda i: scores[i])

    @property
    def best_candidate(self) -> Dict[str, str]:
        return self.candidates[self.best_idx]

    def to_dict(self) -> Dict[str, Any]:
        cands = [
            {k: v for k, v in cand.items()}
            for cand in self.candidates
        ]

        return dict(
            candidates=cands,
            parents=self.parents,
            val_aggregate_scores=self.val_aggregate_scores,
            val_subscores=self.val_subscores,
            per_val_instance_best_candidates=[list(s) for s in self.per_val_instance_best_candidates],
            discovery_eval_counts=self.discovery_eval_counts,
            total_metric_calls=self.total_metric_calls,
            num_full_val_evals=self.num_full_val_evals,
            run_dir=self.run_dir,
            seed=self.seed,
            best_idx=self.best_idx,
        )

    def from_gepa_result(gepa_result: GEPAResult, adapter: "DspyAdapter") -> "DspyGEPAResult":
        return DspyGEPAResult(
            candidates=[adapter.build_program(c) for c in gepa_result.candidates],
            parents=gepa_result.parents,
            val_aggregate_scores=gepa_result.val_aggregate_scores,
            val_subscores=gepa_result.val_subscores,
            per_val_instance_best_candidates=gepa_result.per_val_instance_best_candidates,
            discovery_eval_counts=gepa_result.discovery_eval_counts,
            total_metric_calls=gepa_result.total_metric_calls,
            num_full_val_evals=gepa_result.num_full_val_evals,
            run_dir=gepa_result.run_dir,
            seed=gepa_result.seed,
        )

class DspyAdapter(GEPAAdapter):
    def __init__(
        self,
        student_module,
        metric_fn: Callable,
        feedback_map: Dict[str, Callable],
        failure_score=0.0,
        num_threads: Optional[int] = None,
        add_format_failure_as_feedback: bool = False,
        rng: Optional[random.Random] = None,
    ):
        import dspy
        self.student = student_module
        self.metric_fn = metric_fn
        self.feedback_map = feedback_map
        self.failure_score = failure_score
        self.num_threads = num_threads or os.cpu_count()
        self.add_format_failure_as_feedback = add_format_failure_as_feedback
        self.rng = rng or random.Random(0)

        # Cache predictor names/signatures
        self.named_predictors = list(self.student.named_predictors())

    def build_program(self, candidate: Dict[str, str]):
        new_prog = self.student.deepcopy()
        for name, pred in new_prog.named_predictors():
            if name in candidate:
                pred.signature = pred.signature.with_instructions(candidate[name])
        return new_prog

    def evaluate(self, batch, candidate, capture_traces=False):
        import dspy
        program = self.build_program(candidate)

        if capture_traces:
            # bootstrap_trace_data-like flow with trace capture
            from .bootstrap_finetune import bootstrap_trace_data
            trajs = bootstrap_trace_data(
                program=program,
                dataset=batch,
                metric=self.metric_fn,
                num_threads=self.num_threads,
                raise_on_error=False,
                capture_failed_parses=True,
                failure_score=self.failure_score,
                format_failure_score=self.failure_score,
            )
            scores = []
            outputs = []
            for t in trajs:
                outputs.append(t['prediction'])
                if hasattr(t['prediction'], '__class__') and t.get('score') is None:
                    scores.append(self.failure_score)
                else:
                    scores.append(t['score'])
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajs)
        else:
            evaluator = Evaluate(
                devset=batch,
                metric=self.metric_fn,
                num_threads=self.num_threads,
                return_all_scores=True,
                return_outputs=True,
                failure_score=self.failure_score,
                provide_traceback=True,
                max_errors=len(batch) * 100
            )
            res = evaluator(program)
            outputs = [r[1] for r in res.results]
            scores = [r[2] for r in res.results]
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        import dspy
        from .bootstrap_finetune import FailedPrediction
        program = self.build_program(candidate)

        ret_d: Dict[str, List[Dict[str, Any]]] = {}
        for pred_name in components_to_update:
            feedback_fn = self.feedback_map[pred_name]
            module = None
            for name, m in program.named_predictors():
                if name == pred_name:
                    module = m
                    break
            assert module is not None

            items: List[Dict[str, Any]] = []
            for data in eval_batch.trajectories or []:
                trace = data["trace"]
                example = data["example"]
                prediction = data["prediction"]

                trace_instances = [t for t in trace if t[0].signature.equals(module.signature)]
                if not self.add_format_failure_as_feedback:
                    trace_instances = [t for t in trace_instances if not isinstance(t[2], FailedPrediction)]
                if len(trace_instances) == 0:
                    continue

                selected = None
                for t in trace_instances:
                    if isinstance(t[2], FailedPrediction):
                        selected = t
                        break

                if selected is None:
                    if isinstance(prediction, FailedPrediction):
                        continue
                    selected = self.rng.choice(trace_instances)

                inputs = selected[1]
                outputs = selected[2]

                new_inputs = {}
                new_outputs = {}

                contains_history = False
                history_key_name = None
                for input_key, input_val in inputs.items():
                    if isinstance(input_val, History):
                        contains_history = True
                        assert history_key_name is None
                        history_key_name = input_key
                
                if contains_history:
                    s = "```json\n"
                    for i, message in enumerate(inputs[history_key_name].messages):
                        s += f"  {i}: {message}\n"
                    s += "```"
                    new_inputs["Context"] = s
                
                for input_key, input_val in inputs.items():
                    if contains_history and input_key == history_key_name:
                        continue
                    new_inputs[input_key] = str(input_val)

                if isinstance(outputs, FailedPrediction):
                    s = "Couldn't parse the output as per the expected output format. The model's raw response was:\n"
                    s += "```\n"
                    s += outputs.completion_text + "\n"
                    s += "```\n\n"
                    new_outputs = s
                else:
                    for output_key, output_val in outputs.items():
                        new_outputs[output_key] = str(output_val)

                d = {"Inputs": new_inputs, "Generated Outputs": new_outputs}
                if isinstance(outputs, FailedPrediction):
                    adapter = ChatAdapter()
                    structure_instruction = ""
                    for dd in adapter.format(module.signature, [], {}):
                        structure_instruction += dd["role"] + ": " + dd["content"] + "\n"
                    d['Feedback'] = "Your output failed to parse. Follow this structure:\n" + structure_instruction
                    # d['score'] = self.failure_score
                else:
                    fb = feedback_fn(
                        predictor_output=outputs,
                        predictor_inputs=inputs,
                        module_inputs=example,
                        module_outputs=prediction,
                        captured_trace=trace,
                    )
                    d['Feedback'] = fb["feedback_text"]
                    # d['score'] = fb["feedback_score"]
                items.append(d)

            if len(items) == 0:
                # raise Exception(f"No valid predictions found for module {module.signature}.")
                continue
            ret_d[pred_name] = items
        
        if len(ret_d) == 0:
            raise Exception(f"No valid predictions found for any module.")

        return ret_d

class GEPA(Teleprompter):
    def __init__(
        self,
        named_predictor_to_feedback_fn_map: Dict[str, Callable],
        metric: Callable,
        run_dir: str,
        run_linearized_gepa: bool = True,  # kept for API compatibility
        num_threads: Optional[int] = None,
        num_iters: Optional[int] = None,
        failure_score: float = 0.0,
        perfect_score: float = 1.0,
        teacher_lm: Optional[LM] = None,
        use_wandb: bool = False,
        wandb_api_key: Optional[str] = None,
        max_evals_per_trainval_instance: Optional[int] = None,
        seed: int = 0,
        skip_perfect_score: bool = True,
        use_merge: bool = False,
        max_merge_invocations: int = 5,
        num_dspy_examples_per_gepa_step: int = 3,
        max_metric_calls: Optional[int] = None,
        add_format_failure_as_feedback: bool = False,
        track_stats: bool = False,
    ):
        # Exactly one of the three budget controls must be provided
        assert (
            (max_metric_calls is not None)
            + (max_evals_per_trainval_instance is not None)
            + (num_iters is not None)
            == 1
        ), (
            "Exactly one of max_metric_calls, max_evals_per_trainval_instance or num_iters must be set. "
            f"You set max_metric_calls={max_metric_calls}, "
            f"max_evals_per_trainval_instance={max_evals_per_trainval_instance}, "
            f"num_iters={num_iters}"
        )

        self.named_predictor_to_feedback_fn_map = named_predictor_to_feedback_fn_map
        self.metric_fn = metric
        self.run_dir = run_dir
        self.run_linearized_gepa = run_linearized_gepa

        self.num_threads = num_threads or os.cpu_count()
        self.num_iters = num_iters
        self.max_evals_per_trainval_instance = max_evals_per_trainval_instance
        self.max_metric_calls = max_metric_calls

        self.failure_score = failure_score
        self.perfect_score = perfect_score
        self.teacher_lm = teacher_lm
        self.use_wandb = use_wandb
        self.wandb_api_key = wandb_api_key

        self.seed = seed
        self.skip_perfect_score = skip_perfect_score
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations

        self.num_dspy_examples_per_gepa_step = num_dspy_examples_per_gepa_step
        self.add_format_failure_as_feedback = add_format_failure_as_feedback
        self.track_stats = track_stats

        self._rng = random.Random(seed)

    def _resolve_budget(self, train_n: int, val_n: int) -> Dict[str, Optional[int]]:
        """
        Normalize the 3 user-facing budget options to the engine's (num_iters | max_metric_calls).
        If max_evals_per_trainval_instance is set, approximate the global budget as
        (train_n + val_n) * max_evals_per_trainval_instance.
        """
        if self.max_metric_calls is not None:
            return dict(num_iters=None, max_metric_calls=self.max_metric_calls)

        if self.max_evals_per_trainval_instance is not None:
            # Simple, conservative mapping to the engine's total eval counter
            # Includes both minibatch evals and full-valset evals
            total_instances = train_n + val_n
            return dict(
                num_iters=None,
                max_metric_calls=self.max_evals_per_trainval_instance * max(1, total_instances),
            )

        # Fallback to num_iters if provided
        return dict(num_iters=self.num_iters, max_metric_calls=None)

    def compile(
        self,
        student: Module,
        *,
        trainset: List[Example],
        teacher: Optional[Module] = None,
        valset: Optional[List[Example]] = None,
        **kwargs,
    ) -> Module:
        assert trainset is not None and len(trainset) > 0, "Trainset must be provided and non-empty"
        assert teacher is None, "Teacher is not supported in DspyGEPA yet."

        valset = valset or trainset

        # Build the DSPy adapter that encapsulates evaluation, trace capture, feedback extraction, and instruction proposal
        adapter = DspyAdapter(
            student_module=student,
            metric_fn=self.metric_fn,
            feedback_map=self.named_predictor_to_feedback_fn_map,
            failure_score=self.failure_score,
            num_threads=self.num_threads,
            add_format_failure_as_feedback=self.add_format_failure_as_feedback,
            rng=self._rng,
        )

        # Prepare engine budgets
        budgets = self._resolve_budget(train_n=len(trainset), val_n=len(valset))

        teacher_lm = lambda x: (self.teacher_lm or settings.lm or student.get_lm())(x)[0]

        # Instantiate GEPA with the simpler adapter-based API
        base_program = {name: pred.signature.instructions for name, pred in student.named_predictors()}
        gepa_result: GEPAResult = optimize(
            base_program=base_program,
            trainset=trainset,
            adapter=adapter,
            valset=valset,
            logger=LoggerAdapter(logger),
            run_dir=self.run_dir,
            teacher_lm=teacher_lm,
            candidate_selection_strategy="pareto",
            num_iters=budgets["num_iters"],
            perfect_score=self.perfect_score,
            use_wandb=self.use_wandb,
            wandb_api_key=self.wandb_api_key,
            seed=self.seed,
            skip_perfect_score=self.skip_perfect_score,
            use_merge=self.use_merge,
            max_merge_invocations=self.max_merge_invocations,
            num_examples_per_gepa_step=self.num_dspy_examples_per_gepa_step,
            max_metric_calls=budgets["max_metric_calls"],
        )

        new_prog = adapter.build_program(gepa_result.best_candidate)

        if self.track_stats:
            dspy_gepa_result = DspyGEPAResult.from_gepa_result(gepa_result, adapter)
            setattr(new_prog, "dspy_gepa_result", dspy_gepa_result)

        return new_prog
