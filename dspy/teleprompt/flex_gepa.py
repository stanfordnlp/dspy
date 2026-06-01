from __future__ import annotations

import logging
import math
import random
from typing import Any, Callable, Literal, cast

from gepa import EvaluationBatch, GEPAAdapter, GEPAResult, optimize

import dspy
from dspy.clients.lm import LM
from dspy.evaluate.evaluate import Evaluate
from dspy.flex.codegen import _strip_code_fences
from dspy.flex.exploration import ExplorationStore, candidate_id
from dspy.flex.flex import Flex
from dspy.flex.manifest import ManifestStore
from dspy.flex.primitives_doc import PRIMITIVES_CATALOG
from dspy.primitives import Example, Prediction
from dspy.teleprompt.bootstrap_trace import FailedPrediction, TraceData, bootstrap_trace_data
from dspy.teleprompt.gepa.gepa_utils import LoggerAdapter
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.annotation import experimental

logger = logging.getLogger(__name__)


AUTO_RUN_SETTINGS = {
    "light": {"n": 6},
    "medium": {"n": 12},
    "heavy": {"n": 18},
}


_COMPONENTS = ("predictors_src", "forward_src")


class CodeProposalSignature(dspy.Signature):
    """Revise the source of one code component of a dspy.Flex module.

    You will receive the user-facing task description, the catalog of allowed
    primitives, the current source of one component (either the ``PREDICTORS``
    dict or the ``forward`` function), and a batch of failing examples with
    feedback. Produce a revised source that fixes the observed failures.

    The revised source MUST be drop-in compatible with the rest of the module
    (the other component is unchanged unless you are revising it).
    """

    task_description: str = dspy.InputField(
        desc="Rendered description of the Flex module's Signature: name, objective, input and output fields."
    )
    primitives_catalog: str = dspy.InputField(
        desc="Catalog of allowed primitives. The revised source should follow these conventions."
    )
    component_name: str = dspy.InputField(
        desc="The component being revised: either 'predictors_src' (the PREDICTORS dict) or 'forward_src' (the forward function)."
    )
    current_source: str = dspy.InputField(desc="Current source of the component being revised.")
    sibling_source: str = dspy.InputField(
        desc="Current source of the OTHER component (read-only context — do not edit, just keep compatible)."
    )
    failures: str = dspy.InputField(
        desc="A batch of failing examples and feedback. Diagnose the failures and revise the component to fix them."
    )

    revised_source: str = dspy.OutputField(
        desc="The full revised source of the component."
    )


def _short_repr(value: Any, max_len: int = 800) -> str:
    s = str(value)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _render_prediction(prediction: Any) -> Any:
    if isinstance(prediction, FailedPrediction):
        return f"FailedPrediction: {prediction.completion_text[:1000]}"
    if isinstance(prediction, Prediction):
        try:
            return {k: _short_repr(v) for k, v in prediction.items()}
        except Exception:
            pass
    return _short_repr(prediction)


class FlexAdapter(GEPAAdapter[Example, TraceData, Prediction]):
    """GEPA adapter that evolves the source code of a `Flex` module."""

    def __init__(
        self,
        flex_module: Flex,
        metric_fn: Callable,
        *,
        reflection_lm: LM | None = None,
        proposer_type: type = dspy.Predict,
        proposer_kwargs: dict[str, Any] | None = None,
        failure_score: float = 0.0,
        num_threads: int | None = None,
        rng: random.Random | None = None,
        exploration_store: ExplorationStore | None = None,
    ):
        self.student = flex_module
        self.metric_fn = metric_fn
        self.reflection_lm = reflection_lm
        self.proposer_type = proposer_type
        self.proposer_kwargs = proposer_kwargs or {}
        self.failure_score = failure_score
        self.num_threads = num_threads
        self.rng = rng or random.Random(0)

        self._task_description = flex_module._flex_ctx.render_signature_spec()
        self._signature_hash = flex_module._signature_hash()
        self.exploration = exploration_store or flex_module._exploration

    def build_program(self, candidate: dict[str, str]) -> Flex:
        program = self.student.deepcopy()
        program._bind_code(candidate["predictors_src"], candidate["forward_src"])
        return program

    def evaluate(self, batch, candidate, capture_traces=False) -> EvaluationBatch:
        program = self.build_program(candidate)

        if capture_traces:
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
            outputs: list[Any] = []
            scores: list[float] = []
            for t in trajs:
                outputs.append(t["prediction"])
                s: Any = t.get("score")
                if hasattr(s, "score"):
                    s = cast(dict, s)["score"]
                scores.append(self.failure_score if s is None else s)
            self._record_evaluation(candidate, scores)
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajs)

        evaluator = Evaluate(
            devset=batch,
            metric=self.metric_fn,
            num_threads=self.num_threads,
            return_all_scores=True,
            failure_score=self.failure_score,
            provide_traceback=True,
            max_errors=len(batch) * 100,
        )
        res = evaluator(program)
        outputs = [r[1] for r in res.results]
        raw_scores: list[Any] = [r[2] for r in res.results]
        final_scores: list[float] = []
        for s in raw_scores:
            if hasattr(s, "score"):
                final_scores.append(cast(dict, s)["score"])
            else:
                final_scores.append(s)
        self._record_evaluation(candidate, final_scores)
        return EvaluationBatch(outputs=outputs, scores=final_scores, trajectories=None)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        records: list[dict[str, Any]] = []
        for data in eval_batch.trajectories or []:
            example = data["example"]
            prediction = data["prediction"]
            raw_score: Any = data.get("score")

            if hasattr(raw_score, "score"):
                score_dict = cast(dict, raw_score)
                feedback_text = score_dict.get("feedback") or ""
                score_val = score_dict["score"]
            else:
                feedback_text = ""
                score_val = raw_score

            inputs_repr = {k: _short_repr(v) for k, v in example.inputs().items()}
            outputs_repr = _render_prediction(prediction)
            fb = feedback_text or f"This trajectory got a score of {score_val}."
            records.append(
                {
                    "Inputs": inputs_repr,
                    "Generated Outputs": outputs_repr,
                    "Feedback": fb,
                }
            )

        if not records:
            raise Exception("No reflective records could be built from the evaluation batch.")

        return {component: records for component in components_to_update if component in candidate}

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        reflection_lm = self.reflection_lm or dspy.settings.lm
        proposer = self.proposer_type(CodeProposalSignature, **self.proposer_kwargs)
        parent_id = candidate_id(candidate["predictors_src"], candidate["forward_src"])

        results: dict[str, str] = {}
        with dspy.context(lm=reflection_lm):
            for component in components_to_update:
                if component not in candidate:
                    continue
                sibling = "forward_src" if component == "predictors_src" else "predictors_src"
                failures_blob = _format_failures(reflective_dataset.get(component, []))
                try:
                    out = proposer(
                        task_description=self._task_description,
                        primitives_catalog=PRIMITIVES_CATALOG,
                        component_name=component,
                        current_source=candidate[component],
                        sibling_source=candidate.get(sibling, ""),
                        failures=failures_blob,
                    )
                    revised = _strip_code_fences(out.revised_source)
                except Exception as e:
                    logger.warning("FlexAdapter: proposer raised %s on %s; keeping original.", e, component)
                    results[component] = candidate[component]
                    continue

                results[component] = revised
                proposed = dict(candidate)
                proposed[component] = revised
                self.exploration.record(
                    "propose",
                    predictors_src=proposed["predictors_src"],
                    forward_src=proposed["forward_src"],
                    signature_hash=self._signature_hash,
                    parents=[parent_id],
                    extra={"changed_component": component},
                )

        return results

    def _record_evaluation(self, candidate: dict[str, str], scores: list[float]) -> None:
        if not scores:
            return
        mean = sum(scores) / len(scores)
        self.exploration.record(
            "evaluate",
            predictors_src=candidate.get("predictors_src"),
            forward_src=candidate.get("forward_src"),
            signature_hash=self._signature_hash,
            score=mean,
            extra={"n_examples": len(scores)},
        )


def _format_failures(records: list[dict[str, Any]]) -> str:
    if not records:
        return "(no failing examples available)"
    chunks: list[str] = []
    for i, rec in enumerate(records):
        chunks.append(
            f"=== Example {i} ===\n"
            f"Inputs:\n{rec.get('Inputs')!r}\n"
            f"Generated Outputs:\n{rec.get('Generated Outputs')!r}\n"
            f"Feedback:\n{rec.get('Feedback')}"
        )
    return "\n\n".join(chunks)


@experimental(version="3.3.0b2")
class FlexGEPA(Teleprompter):
    """Reflective evolutionary optimizer for the source code of a `dspy.Flex` module.

    Wraps the external ``gepa`` package's optimization loop with a custom
    `FlexAdapter` whose candidate shape is
    ``{"predictors_src": str, "forward_src": str}``. The reflection LM proposes
    revised source per component which replaces the prior version verbatim.

    Args:
        metric: Standard DSPy metric ``(example, prediction, trace=None) -> float | ScoreWithFeedback``.
        reflection_lm: ``dspy.LM`` used to author code revisions. Prefer a strong
            model (e.g. ``dspy.LM("openai/gpt-5", temperature=1.0)``).
        proposer_type: DSPy primitive class wrapping the reflection LM. Defaults
            to ``dspy.Predict``. Use ``dspy.ChainOfThought`` for more deliberate
            reflection or ``dspy.ReAct`` for tool-using reflection (in which case
            pass ``proposer_kwargs={"tools": [...]}``).
        proposer_kwargs: Extra constructor kwargs for ``proposer_type``. Most
            useful for ``dspy.ReAct`` (``{"tools": [...]}``).
        auto / max_full_evals / max_metric_calls: Budget. Exactly one must be set.
        reflection_minibatch_size: How many examples per reflection step.
        candidate_selection_strategy: ``"pareto"`` (default) or ``"current_best"``.
        component_selector: Which components to update each iteration —
            ``"round_robin"`` (default) or ``"all"``.
        use_merge / max_merge_invocations: GEPA merge configuration.
        num_threads: Evaluation parallelism.
        failure_score / perfect_score: Score range.
        log_dir / track_stats / seed: Logging + reproducibility.
        gepa_kwargs: Passthrough to ``gepa.optimize``.
    """

    def __init__(
        self,
        metric: Callable,
        *,
        reflection_lm: LM | None = None,
        proposer_type: type = dspy.Predict,
        proposer_kwargs: dict[str, Any] | None = None,
        auto: Literal["light", "medium", "heavy"] | None = None,
        max_full_evals: int | None = None,
        max_metric_calls: int | None = None,
        reflection_minibatch_size: int = 3,
        candidate_selection_strategy: Literal["pareto", "current_best"] = "pareto",
        component_selector: str = "round_robin",
        use_merge: bool = True,
        max_merge_invocations: int | None = 5,
        num_threads: int | None = None,
        failure_score: float = 0.0,
        perfect_score: float = 1.0,
        skip_perfect_score: bool = True,
        log_dir: str | None = None,
        track_stats: bool = False,
        seed: int | None = 0,
        gepa_kwargs: dict | None = None,
    ):
        budgets_set = sum(b is not None for b in (auto, max_full_evals, max_metric_calls))
        if budgets_set != 1:
            raise ValueError(
                "Exactly one of `auto`, `max_full_evals`, or `max_metric_calls` must be set, "
                f"got auto={auto!r}, max_full_evals={max_full_evals!r}, max_metric_calls={max_metric_calls!r}."
            )

        self.metric_fn = metric
        self.reflection_lm = reflection_lm
        self.proposer_type = proposer_type
        self.proposer_kwargs = proposer_kwargs or {}
        self.auto = auto
        self.max_full_evals = max_full_evals
        self.max_metric_calls = max_metric_calls
        self.reflection_minibatch_size = reflection_minibatch_size
        self.candidate_selection_strategy = candidate_selection_strategy
        self.component_selector = component_selector
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations
        self.num_threads = num_threads
        self.failure_score = failure_score
        self.perfect_score = perfect_score
        self.skip_perfect_score = skip_perfect_score
        self.log_dir = log_dir
        self.track_stats = track_stats
        self.seed = seed
        self.gepa_kwargs = gepa_kwargs or {}

    def _auto_budget(self, num_candidates: int, valset_size: int) -> int:
        """Mirrors :meth:`dspy.GEPA.auto_budget` for the 2-component case."""
        num_components = len(_COMPONENTS)
        num_trials = int(max(2 * (num_components * 2) * math.log2(num_candidates), 1.5 * num_candidates))
        minibatch_size = 35
        full_eval_steps = 5
        V = valset_size
        N = num_trials
        M = minibatch_size
        m = full_eval_steps

        total = V
        total += num_candidates * 5
        total += N * M
        if N == 0:
            return total
        periodic_fulls = (N + 1) // m + 1
        extra_final = 1 if N < m else 0
        total += (periodic_fulls + extra_final) * V
        return total

    def compile(  # type: ignore[override]
        self,
        student: Flex,
        *,
        trainset: list[Example],
        teacher: Any = None,
        valset: list[Example] | None = None,
    ) -> Flex:
        if teacher is not None:
            raise ValueError("FlexGEPA does not accept a `teacher` module.")
        if not isinstance(student, Flex):
            raise TypeError(f"FlexGEPA.compile() requires a dspy.Flex student, got {type(student).__name__}.")
        if not trainset:
            raise ValueError("`trainset` must be a non-empty list of dspy.Example.")
        if student.predictors_src is None or student.forward_src is None:
            raise ValueError(
                "Flex student is not yet implemented; call `.implement()` or run it once before optimizing."
            )

        valset = valset or trainset
        if valset is trainset:
            logger.warning(
                "FlexGEPA: no valset provided; using trainset as valset. For better generalization, "
                "pass a small held-out valset."
            )

        if self.auto is not None:
            self.max_metric_calls = self._auto_budget(
                num_candidates=AUTO_RUN_SETTINGS[self.auto]["n"], valset_size=len(valset)
            )
        elif self.max_full_evals is not None:
            self.max_metric_calls = self.max_full_evals * (len(trainset) + len(valset))

        rng = random.Random(self.seed)

        adapter = FlexAdapter(
            flex_module=student,
            metric_fn=self.metric_fn,
            reflection_lm=self.reflection_lm,
            proposer_type=self.proposer_type,
            proposer_kwargs=self.proposer_kwargs,
            failure_score=self.failure_score,
            num_threads=self.num_threads,
            rng=rng,
        )

        seed_candidate: dict[str, str] = {
            "predictors_src": student.predictors_src,
            "forward_src": student.forward_src,
        }

        reflection_callable: Callable[[str], str] | None = None
        if isinstance(self.reflection_lm, LM):
            captured_lm = self.reflection_lm

            def _call(x: str) -> str:
                raw = captured_lm(x)
                if isinstance(raw, list) and raw:
                    head = raw[0]
                    if isinstance(head, dict):
                        return head.get("text", "")
                    return str(head)
                return str(raw)

            reflection_callable = _call

        optimize_kwargs: dict[str, Any] = dict(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection_callable,
            candidate_selection_strategy=self.candidate_selection_strategy,
            skip_perfect_score=self.skip_perfect_score,
            reflection_minibatch_size=self.reflection_minibatch_size,
            module_selector=self.component_selector,
            perfect_score=self.perfect_score,
            use_merge=self.use_merge,
            max_merge_invocations=self.max_merge_invocations,
            max_metric_calls=self.max_metric_calls,
            logger=LoggerAdapter(logger),
            run_dir=self.log_dir,
            display_progress_bar=True,
            raise_on_exception=True,
            seed=self.seed,
        )
        optimize_kwargs.update(self.gepa_kwargs)
        gepa_result: GEPAResult = optimize(**optimize_kwargs)

        best = cast(dict, gepa_result.best_candidate)
        new_prog = adapter.build_program(best)
        best_score = max(gepa_result.val_aggregate_scores) if gepa_result.val_aggregate_scores else None

        if new_prog._persist_to is not None:
            sig_hash = new_prog._signature_hash()
            new_prog._write_persisted(best["predictors_src"], best["forward_src"], sig_hash)
            parents_at_best: list[int] = []
            if gepa_result.parents:
                parents_at_best = [p for p in gepa_result.parents[gepa_result.best_idx] if p is not None]
            version_id = ManifestStore(new_prog._flex_root).append_version(
                flex_id=new_prog._flex_id,
                src_path=new_prog._persist_to,
                signature_hash=sig_hash,
                score=best_score,
                parents=parents_at_best,
                notes="FlexGEPA optimized",
            )
            adapter.exploration.record(
                "accept",
                predictors_src=best["predictors_src"],
                forward_src=best["forward_src"],
                signature_hash=sig_hash,
                score=best_score,
                extra={"version_id": version_id, "src_path": str(new_prog._persist_to)},
            )

        if self.track_stats:
            new_prog.detailed_results = gepa_result

        return new_prog
