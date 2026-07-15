from __future__ import annotations

import inspect
import logging
from typing import Any

from gepa import EvaluationBatch

import dspy
from dspy.primitives import Prediction
from dspy.teleprompt.bootstrap_trace import FailedPrediction

logger = logging.getLogger(__name__)

_CODE_KEY_SUFFIX = "::code"


def is_code_key(key: str) -> bool:
    return key.endswith(_CODE_KEY_SUFFIX)


def make_code_key(path: str) -> str:
    return f"{path}{_CODE_KEY_SUFFIX}"


def code_key_path(key: str) -> str:
    return key[: -len(_CODE_KEY_SUFFIX)]


def enumerate_flex_submodules(root) -> dict[str, Any]:
    """Map submodule path -> module for every code-optimizable (``dspy.Flex``) submodule."""
    out: dict[str, Any] = {}
    for name, sub in root.named_sub_modules():
        if getattr(sub, "_code_optimizable", False) and hasattr(sub, "_bind_code"):
            out[name] = sub
    return out


def flex_internal_predictor_ids(flex_submodules: dict[str, Any]) -> set[int]:
    """Object ids of predictors owned by ``dspy.Flex`` submodules."""
    ids: set[int] = set()
    for flex in flex_submodules.values():
        for _, pred in flex.named_predictors():
            ids.add(id(pred))
    return ids


def flex_task_context(student_module) -> tuple[dict[str, str], dict[str, str]]:
    """Per flex submodule path, the ``(task_description, available_context)`` shown to the code
    proposer, keyed by the path used in the submodule's code candidate key.
    """
    task_descriptions: dict[str, str] = {}
    context_blurbs: dict[str, str] = {}
    for path, sub in enumerate_flex_submodules(student_module).items():
        ctx = getattr(sub, "_flex_ctx", None)
        if ctx is not None and hasattr(ctx, "render_signature_spec"):
            task_descriptions[path] = ctx.render_signature_spec()
            sandboxed = getattr(sub, "_bridge", None) is not None
            context_blurbs[path] = ctx.render_context_blurb(sandboxed=sandboxed)
    return task_descriptions, context_blurbs


def _short_repr(value: Any, max_len: int = 800) -> str:
    s = str(value)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _render_prediction(prediction: Any) -> Any:
    if isinstance(prediction, FailedPrediction):
        return f"FailedPrediction: {prediction.completion_text[:1000]}"
    if isinstance(prediction, Prediction):
        try:
            return {k: _short_repr(v) for k, v in prediction.items()}
        except Exception:
            pass
    return _short_repr(prediction)


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


class CodeProposalSignature(dspy.Signature):
    """Revise the full source code of a flex-marked dspy.Flex submodule.

    You receive the submodule's task description (its Signature), the available context (any tools
    and style notes), the catalog of allowed primitives, the module's current source, and a batch
    of failing examples with feedback. Produce a revised source that fixes the observed failures
    and follows the catalog.

    The source is ONE ``dspy.Module`` subclass with two coupled methods, and you MUST output the
    entire, internally-consistent class:
      1. ``def __init__(self):`` calling ``super().__init__()`` and assigning the predictors it
         needs. Pick the simplest primitive that fits each step: ``dspy.Predict("...")`` for a
         direct call (the common default), ``dspy.ChainOfThought("...")`` when explicit reasoning
         helps, and ``dspy.RLM`` / ``dspy.ReAct`` when the step must call tools or explore a
         large/structured input. Assign no predictors at all if the task needs no LM.
      2. ``def forward(self, **inputs):`` that calls those predictors as ``self.<name>`` and
         returns ``dspy.Prediction(<output fields>=...)``.
    Because ``forward`` calls predictors by name, never rename a predictor in one place without
    updating the other.

    Tools are OPTIONAL — use them only when a step genuinely needs one; many good modules are just
    a ``dspy.Predict`` or two plus plain Python. When you do use tools, they come from two places.
    (1) Any listed in ``available_context`` are in scope by name — wire the useful ones into
    ``dspy.RLM(..., tools=[...])`` / ``dspy.ReAct(..., tools=[...])`` or call them directly
    (reference them by the exact names; do not import or redefine them). If ``available_context``
    is '(no extra context)', no tools were provided — don't reference any.
    (2) AUTHOR your own: when a sub-step needs a capability the provided tools don't cover, define
    a documented function inside ``__init__`` and pass it via ``tools=[...]``. Tools you author
    live in this source, so they are optimized and persisted exactly like the rest of the code.
    """

    task_description: str = dspy.InputField(
        desc="The submodule's Signature: name, objective, input and output fields."
    )
    available_context: str = dspy.InputField(
        desc="Tools (in scope by name) and style notes available to the module. May be "
        "'(no extra context)'."
    )
    primitives_catalog: str = dspy.InputField(
        desc="Catalog of allowed primitives and conventions the revised code must follow."
    )
    current_source: str = dspy.InputField(
        desc="The module's current full source: one dspy.Module subclass (its __init__ and forward)."
    )
    failures: str = dspy.InputField(
        desc="A batch of failing examples and feedback. Diagnose them and revise the module to fix them."
    )
    revised_source: str = dspy.OutputField(
        desc="The full revised module source: one `dspy.Module` subclass with `__init__` (predictors) and `forward`."
    )


def propose_code(
    code_keys: list[str],
    candidate: dict[str, str],
    reflective_dataset: dict[str, list[dict[str, Any]]],
    task_descriptions: dict[str, str],
    context_blurbs: dict[str, str],
    reflection_lm,
) -> dict[str, str]:
    """Propose a revised ``module_src`` for each code component; keep the original on failure."""
    from dspy.flex.ctx import _strip_code_fences
    from dspy.flex.primitives_doc import PRIMITIVES_CATALOG

    results: dict[str, str] = {}
    proposer = dspy.Predict(CodeProposalSignature)
    with dspy.context(lm=reflection_lm):
        for ckey in code_keys:
            path = code_key_path(ckey)
            try:
                out = proposer(
                    task_description=task_descriptions.get(path, path),
                    available_context=context_blurbs.get(path, "(no extra context)"),
                    primitives_catalog=PRIMITIVES_CATALOG,
                    current_source=candidate[ckey],
                    failures=_format_failures(reflective_dataset.get(ckey, [])),
                )
                results[ckey] = _strip_code_fences(out.revised_source)
            except Exception as e:
                logger.warning("Code proposer failed on %s (%s); keeping original.", ckey, e)
                results[ckey] = candidate[ckey]
    return results


def rebind_flex_code(program, candidate: dict[str, str]) -> None:
    """Rebind ``module_src`` on each ``dspy.Flex`` submodule whose code key is in the candidate.

    Raises if a candidate's source is broken; ``DspyAdapter.evaluate`` catches that and scores the
    batch as a failure.
    """
    for path, flex in enumerate_flex_submodules(program).items():
        key = make_code_key(path)
        if key in candidate:
            flex._bind_code(candidate[key])


def code_reflective_records(eval_batch) -> list[dict[str, Any]]:
    """Module-level reflective records for a code component (whole-program I/O).

    Unlike instruction components (per-predictor traces), a flex submodule's code is reflected on
    using the whole program's inputs/outputs/feedback per example.
    """
    records: list[dict[str, Any]] = []
    for data in eval_batch.trajectories or []:
        example = data["example"]
        prediction = data["prediction"]
        raw_score = data.get("score")
        if hasattr(raw_score, "score"):
            feedback = raw_score.get("feedback") or ""
            score_val = raw_score["score"]
        else:
            feedback = ""
            score_val = raw_score
        records.append(
            {
                "Inputs": {k: _short_repr(v) for k, v in example.inputs().items()},
                "Generated Outputs": _render_prediction(prediction),
                "Feedback": feedback or f"This trajectory got a score of {score_val}.",
            }
        )
    return records


def _metric_wants_program_trace(metric_fn) -> bool:
    """True if ``metric_fn`` declares a ``program_trace`` parameter.

    DSPy metrics conventionally treat a non-None third argument (``trace``) as bootstrapping
    mode and return a strict bool, so the execution trace cannot be passed there at scoring
    time without silently coarsening scores. Instead, a metric opts in to receiving the trace
    during flex scoring by declaring ``program_trace=None``.
    """
    try:
        param = inspect.signature(metric_fn).parameters.get("program_trace")
    except (TypeError, ValueError):
        return False
    return param is not None and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)


def evaluate_with_trace(
    program,
    batch,
    *,
    metric_fn,
    num_threads,
    failure_score,
    callback_metadata,
    capture_traces,
) -> EvaluationBatch:
    """Trace-capturing evaluation used when a flex submodule is present.

    A metric that declares a ``program_trace`` parameter receives the execution trace at
    scoring time (e.g. to penalize LM calls and reward deterministic code); its ``trace``
    argument stays None, preserving the eval-mode semantics of non-Flex GEPA scoring.
    """
    from dspy.teleprompt import bootstrap_trace as bootstrap_trace_module

    trajs = bootstrap_trace_module.bootstrap_trace_data(
        program=program,
        dataset=batch,
        metric=None,  # capture traces only; we score with the trace just below
        num_threads=num_threads,
        raise_on_error=False,
        capture_failed_parses=True,
        failure_score=failure_score,
        format_failure_score=failure_score,
        callback_metadata=callback_metadata,
    )
    wants_trace = _metric_wants_program_trace(metric_fn)
    outputs: list[Any] = [None] * len(batch)
    scores: list[float] = [failure_score] * len(batch)
    for t in trajs:
        pred = t["prediction"]
        outputs[t["example_ind"]] = pred
        if isinstance(pred, FailedPrediction):
            result = failure_score
        elif wants_trace:
            result = metric_fn(t["example"], pred, program_trace=t["trace"])
        else:
            result = metric_fn(t["example"], pred)
        t["score"] = result  # make_reflective_dataset reads this for the (trace-aware) feedback
        score = result["score"] if hasattr(result, "score") else result
        scores[t["example_ind"]] = failure_score if score is None else score
    return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajs if capture_traces else None)
