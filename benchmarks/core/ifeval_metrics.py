"""
IFEval metric helpers built on the vendored Google Research evaluators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dspy

from ifeval_eval import instructions_registry
from ifeval_eval.evaluation_lib import InputExample, test_instruction_following_strict

if TYPE_CHECKING:
    from dspy import Example, Prediction


def _filter_instruction_kwargs(instruction_id: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs accepted by the instruction checker.

    The HF IFEval dataset stores a full kwargs template per instruction; passing
    unused keys (e.g. ``relation``) causes ``build_description`` to fail.
    """
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    allowed_keys = set(instruction_cls(instruction_id).get_instruction_args_keys())
    return {key: value for key, value in kwargs.items() if key in allowed_keys}


def _to_input_example(example: Example) -> InputExample:
    filtered_kwargs = [
        _filter_instruction_kwargs(instruction_id, kw)
        for instruction_id, kw in zip(example.instruction_id_list, example.kwargs, strict=True)
    ]
    return InputExample(
        key=example.key,
        instruction_id_list=list(example.instruction_id_list),
        prompt=example.prompt,
        kwargs=filtered_kwargs,
    )


def _evaluate_example(example: Example, response: str) -> tuple[float, list[str], list[bool]]:
    """Evaluate a single response against IFEval constraints.

    Returns:
        Tuple of (constraint satisfaction rate, instruction ids, per-instruction pass flags).
    """
    inp = _to_input_example(example)
    output = test_instruction_following_strict(inp, {example.prompt: response})
    if not output.follow_instruction_list:
        return 0.0, output.instruction_id_list, output.follow_instruction_list

    satisfied = sum(output.follow_instruction_list)
    total = len(output.follow_instruction_list)
    constraint_rate = satisfied / total if total else 0.0
    return constraint_rate, output.instruction_id_list, output.follow_instruction_list


def _get_response(pred: Prediction) -> str:
    return getattr(pred, "response", "") or ""


def ifeval_metric(example: Example, pred: Prediction, trace: Any = None) -> float:
    """Prompt-level hard satisfaction: 1.0 only if all constraints pass."""
    _, _, follow_list = _evaluate_example(example, _get_response(pred))
    if not follow_list:
        return 0.0
    return 1.0 if all(follow_list) else 0.0


def ifeval_constraint_metric(example: Example, pred: Prediction, trace: Any = None) -> float:
    """Per-example constraint satisfaction rate (0.0 to 1.0)."""
    constraint_rate, _, _ = _evaluate_example(example, _get_response(pred))
    return constraint_rate


def ifeval_metric_gepa(
    gold: Example,
    pred: Prediction,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> Prediction:
    """GEPA-compatible metric with constraint-level feedback."""
    response = _get_response(pred)
    constraint_rate, instruction_ids, follow_list = _evaluate_example(gold, response)
    hard_score = 1.0 if follow_list and all(follow_list) else 0.0

    failed = [
        instruction_id
        for instruction_id, followed in zip(instruction_ids, follow_list, strict=True)
        if not followed
    ]
    if failed:
        feedback = (
            f"Response failed {len(failed)} of {len(follow_list)} verifiable constraints: "
            f"{', '.join(failed)}. Constraint satisfaction rate: {constraint_rate:.2f}."
        )
    else:
        feedback = "All verifiable constraints satisfied."

    return dspy.Prediction(score=hard_score, feedback=feedback)
