"""Helpers for executing metrics and resolving their scores."""

from __future__ import annotations

import warnings
from typing import Any, Callable

from ._subscores import (
    Score,
    SubscoreExpr,
    SubscoreValue,
    _begin_collect,
    _end_collect,
    finalize_scores,
)


def resolve_metric_score(
    metric: Callable[..., Any],
    *args: Any,
    context: str = "metric",
    warn_on_expression: bool = True,
    ctx_info: dict[str, Any] | None = None,
) -> tuple[Score, dict[str, Any]]:
    """Run ``metric`` under the subscore collector and normalize the result.

    Parameters
    ----------
    metric:
        Callable metric to execute. The callable is invoked as ``metric(*args)``.
    *args:
        Positional arguments to forward to ``metric``.
    context:
        Human-readable context used in warning messages.
    warn_on_expression:
        Whether to emit runtime warnings if the metric returns a raw
        ``SubscoreExpr``/``SubscoreValue`` that needs to be resolved.
    ctx_info:
        Optional context dictionary that will be folded into the resulting
        ``Score.info`` via :func:`finalize_scores`.

    Returns
    -------
    tuple[Score, dict[str, Any]]
        The resolved ``Score`` object and any auxiliary metadata returned next
        to the score (e.g., extra fields from a ``dspy.Prediction``).
    """

    token = _begin_collect()
    try:
        raw_output = metric(*args)
    finally:
        collector = _end_collect(token)

    value, metadata = _unwrap_metric_output(
        raw_output,
        context=context,
        warn_on_expression=warn_on_expression,
    )

    score = finalize_scores(value, collector, ctx_info=ctx_info)
    return score, metadata


def _unwrap_metric_output(
    raw_output: Any,
    *,
    context: str,
    warn_on_expression: bool,
) -> tuple[Any, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    value = raw_output

    prediction = _as_prediction(raw_output)
    if prediction is not None:
        if not hasattr(prediction, "score"):
            raise ValueError(
                f"{context}: metric returned a Prediction without a `score` field."
            )
        value = prediction.score
        metadata = {k: v for k, v in prediction.items() if k != "score"}
    elif isinstance(raw_output, dict) and "score" in raw_output:
        metadata = {k: v for k, v in raw_output.items() if k != "score"}
        value = raw_output["score"]
    else:
        attr_score = getattr(raw_output, "score", None)
        if attr_score is not None:
            value = attr_score
            if hasattr(raw_output, "items"):
                try:
                    metadata = {k: v for k, v in raw_output.items() if k != "score"}
                except Exception:
                    metadata = {}

    if value is None:
        raise ValueError(f"{context}: metric did not return a score.")

    if warn_on_expression and isinstance(value, SubscoreExpr):
        warnings.warn(
            f"{context}: metric returned a subscore expression; resolving to a Score.",
            RuntimeWarning,
            stacklevel=3,
        )
    elif warn_on_expression and isinstance(value, SubscoreValue):
        warnings.warn(
            f"{context}: metric returned a subscore component; resolving to a Score.",
            RuntimeWarning,
            stacklevel=3,
        )

    return value, metadata


def _as_prediction(obj: Any) -> Any | None:
    try:
        from dspy.primitives.prediction import Prediction  # type: ignore
    except Exception:  # pragma: no cover - defensive import guard
        return None

    return obj if isinstance(obj, Prediction) else None

