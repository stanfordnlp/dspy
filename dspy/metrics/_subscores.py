"""Utilities for declaring and aggregating metric subscores.

This module implements the tiny algebra used to trace metric subscores and
resolve arithmetic expressions that combine them.  Users interact with the
`subscore` helper, which behaves like a float while carrying metadata and a
stable name.  Internally, we track subscores in a ``ContextVar`` so concurrent
evaluations remain isolated.

The module also exposes a :class:`Scores` dataclass that carries the aggregate
score, the resolved axes, and any auxiliary information collected during
evaluation (e.g., canonical expressions, metadata, usage statistics).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Iterable

import contextvars

__all__ = [
    "Scores",
    "subscore",
    "axis_abs",
    "axis_min",
    "axis_max",
    "axis_clip",
    "_begin_collect",
    "_end_collect",
    "finalize_scores",
]


@dataclasses.dataclass(frozen=True)
class Scores:
    """Resolved aggregate metric information.

    Attributes
    ----------
    aggregate:
        The scalar score obtained from evaluating the user's arithmetic
        expression.  For backwards compatibility this behaves exactly like the
        previous single metric number.
    axes:
        Mapping from axis name to numeric value.  Each axis corresponds to a
        call to :func:`subscore` that was used in the returned expression (or
        registered during metric execution even if the user coerced it to a
        float early).
    info:
        A free-form dictionary used to carry auxiliary details such as the
        canonical expression string, per-axis metadata, latency, usage, etc.
    """

    aggregate: float
    axes: dict[str, float] = dataclasses.field(default_factory=dict)
    info: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - trivial conversions
        object.__setattr__(self, "aggregate", float(self.aggregate))
        axes = {name: float(value) for name, value in self.axes.items()}
        object.__setattr__(self, "axes", axes)


class AxisExpr:
    """Tiny expression DAG that keeps arithmetic over subscores traceable."""

    __slots__ = ("op", "args")

    def __init__(self, op: str, args: tuple[Any, ...]):
        self.op = op
        self.args = args

    # Arithmetic ---------------------------------------------------------
    def __add__(self, other: Any) -> "AxisExpr":
        return AxisExpr("add", (self, other))

    def __radd__(self, other: Any) -> "AxisExpr":
        return AxisExpr("add", (other, self))

    def __sub__(self, other: Any) -> "AxisExpr":
        return AxisExpr("sub", (self, other))

    def __rsub__(self, other: Any) -> "AxisExpr":
        return AxisExpr("sub", (other, self))

    def __mul__(self, other: Any) -> "AxisExpr":
        return AxisExpr("mul", (self, other))

    def __rmul__(self, other: Any) -> "AxisExpr":
        return AxisExpr("mul", (other, self))

    def __truediv__(self, other: Any) -> "AxisExpr":
        return AxisExpr("div", (self, other))

    def __rtruediv__(self, other: Any) -> "AxisExpr":
        return AxisExpr("div", (other, self))

    def __pow__(self, other: Any) -> "AxisExpr":
        return AxisExpr("pow", (self, other))

    def __rpow__(self, other: Any) -> "AxisExpr":
        return AxisExpr("pow", (other, self))

    def __neg__(self) -> "AxisExpr":
        return AxisExpr("neg", (self,))

    def __float__(self) -> float:
        return float(self.eval())

    # Evaluation ---------------------------------------------------------
    def eval(self) -> float:
        args = [ _eval_node(arg) for arg in self.args ]
        op = self.op
        if op == "add":
            return args[0] + args[1]
        if op == "sub":
            return args[0] - args[1]
        if op == "mul":
            return args[0] * args[1]
        if op == "div":
            return args[0] / args[1]
        if op == "pow":
            return args[0] ** args[1]
        if op == "neg":
            return -args[0]
        if op == "abs":
            return abs(args[0])
        if op == "min":
            return min(args)
        if op == "max":
            return max(args)
        if op == "clip":
            value, lower, upper = args
            if lower is not None and value < lower:
                value = lower
            if upper is not None and value > upper:
                value = upper
            return value
        raise ValueError(f"Unsupported operator: {op!r}")

    def used_axes(self) -> dict[str, float]:
        axes: dict[str, float] = {}
        _collect_axes(self, axes)
        return axes

    def to_repr(self) -> str:
        return _repr_node(self)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"AxisExpr({self.to_repr()})"


class AxisValue(AxisExpr):
    """Leaf node representing a named axis."""

    __slots__ = ("name", "value", "meta")

    def __init__(self, name: str, value: float, meta: dict[str, Any]):
        self.name = name
        self.value = float(value)
        self.meta = meta
        super().__init__("axis", (self,))

    # AxisValue inherits operator overloads from AxisExpr via AxisExpr methods.

    def eval(self) -> float:
        return self.value

    def used_axes(self) -> dict[str, float]:
        return {self.name: self.value}

    def to_repr(self) -> str:
        return self.name

    def __float__(self) -> float:
        return self.value


@dataclasses.dataclass
class _Collector:
    axes: dict[str, tuple[float, dict[str, Any]]] = dataclasses.field(default_factory=dict)


_CONTEXT: contextvars.ContextVar[_Collector | None] = contextvars.ContextVar(
    "_dspy_subscores", default=None
)


def _begin_collect() -> contextvars.Token:
    token = _CONTEXT.set(_Collector())
    return token


def _end_collect(token: contextvars.Token) -> _Collector:
    collector = _CONTEXT.get()
    _CONTEXT.reset(token)
    return collector or _Collector()


def subscore(name: str, value: float, /, **meta: Any) -> AxisValue:
    """Register a named axis for the active metric evaluation."""

    if not isinstance(name, str) or not name:
        raise ValueError("subscore name must be a non-empty string")
    numeric = float(value)
    collector = _CONTEXT.get()
    if collector is not None:
        if name in collector.axes:
            raise ValueError(f"Duplicate subscore name: {name}")
        collector.axes[name] = (numeric, dict(meta))
    return AxisValue(name, numeric, dict(meta))


def axis_abs(value: Any) -> AxisExpr:
    return AxisExpr("abs", (_ensure_expr(value),))


def axis_min(*values: Any) -> AxisExpr:
    if not values:
        raise ValueError("axis_min requires at least one argument")
    return AxisExpr("min", tuple(_ensure_expr(v) for v in values))


def axis_max(*values: Any) -> AxisExpr:
    if not values:
        raise ValueError("axis_max requires at least one argument")
    return AxisExpr("max", tuple(_ensure_expr(v) for v in values))


def axis_clip(value: Any, lower: float | None = None, upper: float | None = None) -> AxisExpr:
    return AxisExpr("clip", (_ensure_expr(value), lower, upper))


def finalize_scores(
    result: Any,
    collector: _Collector,
    *,
    ctx_info: dict[str, Any] | None = None,
) -> Scores:
    """Convert a metric return value into a :class:`Scores` object."""

    ctx_info = dict(ctx_info or {})
    if isinstance(result, Scores):
        info = dict(result.info)
        info.update({k: v for k, v in ctx_info.items() if v is not None})
        return Scores(result.aggregate, dict(result.axes), info)

    if isinstance(result, AxisValue):
        axes = result.used_axes()
        meta = {name: collector.axes.get(name, (None, {}))[1] for name in axes}
        info = {"expr": result.to_repr(), "meta": meta}
        info.update({k: v for k, v in ctx_info.items() if v is not None})
        _validate_axes(axes)
        return Scores(result.value, axes, info)

    if isinstance(result, AxisExpr):
        aggregate = float(result.eval())
        axes = result.used_axes()
        meta = {name: collector.axes.get(name, (None, {}))[1] for name in axes}
        info = {"expr": result.to_repr(), "meta": meta}
        info.update({k: v for k, v in ctx_info.items() if v is not None})
        _validate_axes(axes)
        _validate_scalar("aggregate", aggregate)
        return Scores(aggregate, axes, info)

    aggregate = float(result)
    axes = {name: value for name, (value, _) in collector.axes.items()}
    meta = {name: meta for name, (_, meta) in collector.axes.items()}
    info = {"expr": None, "meta": meta}
    info.update({k: v for k, v in ctx_info.items() if v is not None})
    _validate_axes(axes)
    _validate_scalar("aggregate", aggregate)
    return Scores(aggregate, axes, info)


# Helpers -----------------------------------------------------------------

def _ensure_expr(value: Any) -> Any:
    if isinstance(value, (AxisExpr, AxisValue)):
        return value
    return value


def _eval_node(node: Any) -> float:
    if isinstance(node, AxisExpr):
        return node.eval()
    if isinstance(node, AxisValue):
        return node.value
    return float(node)


def _collect_axes(node: Any, axes: dict[str, float]) -> None:
    if isinstance(node, AxisExpr) and node.op != "axis":
        for arg in node.args:
            _collect_axes(arg, axes)
    elif isinstance(node, AxisValue):
        axes[node.name] = node.value


def _repr_node(node: Any) -> str:
    if isinstance(node, AxisValue):
        return node.to_repr()
    if isinstance(node, AxisExpr):
        op = node.op
        args = node.args
        if op == "neg":
            return f"-({_repr_node(args[0])})"
        if op == "abs":
            return f"abs({_repr_node(args[0])})"
        if op == "min":
            return "min(" + ", ".join(_repr_node(a) for a in args) + ")"
        if op == "max":
            return "max(" + ", ".join(_repr_node(a) for a in args) + ")"
        if op == "clip":
            value, lower, upper = args
            return f"clip({_repr_node(value)}, {lower!r}, {upper!r})"
        if op == "axis":
            return _repr_node(args[0])
        symbol = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "div": "/",
            "pow": "**",
        }[op]
        if op == "sub" and len(args) == 1:
            return f"-({_repr_node(args[0])})"
        left = _repr_node(args[0])
        right = _repr_node(args[1])
        return f"({left}{symbol}{right})"
    if isinstance(node, str):
        return node
    return repr(node)


def _validate_axes(axes: Iterable[tuple[str, float]] | dict[str, float]) -> None:
    if isinstance(axes, dict):
        items = axes.items()
    else:
        items = axes
    for name, value in items:
        _validate_scalar(name, value)


def _validate_scalar(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"Non-finite value for {name}: {value!r}")

