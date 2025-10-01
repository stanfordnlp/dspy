"""Utilities for declaring and aggregating metric subscores.

This module implements the tiny algebra used to trace metric subscores and
resolve arithmetic expressions that combine them.  Users interact with the
`subscore` helper, which behaves like a float while carrying metadata and a
stable name.  Internally, we track subscores in a ``ContextVar`` so concurrent
evaluations remain isolated.

The module also exposes a :class:`Score` dataclass that carries the resolved
scalar score, the subscores, and any auxiliary information collected during
evaluation (e.g., canonical expressions, metadata, usage statistics).
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Any, Iterable

import contextvars

__all__ = [
    "Score",
    "subscore",
    "subscore_abs",
    "subscore_min",
    "subscore_max",
    "subscore_clip",
    "_begin_collect",
    "_end_collect",
    "finalize_scores",
]


@dataclasses.dataclass(frozen=True)
class Score:
    """Resolved metric information.

    Attributes
    ----------
    scalar:
        The scalar score obtained from evaluating the user's arithmetic
        expression.  For backwards compatibility this behaves exactly like the
        previous single metric number.
    subscores:
        Mapping from subscore name to numeric value.  Each entry corresponds to
        a call to :func:`subscore` that was used in the returned expression (or
        registered during metric execution even if the user coerced it to a
        float early).
    info:
        A free-form dictionary used to carry auxiliary details such as the
        canonical expression string, per-subscore metadata, latency, usage, etc.
    """

    scalar: float
    subscores: dict[str, float] = dataclasses.field(default_factory=dict)
    info: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - trivial conversions
        object.__setattr__(self, "scalar", float(self.scalar))
        subscores = {name: float(value) for name, value in self.subscores.items()}
        object.__setattr__(self, "subscores", subscores)


class SubscoreExpr:
    """Tiny expression DAG that keeps arithmetic over subscores traceable."""

    __slots__ = ("op", "args")

    def __init__(self, op: str, args: tuple[Any, ...]):
        self.op = op
        self.args = args

    # Arithmetic ---------------------------------------------------------
    def __add__(self, other: Any) -> "SubscoreExpr":
        return SubscoreExpr("add", (self, other))

    def __radd__(self, other: Any) -> "SubscoreExpr":
        return SubscoreExpr("add", (other, self))

    def __sub__(self, other: Any) -> "SubscoreExpr":
        return SubscoreExpr("sub", (self, other))

    def __rsub__(self, other: Any) -> "SubscoreExpr":
        return SubscoreExpr("sub", (other, self))

    def __mul__(self, other: Any) -> "SubscoreExpr":
        return SubscoreExpr("mul", (self, other))

    def __rmul__(self, other: Any) -> "SubscoreExpr":
        return SubscoreExpr("mul", (other, self))

    def __truediv__(self, other: Any) -> "SubscoreExpr":
        return SubscoreExpr("div", (self, other))

    def __rtruediv__(self, other: Any) -> "SubscoreExpr":
        return SubscoreExpr("div", (other, self))

    def __pow__(self, other: Any) -> "SubscoreExpr":
        return SubscoreExpr("pow", (self, other))

    def __rpow__(self, other: Any) -> "SubscoreExpr":
        return SubscoreExpr("pow", (other, self))

    def __neg__(self) -> "SubscoreExpr":
        return SubscoreExpr("neg", (self,))

    def __float__(self) -> float:
        return float(self.eval())

    # Comparisons --------------------------------------------------------
    def _cmp(self, other: Any, op: str) -> bool:
        left = self.eval()
        right = _eval_node(other)
        if op == "lt":
            return left < right
        if op == "le":
            return left <= right
        if op == "gt":
            return left > right
        if op == "ge":
            return left >= right
        raise ValueError(f"Unsupported comparison op: {op}")

    def __lt__(self, other: Any) -> bool:
        return self._cmp(other, "lt")

    def __le__(self, other: Any) -> bool:
        return self._cmp(other, "le")

    def __gt__(self, other: Any) -> bool:
        return self._cmp(other, "gt")

    def __ge__(self, other: Any) -> bool:
        return self._cmp(other, "ge")

    def __bool__(self) -> bool:
        return bool(self.eval())

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

    def used_subscores(self) -> dict[str, float]:
        subscores: dict[str, float] = {}
        _collect_subscores(self, subscores)
        return subscores

    def to_repr(self) -> str:
        return _repr_node(self)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"SubscoreExpr({self.to_repr()})"


class SubscoreValue(SubscoreExpr):
    """Leaf node representing a named subscore."""

    __slots__ = ("name", "value", "meta")

    def __init__(self, name: str, value: float, meta: dict[str, Any]):
        self.name = name
        self.value = float(value)
        self.meta = meta
        super().__init__("subscore", (self,))

    # SubscoreValue inherits operator overloads from SubscoreExpr via SubscoreExpr methods.

    def eval(self) -> float:
        return self.value

    def used_subscores(self) -> dict[str, float]:
        return {self.name: self.value}

    def to_repr(self) -> str:
        return self.name

    def __float__(self) -> float:
        return self.value


@dataclasses.dataclass
class _Collector:
    subscores: dict[str, tuple[float, dict[str, Any]]] = dataclasses.field(default_factory=dict)


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


def subscore(name: str, value: float, /, **meta: Any) -> SubscoreValue:
    """Register a named subscore for the active metric evaluation."""

    if not isinstance(name, str) or not name:
        raise ValueError("subscore name must be a non-empty string")
    numeric = float(value)
    collector = _CONTEXT.get()
    if collector is not None:
        if name in collector.subscores:
            raise ValueError(f"Duplicate subscore name: {name}")
        collector.subscores[name] = (numeric, dict(meta))
    return SubscoreValue(name, numeric, dict(meta))


def subscore_abs(value: Any) -> SubscoreExpr:
    return SubscoreExpr("abs", (_ensure_expr(value),))


def subscore_min(*values: Any) -> SubscoreExpr:
    if not values:
        raise ValueError("subscore_min requires at least one argument")
    return SubscoreExpr("min", tuple(_ensure_expr(v) for v in values))


def subscore_max(*values: Any) -> SubscoreExpr:
    if not values:
        raise ValueError("subscore_max requires at least one argument")
    return SubscoreExpr("max", tuple(_ensure_expr(v) for v in values))


def subscore_clip(value: Any, lower: float | None = None, upper: float | None = None) -> SubscoreExpr:
    return SubscoreExpr("clip", (_ensure_expr(value), lower, upper))


def finalize_scores(
    result: Any,
    collector: _Collector,
    *,
    ctx_info: dict[str, Any] | None = None,
) -> Score:
    """Convert a metric return value into a :class:`Score` object."""

    ctx_info = dict(ctx_info or {})
    if isinstance(result, Score):
        info = dict(result.info)
        info.update({k: v for k, v in ctx_info.items() if v is not None})
        return Score(result.scalar, dict(result.subscores), info)

    if isinstance(result, SubscoreValue):
        subscores = result.used_subscores()
        meta = {name: collector.subscores.get(name, (None, {}))[1] for name in subscores}
        info = {"expr": result.to_repr(), "meta": meta}
        info.update({k: v for k, v in ctx_info.items() if v is not None})
        _validate_subscores(subscores)
        return Score(result.value, subscores, info)

    if isinstance(result, SubscoreExpr):
        scalar = float(result.eval())
        subscores = result.used_subscores()
        meta = {name: collector.subscores.get(name, (None, {}))[1] for name in subscores}
        info = {"expr": result.to_repr(), "meta": meta}
        info.update({k: v for k, v in ctx_info.items() if v is not None})
        _validate_subscores(subscores)
        _validate_scalar("scalar", scalar)
        return Score(scalar, subscores, info)

    scalar = float(result)
    subscores = {name: value for name, (value, _) in collector.subscores.items()}
    meta = {name: meta for name, (_, meta) in collector.subscores.items()}
    info = {"expr": None, "meta": meta}
    info.update({k: v for k, v in ctx_info.items() if v is not None})
    _validate_subscores(subscores)
    _validate_scalar("scalar", scalar)
    return Score(scalar, subscores, info)


# Helpers -----------------------------------------------------------------

def _ensure_expr(value: Any) -> Any:
    if isinstance(value, (SubscoreExpr, SubscoreValue)):
        return value
    return value


def _eval_node(node: Any) -> float:
    if isinstance(node, SubscoreExpr):
        return node.eval()
    if isinstance(node, SubscoreValue):
        return node.value
    return float(node)


def _collect_subscores(node: Any, subscores: dict[str, float]) -> None:
    if isinstance(node, SubscoreExpr) and node.op != "subscore":
        for arg in node.args:
            _collect_subscores(arg, subscores)
    elif isinstance(node, SubscoreValue):
        subscores[node.name] = node.value


def _repr_node(node: Any) -> str:
    if isinstance(node, SubscoreValue):
        return node.to_repr()
    if isinstance(node, SubscoreExpr):
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
        if op == "subscore":
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


def _validate_subscores(subscores: Iterable[tuple[str, float]] | dict[str, float]) -> None:
    if isinstance(subscores, dict):
        items = subscores.items()
    else:
        items = subscores
    for name, value in items:
        _validate_scalar(name, value)


def _validate_scalar(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"Non-finite value for {name}: {value!r}")


def coerce_metric_value(value: Any, *, context: str = "metric", warn_on_expression: bool = True) -> Any:
    """Coerce subscore expressions into scalar values for legacy metric handlers.

    Optimizers that do not capture subscores may invoke metrics that return SubscoreExpr/SubscoreValue.
    This helper warns once per call (when desired) and evaluates the expression so callers
    can continue working with numeric scores.
    """

    if value is None:
        return None
    if isinstance(value, Score):
        return value.scalar
    if isinstance(value, SubscoreExpr):
        if warn_on_expression:
            warnings.warn(
                f"{context}: metric returned a subscore expression; coercing to float and discarding subscores.",
                RuntimeWarning,
                stacklevel=3,
            )
        return float(value)
    if isinstance(value, SubscoreValue):
        if warn_on_expression:
            warnings.warn(
                f"{context}: metric returned a subscore component; coercing to float and discarding metadata.",
                RuntimeWarning,
                stacklevel=3,
            )
        return float(value)
    return value
