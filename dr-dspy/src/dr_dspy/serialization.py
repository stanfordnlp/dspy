"""DSPy-aware serialization helpers for experiment telemetry."""

from __future__ import annotations

import inspect
import json
from typing import Any

import pydantic

import dspy

PAYLOAD_MAX_BYTES = 256 * 1024
REPR_TRUNCATE = 4096
SANITIZE_KEYS = frozenset(
    {"api_key", "api_base", "base_url", "model_list", "authorization"}
)


def sanitize_lm_kwargs(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Strip credential-like keys from an LM kwargs dict before logging."""
    if not kwargs:
        return {}
    return {
        k: ("<redacted>" if k.lower() in SANITIZE_KEYS else v)
        for k, v in kwargs.items()
    }


def _signature_summary(sig_cls: type[dspy.Signature]) -> dict[str, Any]:
    """Summarize a Signature class for logging."""
    try:
        fields_summary = [
            (
                name,
                str(field.annotation),
                (field.json_schema_extra or {}).get("__dspy_field_type")
                if isinstance(field.json_schema_extra, dict)
                else None,
            )
            for name, field in sig_cls.fields.items()
        ]
    except Exception:
        fields_summary = []
    return {
        "signature": getattr(sig_cls, "signature", repr(sig_cls)),
        "instructions": getattr(sig_cls, "instructions", ""),
        "fields": fields_summary,
    }


def _to_jsonable_inner(x: Any, depth: int = 0) -> Any:
    """Recursive, depth-bounded worker for to_jsonable."""
    if depth > 12:
        return repr(x)[:REPR_TRUNCATE]
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_to_jsonable_inner(v, depth + 1) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable_inner(v, depth + 1) for k, v in x.items()}
    if isinstance(x, bytes):
        return f"<bytes len={len(x)}>"
    if isinstance(x, dspy.Example):
        try:
            return _to_jsonable_inner(x.toDict(), depth + 1)
        except Exception:
            return repr(x)[:REPR_TRUNCATE]
    if isinstance(x, type):
        try:
            if issubclass(x, dspy.Signature):
                return _signature_summary(x)
        except TypeError:
            pass
        return f"<class {x.__module__}.{x.__name__}>"
    if isinstance(x, dspy.BaseLM):
        return {
            "_kind": "BaseLM",
            "class": f"{type(x).__module__}.{type(x).__name__}",
            "model": getattr(x, "model", None),
            "kwargs": sanitize_lm_kwargs(getattr(x, "kwargs", {})),
        }
    if isinstance(x, pydantic.BaseModel):
        try:
            return x.model_dump(mode="json")
        except Exception:
            return repr(x)[:REPR_TRUNCATE]
    if (
        inspect.iscoroutine(x)
        or inspect.isasyncgen(x)
        or inspect.isgenerator(x)
    ):
        return f"<{type(x).__name__}>"
    if hasattr(x, "__dict__") and not callable(x):
        try:
            return {
                k: _to_jsonable_inner(v, depth + 1) for k, v in vars(x).items()
            }
        except Exception:
            return repr(x)[:REPR_TRUNCATE]
    return repr(x)[:REPR_TRUNCATE]


def to_jsonable(x: Any, *, max_bytes: int = PAYLOAD_MAX_BYTES) -> Any:
    """Serialize any object to a JSON-friendly structure. Never raises.

    If the resulting JSON exceeds ``max_bytes``, returns a truncated preview
    wrapped in ``{"_truncated": True, "preview": "..."}``.
    """
    try:
        value = _to_jsonable_inner(x)
        encoded = json.dumps(value, ensure_ascii=False, default=repr)
        if len(encoded.encode("utf-8")) > max_bytes:
            return {"_truncated": True, "preview": encoded[:max_bytes]}
        return value
    except Exception as e:
        return {"_serialize_error": repr(e), "repr": repr(x)[:REPR_TRUNCATE]}
