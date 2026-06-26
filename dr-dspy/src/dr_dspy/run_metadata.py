"""Reusable run metadata helpers."""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import pydantic

import dspy

DEFAULT_REDACTED_ARG_NAMES = frozenset({"database_url"})
DEFAULT_REDACTED_OPTIONS = ("--database-url",)
REDACTED_VALUE = "<redacted>"

__all__ = ["REDACTED_VALUE", "build_run_metadata"]


class FieldSignature(pydantic.BaseModel):
    name: str
    type: type
    role: Any
    description: str | None = None


def _redact_argv(
    argv: Sequence[str], *, redacted_options: Sequence[str]
) -> list[str]:
    redacted = list(argv)
    option_set = set(redacted_options)
    for i, arg in enumerate(redacted):
        if arg in option_set and i + 1 < len(redacted):
            redacted[i + 1] = REDACTED_VALUE
            continue
        for option in option_set:
            if arg.startswith(f"{option}="):
                redacted[i] = f"{option}={REDACTED_VALUE}"
                break
    return redacted


def _args_payload(
    args: Mapping[str, Any] | object, *, redacted_arg_names: frozenset[str]
) -> dict[str, Any]:
    if isinstance(args, pydantic.BaseModel):
        payload = args.model_dump(mode="json")
    elif isinstance(args, Mapping):
        payload = dict(args)
    else:
        payload = vars(args).copy()
    for name in redacted_arg_names:
        if payload.get(name):
            payload[name] = REDACTED_VALUE
    return payload


def build_run_metadata(
    args: Mapping[str, Any] | object,
    *,
    model_id: str,
    argv: Sequence[str] | None = None,
    redacted_arg_names: frozenset[str] = DEFAULT_REDACTED_ARG_NAMES,
    redacted_options: Sequence[str] = DEFAULT_REDACTED_OPTIONS,
) -> dict[str, Any]:
    """Build JSON-friendly metadata for an experiment run."""
    return {
        "argv": _redact_argv(
            sys.argv if argv is None else argv,
            redacted_options=redacted_options,
        ),
        "args": _args_payload(args, redacted_arg_names=redacted_arg_names),
        "model": model_id,
        "dspy_version": getattr(dspy, "__version__", "?"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
    }
