"""Bounded provider evidence, not a quota model or a retry policy.

Contract: docs/error-diagnostics.md (2026-09-19). Values retain their vendor
units and duplicates. No credential/header outside the closed set is copied.
"""
from __future__ import annotations

import json
import math
import re
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

RATE_LIMIT_HEADERS = frozenset({
    "retry-after", "retry-after-ms", "x-ms-retry-after-ms",
    "x-ratelimit-type", "x-ratelimit-abusepenalty-active",
    *(f"x-ratelimit-{field}-{unit}"
      for field in ("limit", "remaining", "reset", "renewalperiod")
      for unit in ("requests", "tokens")),
    *(f"anthropic-ratelimit-{unit}-{field}"
      for unit in ("requests", "tokens", "input-tokens", "output-tokens")
      for field in ("limit", "remaining", "reset")),
})


def capture_rate_limits(headers: Iterable[tuple[str, str]]) -> Mapping[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for name, value in headers:
        if not isinstance(name, str) or not isinstance(value, str):
            continue
        name = name.lower()
        if name not in RATE_LIMIT_HEADERS or not 1 <= len(value) <= 256:
            continue
        if any(ord(c) < 32 or ord(c) > 126 for c in value):
            continue
        values = out.setdefault(name, [])
        if len(values) < 4:
            values.append(value)
    return MappingProxyType({name: tuple(values) for name, values in out.items()})


def freeze_rate_limits(snapshot: Mapping[str, Sequence[str]] | None) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(snapshot, Mapping):
        return capture_rate_limits(())
    return capture_rate_limits((name, value) for name, values in snapshot.items()
                               if isinstance(values, (list, tuple)) for value in values)


def milliseconds_seconds(value: object) -> float | None:
    if not isinstance(value, str) or len(value) > 256:
        return None
    if not re.fullmatch(r"\+?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value.strip(), flags=re.ASCII):
        return None
    number = float(value)
    return number / 1000 if math.isfinite(number) and number >= 0 else None


def normalize_http_response(value: object) -> dict:
    """Canonical ErrorDetail handshake block: three optional fields only."""
    if not isinstance(value, dict):
        raise TypeError("ErrorDetail.http_response must be an object")
    if set(value) - {"request_id", "retry_after", "rate_limit_headers"}:
        raise ValueError("unknown ErrorDetail.http_response field")
    out = {}
    request_id = value.get("request_id")
    if request_id is not None:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("http_response.request_id must be a non-empty string")
        out["request_id"] = request_id
    wait = value.get("retry_after")
    if wait is not None:
        if isinstance(wait, bool) or not isinstance(wait, (int, float)) or not math.isfinite(wait) or wait < 0:
            raise ValueError("http_response.retry_after must be finite nonnegative seconds")
        out["retry_after"] = float(wait)
    if "rate_limit_headers" in value:
        headers = value["rate_limit_headers"]
        if not isinstance(headers, dict) or any(not isinstance(v, (list, tuple)) or any(not isinstance(s, str) for s in v) for v in headers.values()):
            raise TypeError("http_response.rate_limit_headers must map names to string arrays")
        snapshot = freeze_rate_limits(headers)
        if snapshot:
            out["rate_limit_headers"] = {k: list(v) for k, v in snapshot.items()}
    return out


def diagnostics_text(snapshot: Mapping[str, Sequence[str]], retry_after: float | None) -> str:
    pieces = []
    if isinstance(retry_after, (int, float)) and not isinstance(retry_after, bool) and math.isfinite(retry_after) and retry_after >= 0:
        pieces.append(f"Retry advice: {retry_after:g} seconds (not a guarantee).")
    if snapshot:
        raw = json.dumps(dict(snapshot), sort_keys=True, ensure_ascii=True)
        if len(raw) > 2048:
            raw = raw[:2048] + "... [full retained values in rate_limit_headers]"
        pieces.append("Provider rate-limit headers (raw; advisory): " + raw)
    return "\n\n  " + "\n  ".join(pieces) if pieces else ""
