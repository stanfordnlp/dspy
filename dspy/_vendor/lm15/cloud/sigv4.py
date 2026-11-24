"""
lm15.cloud.sigv4 — AWS Signature Version 4, standard library only.

Rules from the AWS reference (lm15-contract/research/cloud-hosts/sources/
aws-sigv4-create-signed-request.md, lines 46–190), pinned by the AWS test
suite in lm15-contract/auth/sigv4-vectors.json:

- canonical request = method, URI-encoded path, sorted+encoded query,
  lowercase sorted headers (``host`` and every ``x-amz-*`` mandatory;
  ``content-type`` when present), the signed-header list, hex SHA-256 of
  the payload;
- string to sign = ``AWS4-HMAC-SHA256``, ``x-amz-date``, credential scope
  ``YYYYMMDD/region/service/aws4_request``, hex SHA-256 of the canonical
  request;
- signing key = HMAC chain ``AWS4``+secret → date → region → service →
  ``aws4_request``;
- header values are trimmed and inner runs of whitespace collapsed to one
  space (test-suite case ``get-header-value-trim``);
- ``x-amz-content-sha256`` is an S3 requirement and is NOT added here.

Deterministic under a fixed clock and fixed keys: the harness compares the
``authorization`` header byte for byte.
"""

from __future__ import annotations

import hashlib
import hmac
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone

from ..credentials import AwsCredentials

__all__ = ["SigV4Signature", "sign", "canonicalize"]

_ALGORITHM = "AWS4-HMAC-SHA256"
_SAFE = "-_.~"


@dataclass(frozen=True, slots=True, repr=False)
class SigV4Signature:
    canonical_request: str
    string_to_sign: str
    authorization: str
    headers: dict[str, str]  # every header to send, lowercase names, including the new ones


def _encode(text: str) -> str:
    return urllib.parse.quote(text, safe=_SAFE)


def _remove_dot_segments(path: str) -> str:
    """RFC 3986 §5.2.4 as the AWS SDKs apply it to non-S3 paths: drop `.`
    and empty segments, pop on `..`, keep a leading and a trailing slash.
    Pinned by the AWS test-suite vectors get-relative*, get-slash*,
    get-slashes and get-slash-pointless-dot (auth/sigv4-vectors.json)."""
    kept: list[str] = []
    for segment in path.split("/"):
        if segment == "..":
            if kept:
                kept.pop()
        elif segment and segment != ".":
            kept.append(segment)
    first = "/" if path.startswith("/") else ""
    last = "/" if path.endswith("/") and kept else ""
    return first + "/".join(kept) + last


def _canonical_path(path: str) -> str:
    if not path:
        return "/"
    normalized = _remove_dot_segments(path) or "/"
    return "/".join(_encode(urllib.parse.unquote(seg)) for seg in normalized.split("/"))


def _canonical_query(query: str) -> str:
    pairs = urllib.parse.parse_qsl(query, keep_blank_values=True)
    encoded = sorted((_encode(k), _encode(v)) for k, v in pairs)
    return "&".join(f"{k}={v}" for k, v in encoded)


def _trim(value: str) -> str:
    return " ".join(value.split())


def canonicalize(
    method: str,
    url: str,
    headers: dict[str, str],
    payload: bytes,
) -> tuple[str, str]:
    """(canonical_request, signed_headers) for already-complete headers."""
    parts = urllib.parse.urlsplit(url)
    lowered = {k.lower(): _trim(v) for k, v in headers.items()}
    signed = ";".join(sorted(lowered))
    canonical_headers = "".join(f"{k}:{lowered[k]}\n" for k in sorted(lowered))
    canonical = "\n".join([
        method.upper(),
        _canonical_path(parts.path),
        _canonical_query(parts.query),
        canonical_headers,
        signed,
        hashlib.sha256(payload).hexdigest(),
    ])
    return canonical, signed


def sign(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    payload: bytes,
    credentials: AwsCredentials,
    region: str,
    service: str,
    now: datetime,
) -> SigV4Signature:
    """Sign a request. ``headers`` are the caller's headers (any case); the
    result carries every header to send, lowercase, with ``host``,
    ``x-amz-date``, ``x-amz-security-token`` (when a session token exists)
    and ``authorization`` added."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date = now.strftime("%Y%m%d")
    parts = urllib.parse.urlsplit(url)

    to_sign = {k.lower(): v for k, v in headers.items() if k.lower() != "authorization"}
    to_sign["host"] = parts.netloc
    to_sign["x-amz-date"] = amz_date
    to_sign.pop("x-amz-security-token", None)
    if credentials.session_token:
        to_sign["x-amz-security-token"] = credentials.session_token

    canonical, signed = canonicalize(method, url, to_sign, payload)
    scope = f"{date}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([_ALGORITHM, amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])

    key = ("AWS4" + credentials.secret_access_key).encode()
    for piece in (date, region, service, "aws4_request"):
        key = hmac.new(key, piece.encode(), hashlib.sha256).digest()
    signature = hmac.new(key, string_to_sign.encode(), hashlib.sha256).hexdigest()

    authorization = (
        f"{_ALGORITHM} Credential={credentials.access_key_id}/{scope}, "
        f"SignedHeaders={signed}, Signature={signature}"
    )
    out = {k: _trim(v) for k, v in to_sign.items()}
    out["authorization"] = authorization
    return SigV4Signature(canonical_request=canonical, string_to_sign=string_to_sign, authorization=authorization, headers=out)
