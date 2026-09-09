"""
lm15.errors — Canonical error taxonomy.

Every provider maps its idiosyncratic error shapes onto this hierarchy.
Error classes are the primary signal; canonical string codes exist for
serialization and wire formats.

Hierarchy:
    LM15Error
    ├── TransportError              (network/connection failures at the LM layer)
    ├── StreamAssemblyError         (a stream cannot become a Response without inventing a fact; MAP-9)
    ├── ConfigurationError          (local SDK/configuration failures)
    │   └── NotConfiguredError      (no API key or required provider config)
    ├── CapabilityError             (local provider-adapter capability failures)
    │   └── UnsupportedFeatureError
    └── ProviderError               (provider returned an error response)
        ├── AuthError               (401/403 — bad or missing API key)
        ├── BillingError            (402 — payment/quota issue)
        ├── RateLimitError          (429 — too many requests)
        ├── InvalidRequestError     (4xx request-shape/resource errors)
        │   ├── ContextLengthError  (input too long for model)
        │   └── UnsupportedModelError
        ├── TimeoutError            (408/504 — request timed out)
        └── ServerError             (5xx — provider-side failure)
"""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .types import Response


class LM15Error(Exception):
    """Base for all lm15 errors.

    Errors keep the human-readable exception message in ``str(error)`` while
    also exposing structured metadata for logging, telemetry, retries, and
    programmatic handling.
    """

    default_code: str | None = None

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        provider: str | None = None,
        provider_code: str | None = None,
        status: int | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.message = message
        self.code = code or self.default_code
        self.provider = provider
        self.provider_code = provider_code
        self.status = status
        self.request_id = request_id
        # Number rule (docs/serde-rules.md): retry_after is float-typed;
        # same-valued int input (e.g. a Retry-After header of "30") coerces.
        if type(retry_after) is int:
            retry_after = float(retry_after)
        self.retry_after = retry_after
        super().__init__(message)


class TransportError(LM15Error):
    """High-level LM transport failure.

    Provider LMs wrap lower-level ``lm15.transports.TransportError`` exceptions
    into this class.
    """

    default_code = "transport"


class StreamAssemblyError(LM15Error):
    """A stream could not be assembled into a Response without inventing a fact.

    Raised by the accumulator (MAP-9) when a tool call's fragments never
    carried a name: an unnamed call is not actionable (MAP-1), and guessing
    a name from the request dispatches the wrong function silently.  This is
    an adapter defect, not model behaviour — every shipped dialect names a
    call on its first fragment — so the message points at the adapter.

    ``partial`` is everything that did assemble (text, thinking, other
    parts, usage, finish reason) with the unnamed call(s) left out, so a
    caller that wants to salvage the turn can; ``part_index`` is the first
    offending part.
    """

    default_code = "stream_assembly"

    def __init__(
        self,
        message: str = "",
        *,
        partial: "Response | None" = None,
        part_index: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(message, **kwargs)
        self.partial = partial
        self.part_index = part_index


class ConfigurationError(LM15Error):
    """Local SDK or provider-adapter configuration failure."""

    default_code = "not_configured"


class CapabilityError(LM15Error):
    """Requested capability is not supported by this provider adapter."""

    default_code = "unsupported_feature"


class ProviderError(LM15Error):
    """The provider returned an error response."""

    default_code = "provider"

    def __str__(self) -> str:
        # The displayed form carries provider / HTTP status / request id so
        # a bare relay like "model: x" still says who said it.  The
        # ``message`` FIELD stays untouched (it is what the contract pins).
        context = ", ".join(
            item
            for item in (
                self.provider,
                f"HTTP {self.status}" if self.status is not None else None,
                f"request {self.request_id}" if self.request_id else None,
            )
            if item
        )
        base = self.message or self.code
        if not context:
            return base
        head, sep, tail = base.partition("\n\n")
        suffix = f" ({context})"
        return f"{head}{suffix}\n\n{tail}" if sep else f"{base}{suffix}"


class AuthError(ProviderError):
    """Authentication failed — invalid, expired, or missing API key."""

    default_code = "auth"

    def __init__(
        self,
        message: str = "",
        *,
        provider: str | None = None,
        env_keys: tuple[str, ...] = (),
        credential_hint: str | None = None,
        **kwargs,
    ) -> None:
        if provider is not None:
            kwargs.setdefault("provider", provider)
        provider_name = provider or kwargs.get("provider")
        self.env_keys = tuple(env_keys)
        self.credential_hint = credential_hint

        if credential_hint:
            # Subscription/OAuth adapters: guidance is how to re-login, not
            # which env var to set (there is none).
            guidance = f"\n\n  To fix:\n    - {credential_hint}\n"
        else:
            guidance = (
                "\n\n"
                "  To fix:\n"
                "    - Check that your API key is correct and not expired\n"
            )
            if self.env_keys:
                keys = " or ".join(f"{key}=..." for key in self.env_keys)
                guidance += f"    - Set the provider API key in your environment: {keys}\n"
            else:
                guidance += "    - Set the provider API key in your environment\n"
            if provider_name:
                guidance += f"    - Verify your {provider_name} account/project has access\n"

        super().__init__(_append_guidance(message, guidance), **kwargs)


class RateLimitError(ProviderError):
    """Rate limited by the provider (HTTP 429)."""

    default_code = "rate_limit"

    def __init__(self, message: str = "", **kwargs) -> None:
        guidance = (
            "\n\n"
            "  To fix:\n"
            "    - Wait a moment and retry\n"
            "    - Retry with backoff in your application layer (lm15 never retries for you)\n"
            "    - Reduce request rate or upgrade your API plan\n"
        )
        super().__init__(_append_guidance(message, guidance), **kwargs)


class BillingError(ProviderError):
    """402 — billing or payment issue."""

    default_code = "billing"


class TimeoutError(ProviderError, builtins.TimeoutError):
    """Provider request timed out.

    Also subclasses the builtin ``TimeoutError`` so a user's bare
    ``except TimeoutError:`` catches lm15 timeouts. ``ProviderError`` comes
    first in the MRO, so lm15 metadata (``code``, ``status``, ...) wins.
    """

    default_code = "timeout"


# Descriptive alias that avoids shadowing Python's built-in TimeoutError in new code.
RequestTimeoutError = TimeoutError


class InvalidRequestError(ProviderError):
    """Bad request shape or invalid provider resource (4xx)."""

    default_code = "invalid_request"


class ContextLengthError(InvalidRequestError):
    """The input exceeds the model's context window."""

    default_code = "context_length"

    def __init__(self, message: str = "", **kwargs) -> None:
        guidance = (
            "\n\n"
            "  To fix:\n"
            "    - Reduce the prompt or system prompt length\n"
            "    - Clear conversation history\n"
            "    - Use a model with a larger context window\n"
            "    - Lower max_tokens to leave more room for input\n"
        )
        super().__init__(_append_guidance(message, guidance), **kwargs)


class UnsupportedModelError(InvalidRequestError):
    """Model not found, unavailable, or unsupported by the provider."""

    default_code = "unsupported_model"


class ServerError(ProviderError):
    """Provider-side failure (5xx)."""

    default_code = "server"


class UnsupportedFeatureError(CapabilityError):
    """Feature not supported by this provider adapter."""

    default_code = "unsupported_feature"


class NotConfiguredError(ConfigurationError):
    """No API key or required provider configuration was found."""

    default_code = "not_configured"

    def __init__(
        self,
        message: str = "",
        *,
        provider: str | None = None,
        env_keys: tuple[str, ...] = (),
        credential_hint: str | None = None,
        **kwargs,
    ) -> None:
        if provider is not None:
            kwargs.setdefault("provider", provider)
        provider_name = provider or kwargs.get("provider")
        self.env_keys = tuple(env_keys)
        self.credential_hint = credential_hint

        guidance = ""
        if credential_hint:
            guidance = f"\n\n  To fix:\n    - {credential_hint}\n"
        elif self.env_keys or provider_name:
            guidance = "\n\n  To fix:\n"
            if self.env_keys:
                keys = " or ".join(f"{key}=..." for key in self.env_keys)
                guidance += f"    - Set the provider API key in your environment: {keys}\n"
            if provider_name:
                guidance += f"    - Configure credentials for {provider_name}\n"

        message = _append_guidance(message, guidance) if guidance else message
        super().__init__(message, **kwargs)


_GUIDANCE_MARKER = "\n\n  To fix:"


def with_credential_hint(error: ProviderError, hint: str) -> ProviderError:
    """Rewrite an AuthError's guidance for subscription (OAuth) adapters.

    API-key adapters point at env vars; subscription adapters have no env
    var — the fix is re-running the provider CLI login. Non-auth errors pass
    through unchanged.
    """
    if not isinstance(error, AuthError):
        return error
    base = error.message.split(_GUIDANCE_MARKER, 1)[0]
    return AuthError(
        base,
        provider=error.provider,
        credential_hint=hint,
        provider_code=error.provider_code,
        status=error.status,
        request_id=error.request_id,
        retry_after=error.retry_after,
    )


# ─── HTTP status → error class mapping ───────────────────────────────

def map_http_error(
    status: int,
    message: str,
    *,
    provider: str | None = None,
    env_keys: tuple[str, ...] = (),
    provider_code: str | None = None,
    request_id: str | None = None,
    retry_after: float | None = None,
) -> ProviderError:
    """Map HTTP status + message to a typed ProviderError.

    Provider LMs extract the human-readable message and provider-specific code
    from the provider's error body in their ``normalize_error`` override. This
    function only maps HTTP status codes.
    """
    kwargs = _metadata_kwargs(
        provider=provider,
        provider_code=provider_code,
        status=status,
        request_id=request_id,
        retry_after=retry_after,
    )
    if status in (401, 403):
        return AuthError(message, env_keys=env_keys, **kwargs)
    if status == 402:
        return BillingError(message, **kwargs)
    if status in (408, 504):
        return TimeoutError(message, **kwargs)
    if status == 429:
        return RateLimitError(message, **kwargs)
    if status in (400, 404, 409, 413, 422):
        return InvalidRequestError(message, **kwargs)
    if 500 <= status <= 599:
        return ServerError(message, **kwargs)
    return ProviderError(message, **kwargs)


# ─── Canonical error codes ───────────────────────────────────────────

# Bidirectional mapping between error classes and string codes.
# Codes are provider-agnostic and stable across LMs. More-specific classes must
# appear before their base classes.

_CLASS_TO_CODE: dict[type[LM15Error], str] = {
    ContextLengthError: "context_length",
    UnsupportedModelError: "unsupported_model",
    AuthError: "auth",
    BillingError: "billing",
    RateLimitError: "rate_limit",
    InvalidRequestError: "invalid_request",
    TimeoutError: "timeout",
    ServerError: "server",
    UnsupportedFeatureError: "unsupported_feature",
    NotConfiguredError: "not_configured",
    TransportError: "transport",
    ProviderError: "provider",
}

_CODE_TO_CLASS: dict[str, type[LM15Error]] = {v: k for k, v in _CLASS_TO_CODE.items()}


def canonical_error_code(error: type[LM15Error] | LM15Error) -> str:
    """Return the canonical string code for an error class or instance."""
    cls = error if isinstance(error, type) else type(error)
    for check_cls, code in _CLASS_TO_CODE.items():
        if issubclass(cls, check_cls):
            return code
    default_code = getattr(cls, "default_code", None)
    return default_code or "provider"


def error_class_for_code(code: str) -> type[LM15Error]:
    """Return the LM15Error subclass for a canonical string code."""
    return _CODE_TO_CLASS.get(code, ProviderError)


# Errors that are safe to retry; lm15 never retries itself — this
# classification is data for the caller's own retry policy.
RETRYABLE_ERRORS = (RateLimitError, TimeoutError, ServerError, TransportError)


def _append_guidance(message: str, guidance: str) -> str:
    """Append guidance once while preserving the original provider message."""
    if guidance.strip() in message:
        return message
    return message.rstrip() + guidance


def _metadata_kwargs(
    *,
    provider: str | None = None,
    provider_code: str | None = None,
    status: int | None = None,
    request_id: str | None = None,
    retry_after: float | None = None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if provider:
        kwargs["provider"] = provider
    if provider_code:
        kwargs["provider_code"] = provider_code
    if status is not None:
        kwargs["status"] = status
    if request_id:
        kwargs["request_id"] = request_id
    if retry_after is not None:
        kwargs["retry_after"] = retry_after
    return kwargs
