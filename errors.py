"""
lm15.errors — Canonical error taxonomy.

Every provider maps its idiosyncratic error shapes onto this hierarchy.
Error classes are the primary signal; canonical string codes exist for
serialization and wire formats.

Hierarchy:
    LM15Error
    ├── TransportError              (network/connection failures at the LM layer)
    ├── LockTimeoutError            (the credential-file lock could not be taken; local, transient)
    ├── StreamAssemblyError         (a stream cannot become a Response without inventing a fact; MAP-9)
    ├── ConfigurationError          (local SDK/configuration failures)
    │   ├── NotConfiguredError      (no API key or required provider config)
    │   ├── UnknownModelError       (the router: a model string that routes nowhere)
    │   └── AmbiguousModelError     (the router: a catalog match under more than one provider)
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
from typing import TYPE_CHECKING, Mapping, Sequence

from .rate_limits import diagnostics_text, freeze_rate_limits

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
        rate_limit_headers: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.rate_limit_headers = freeze_rate_limits(rate_limit_headers)
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


class LockTimeoutError(LM15Error):
    """The credential-file lock (spec/auth.md AUTH-4) could not be taken
    within the timeout: another lm15 process is refreshing the same
    credential (``lock_timeout``, 2026-09-08).

    Local and transient — a root-level class beside :class:`TransportError`,
    never a :class:`ProviderError` (no provider was asked) and never an
    :class:`AuthError` (nothing is wrong with the credential; AUTH-6).
    Retryable. Carries ``path`` (the guarded file) and ``lock_path``.
    :class:`lm15.auth.CredentialLockTimeout` is this class and the builtin
    ``TimeoutError`` at once, so ``except TimeoutError`` keeps working.
    """

    default_code = "lock_timeout"

    def __init__(
        self,
        message: str = "",
        *,
        path: str = "",
        lock_path: str = "",
        **kwargs,
    ) -> None:
        self.path = path
        self.lock_path = lock_path
        super().__init__(message, **kwargs)


# spec/auth.md AUTH-24 (ratified core 2026-09-22): the closed reasons a
# managed-auth lifecycle operation can fail with.  None is a provider HTTP
# 401; none is automatically retryable.
AUTH_OPERATION_REASONS: frozenset[str] = frozenset({
    "interaction_required", "method_unavailable", "connection_exists",
    "login_in_progress", "login_required", "connection_changed", "login_denied",
    "login_expired", "invalid_login_state", "attempt_unavailable", "indeterminate",
    "storage_unavailable", "unsupported_store_version", "selection_mismatch",
    "credential_rejected",
})
AUTH_OPERATION_STAGES: frozenset[str] = frozenset({
    "discovery", "reservation", "interaction", "authorization", "polling", "exchange",
    "persistence", "resolution", "renewal", "verification", "catalog", "dispatch",
})
AUTH_OPERATION_RECOVERIES: frozenset[str] = frozenset({
    "provide_input", "choose_method", "resume_attempt", "inspect_attempt", "restart_login",
    "select_connection", "repair_storage", "operator_action", "none",
})
AUTH_COMMIT_STATES: frozenset[str] = frozenset({"not_committed", "committed", "unknown"})


class AuthOperationError(LM15Error):
    """A managed-auth lifecycle operation failed locally (AUTH-24).

    Root-level, beside :class:`TransportError`: nothing here is a provider
    HTTP reply, and nothing here is safe to retry blindly.  ``reason`` is
    one of :data:`AUTH_OPERATION_REASONS`; ``stage`` names where the
    operation stopped; ``commit_state`` says whether the store changed
    (``committed``, ``not_committed``, or ``unknown`` after an interrupted
    write); ``recovery`` is guidance for the caller, never an instruction
    to retry.  ``provider`` is the route; ``connection_id`` and
    ``attempt_id`` are safe references, never secrets.

    The message is for people.  Programs match on ``reason``.
    """

    default_code = "auth_operation"

    def __init__(
        self,
        message: str = "",
        *,
        reason: str,
        stage: str = "resolution",
        commit_state: str = "not_committed",
        recovery: str = "none",
        operation: str | None = None,
        connection_id: str | None = None,
        attempt_id: str | None = None,
        method_id: str | None = None,
        **kwargs,
    ) -> None:
        if reason not in AUTH_OPERATION_REASONS:
            raise ValueError(f"AuthOperationError: unknown reason {reason!r}")
        if stage not in AUTH_OPERATION_STAGES:
            raise ValueError(f"AuthOperationError: unknown stage {stage!r}")
        if commit_state not in AUTH_COMMIT_STATES:
            raise ValueError(f"AuthOperationError: unknown commit_state {commit_state!r}")
        if recovery not in AUTH_OPERATION_RECOVERIES:
            raise ValueError(f"AuthOperationError: unknown recovery {recovery!r}")
        self.reason = reason
        self.stage = stage
        self.commit_state = commit_state
        self.recovery = recovery
        self.operation = operation
        self.connection_id = connection_id
        self.attempt_id = attempt_id
        self.method_id = method_id
        super().__init__(message, **kwargs)

    def __repr__(self) -> str:
        return (
            f"AuthOperationError(reason={self.reason!r}, stage={self.stage!r}, "
            f"commit_state={self.commit_state!r}, provider={self.provider!r})"
        )


class CollectionLimitError(LM15Error):
    """A local turn collector reached its budget, not a provider failure.

    Non-retryable. ``partial_events`` preserves the accepted events without
    copying their payloads. ``partial`` materializes them on demand as an
    incomplete Turn. A byte-limit failure also exposes ``rejected_event``:
    it was received but not yielded or added to the collection. Process it
    before resuming raw session reads if that content is needed.
    """

    default_code = "collection_limit"

    def __init__(
        self, message: str = "", *, limit: str | None = None,
        maximum: int | None = None, retained_bytes: int = 0,
        partial_events: tuple = (), rejected_event=None, **kwargs,
    ) -> None:
        self.limit = limit
        self.maximum = maximum
        self.retained_bytes = retained_bytes
        self.partial_events = tuple(partial_events)
        self.retained_events = len(self.partial_events)
        self.rejected_event = rejected_event
        super().__init__(message, **kwargs)

    @property
    def partial(self):
        # Lazy: allocating combined text/audio while handling a collection
        # limit would amplify memory pressure. Raw events are always available.
        from dataclasses import replace
        from .live import _materialize_turn

        return replace(_materialize_turn(self.partial_events), ended_by="incomplete")


class StreamAssemblyError(LM15Error):
    """A stream could not be assembled into a Response without inventing a fact.

    Three defects raise it (MAP-9 and MAP-3, contract change
    2026-09-11-stream-completion):

    - a tool call's fragments never carried a name (MAP-9): an unnamed call
      is not actionable (MAP-1), and guessing a name from the request
      dispatches the wrong function silently;
    - the stream ended without an end event: the finish reason and usage
      never arrived, and reporting the text as a finished turn would invent
      both;
    - an event arrived after the end event (MAP-3): it has no place in the
      Response, and dropping it would be silent loss.

    All three are adapter or source defects, not model behaviour — every
    shipped dialect names a call on its first fragment and ends exactly
    once — so the message points at the adapter.

    ``partial`` is everything that did assemble (text, thinking, other
    parts, usage, finish reason) with the offending material left out, so a
    caller that wants to salvage the turn can; ``part_index`` is the first
    offending part (MAP-9 only).

    What does NOT raise it: a failure after the end event that is not an
    event — the source raising while it drains, or its ``close()`` raising.
    The Response is complete; it is returned, and the failure is reported
    as a :class:`StreamCleanupWarning`.
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


class StreamCleanupWarning(RuntimeWarning):
    """A stream's source failed after the Response was already complete.

    Emitted (``warnings.warn``) when, after the end event has been yielded,
    the source raises while draining or its ``close()``/``aclose()`` raises.
    The provider finished the turn and billed it; the Response is returned
    unchanged.  The failure is about the connection's afterlife, not the
    answer, so it is never raised from ``response`` — a caller who wants it
    programmatically reads ``ResponseStream.cleanup_errors``, and a caller
    who wants it fatal runs with ``-W error::lm15.errors.StreamCleanupWarning``.
    """


class ConfigurationError(LM15Error):
    """Local SDK or provider-adapter configuration failure."""

    default_code = "not_configured"


class CapabilityError(LM15Error):
    """Requested capability is not supported by this provider adapter.

    ``feature`` (MAP-13, 2026-09-14) is the config path of what was
    refused — ``config.top_k``, ``config.reasoning.thinking_budget``,
    ``messages[0].parts[1]``, ``tools[2]`` — so the caller's own policy
    layer can drop it and retry without parsing the message.  Absent
    when the refusal is not about one addressable field.
    """

    default_code = "unsupported_feature"

    def __init__(self, message: str = "", *, feature: str | None = None, **kwargs) -> None:
        self.feature = feature
        super().__init__(message, **kwargs)


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
        base = self.message or self.code or ""
        head, sep, tail = base.partition("\n\n")
        suffix = f" ({context})" if context else ""
        details = diagnostics_text(self.rate_limit_headers, self.retry_after)
        return f"{head}{suffix}{details}" + (f"\n\n{tail}" if sep else "")


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
        # AUTH-1 provenance: where the rejected credential came from (a
        # label, never the value); set by ``with_credential_origin``.
        self.credential_origin: str | None = None

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
                guidance += f"    - Pass the key explicitly (api_key, or RouterConfig api_keys), or on a host with an environment set {keys}\n"
            else:
                guidance += "    - Pass the key explicitly (api_key, or RouterConfig api_keys)\n"
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
            "    - Check the reported limits and deployment capacity; a 429 does not prove the endpoint is unsupported\n"
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
                guidance += f"    - Pass the key explicitly (api_key, or RouterConfig api_keys), or on a host with an environment set {keys}\n"
            if provider_name:
                guidance += f"    - Configure credentials for {provider_name}\n"

        message = _append_guidance(message, guidance) if guidance else message
        super().__init__(message, **kwargs)


class UnknownModelError(ConfigurationError):
    """The router found no provider for a model string.

    No routable provider prefix, no catalog match, and no rule matched
    (spec/vocabularies.md ``unknown_model``, 2026-09-08).  Local and
    pre-network: no provider was asked — a provider's own "no such model"
    reply is :class:`UnsupportedModelError`.  Carries ``model`` (the string
    as requested); ``rules_tried`` and ``catalog_searched`` are the
    reference's diagnostics, not contract payload.
    """

    default_code = "unknown_model"

    def __init__(
        self,
        message: str = "",
        *,
        model: str = "",
        rules_tried: tuple = (),
        catalog_searched: bool = False,
        **kwargs,
    ) -> None:
        self.model = model
        self.rules_tried = tuple(rules_tried)
        self.catalog_searched = catalog_searched
        super().__init__(message, **kwargs)


class AmbiguousModelError(ConfigurationError):
    """The catalog matched a model string under more than one provider, or
    under more than one entry of one provider (``ambiguous_model``,
    2026-09-08).  The fix is an explicit ``provider:`` prefix.  Carries
    ``model`` and ``providers`` (every candidate, catalog order,
    deduplicated).
    """

    default_code = "ambiguous_model"

    def __init__(
        self,
        message: str = "",
        *,
        model: str = "",
        providers: tuple[str, ...] = (),
        **kwargs,
    ) -> None:
        self.model = model
        self.providers = tuple(providers)
        self.candidates = self.providers  # alias: full candidate list
        super().__init__(message, **kwargs)


_GUIDANCE_MARKER = "\n\n  To fix:"


_ORIGIN_MARKER = "\n\n  credential came from: "


def with_credential_origin(error: ProviderError, origin: str) -> ProviderError:
    """Name where an AuthError's credential came from (AUTH-1 provenance,
    amended 2026-09-19): its own line under the provider's message (a
    paragraph of its own, so ``__str__`` keeps the provider/HTTP suffix
    on the provider's line), before the guidance, so a log line at 3 a.m.
    answers "which identity?" without a second investigation.  Non-auth
    errors pass through."""
    if not isinstance(error, AuthError) or not origin:
        return error
    base = error.message.split(_GUIDANCE_MARKER, 1)[0]
    if _ORIGIN_MARKER in base:
        return error
    out = AuthError(
        base.rstrip() + _ORIGIN_MARKER + origin,
        provider=error.provider,
        env_keys=error.env_keys,
        credential_hint=error.credential_hint,
        provider_code=error.provider_code,
        status=error.status,
        request_id=error.request_id,
        retry_after=error.retry_after,
        rate_limit_headers=error.rate_limit_headers,
    )
    out.credential_origin = origin
    return out


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
        rate_limit_headers=error.rate_limit_headers,
    )


# ─── HTTP status → error class mapping ───────────────────────────────

# MAP-15: the pinned forms of a provider's "no such model" answer that carry no
# model-specific code and no not-found class (lm15-contract
# spec/model-not-found.json, carried verbatim; each form has a live receipt).
MODEL_NOT_FOUND_FORMS: tuple[dict[str, str], ...] = (
    {"code": "not_found_error", "prefix": "model: "},  # Anthropic, Claude Code
    {"code": "invalid_request_error", "contains": "The supported API model names are "},  # DeepSeek
    {"code": "1211"},  # Z.AI: Unknown Model
    {"code": "1214", "prefix": "modelCode: "},  # Z.AI: the model field is invalid
    {"code": "400", "suffix": " is not a valid model ID"},  # OpenRouter
    {"code": "invalid-argument", "prefix": "Model not found: "},  # xAI (2026-09-01)
    {"code": "validation_error", "contains": "The provided model identifier is invalid"},  # Bedrock Chat
    {"code": "invalid_request_error", "prefix": "Deployment ", "suffix": " doesn't exist or isn't accessible."},  # Parasail
)


def is_pinned_model_not_found(provider_code: str | None, message: str | None) -> bool:
    """True when the error is one of the pinned MAP-15 forms: the provider
    code matches exactly and the message passes every text test the form
    gives.  Never widened beyond the captured answers."""
    if not provider_code:
        return False
    text = message or ""
    for form in MODEL_NOT_FOUND_FORMS:
        if form["code"] != provider_code:
            continue
        if "prefix" in form and not text.startswith(form["prefix"]):
            continue
        if "contains" in form and form["contains"] not in text:
            continue
        if "suffix" in form and not text.endswith(form["suffix"]):
            continue
        return True
    return False


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
    UnknownModelError: "unknown_model",
    AmbiguousModelError: "ambiguous_model",
    TransportError: "transport",
    LockTimeoutError: "lock_timeout",
    AuthOperationError: "auth_operation",
    StreamAssemblyError: "stream_assembly",
    CollectionLimitError: "collection_limit",
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
RETRYABLE_ERRORS = (RateLimitError, TimeoutError, ServerError, TransportError, LockTimeoutError)


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
