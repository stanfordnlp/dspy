"""DSPy's application-facing projection of engine failures.

Engines speak dspy.lm15 errors. Only DSPy-owned boundaries translate them into
public DSPy errors; SDK-specific interpretation belongs to the owning engine.
"""

import re
from contextlib import contextmanager

from dspy import lm15
from dspy.utils.exceptions import (
    ContextWindowExceededError,
    LMAuthError,
    LMBillingError,
    LMCollectionLimitError,
    LMConfigurationError,
    LMError,
    LMInvalidRequestError,
    LMLockTimeoutError,
    LMNotConfiguredError,
    LMProviderError,
    LMRateLimitError,
    LMServerError,
    LMStreamAssemblyError,
    LMTimeoutError,
    LMTransportError,
    LMUnexpectedError,
    LMUnsupportedFeatureError,
    LMUnsupportedModelError,
)

# Specific classes precede their parents. Unknown future subclasses inherit the
# nearest known meaning; an unknown root error fails closed without retrying.
ERROR_MAPPING = (
    (lm15.ContextLengthError, ContextWindowExceededError),
    (lm15.AuthError, LMAuthError),
    (lm15.BillingError, LMBillingError),
    (lm15.RateLimitError, LMRateLimitError),
    (lm15.UnsupportedModelError, LMUnsupportedModelError),
    (lm15.InvalidRequestError, LMInvalidRequestError),
    (lm15.TimeoutError, LMTimeoutError),
    (lm15.LockTimeoutError, LMLockTimeoutError),
    (lm15.StreamAssemblyError, LMStreamAssemblyError),
    (lm15.CollectionLimitError, LMCollectionLimitError),
    (lm15.ServerError, LMServerError),
    (lm15.CapabilityError, LMUnsupportedFeatureError),
    (lm15.NotConfiguredError, LMNotConfiguredError),
    (lm15.ConfigurationError, LMConfigurationError),
    (lm15.ToolDerivationError, LMConfigurationError),
    (lm15.TransportError, LMTransportError),
    (lm15.ProviderError, LMProviderError),
)


_ROUTER_CONFIG_KEYS = re.compile(r"pass RouterConfig\(api_keys=\{'[^']+': \"\.\.\.\"\}\)")


def _dspy_remedy(message: str) -> str:
    """Say the remedy in DSPy's terms: a DSPy user cannot pass a RouterConfig."""
    message = _ROUTER_CONFIG_KEYS.sub('pass api_key="..." to dspy.LM(...)', message)
    return message.replace("(api_key, or RouterConfig api_keys)", "(api_key= on dspy.LM)")


def wrap_error(exc: Exception, *, model: str, provider: str | None = None) -> Exception:
    """Project a canonical failure without guessing from arbitrary attributes/text.

    Preserve public DSPy errors by identity (including legacy 3.4 plugins).
    Warnings promoted to errors and missing dependencies remain Python errors.
    The caller raises the returned exception from the original, retaining its
    exact canonical code, diagnostics, and any original SDK cause.
    """
    if isinstance(exc, (LMError, Warning, ImportError)):
        return exc
    if not isinstance(exc, lm15.LM15Error):
        return LMUnexpectedError(str(exc), model=model, provider=provider)
    target = next((target for source, target in ERROR_MAPPING if isinstance(exc, source)), LMUnexpectedError)
    # The message is preserved verbatim (the contract pins it) except for the
    # one local case whose remedy names a RouterConfig: a missing key.
    message = _dspy_remedy(exc.message) if isinstance(exc, lm15.NotConfiguredError) else exc.message
    details = {}
    if isinstance(exc, lm15.CapabilityError):
        details["feature"] = exc.feature
    if isinstance(exc, lm15.CollectionLimitError):
        details["source"] = exc
    wrapped = target(
        message=message,
        model=getattr(exc, "model", None) or model,
        provider=exc.provider or provider,
        provider_code=exc.provider_code,
        status=exc.status,
        request_id=exc.request_id,
        retry_after=exc.retry_after,
        **details,
    )
    for name in (
        "partial", "part_index", "env_keys", "credential_hint", "path", "lock_path",
        "providers", "candidates", "rules_tried", "catalog_searched", "cleanup_errors",
    ):
        if name == "partial" and isinstance(exc, lm15.CollectionLimitError):
            continue  # Even hasattr() would evaluate its lazy partial property.
        if hasattr(exc, name):
            setattr(wrapped, name, getattr(exc, name))
    return wrapped


@contextmanager
def error_boundary(model: str, *, provider: str | None = None, unexpected: bool = False):
    """Translate once at a DSPy boundary, not around arbitrary application code.

    Setup preserves local TypeError/ValueError. The actual engine invocation
    opts into unexpected failures so an engine bug cannot become a bad model
    prediction or a parsing fallback. Cancellation is never caught here.
    """
    try:
        yield
    except Exception as exc:
        if not unexpected and not isinstance(exc, lm15.LM15Error):
            raise
        error = wrap_error(exc, model=model, provider=provider)
        if error is exc:
            raise
        raise error from exc
