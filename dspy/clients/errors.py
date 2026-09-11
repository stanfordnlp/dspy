"""DSPy's application-facing projection of engine failures.

Engines speak dspy.lm15 errors. Only DSPy-owned boundaries translate them into
public DSPy errors; SDK-specific interpretation belongs to the owning engine.
"""

from contextlib import contextmanager

from dspy import lm15
from dspy.utils.exceptions import (
    ContextWindowExceededError,
    LMAuthError,
    LMBillingError,
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
    (lm15.ServerError, LMServerError),
    (lm15.CapabilityError, LMUnsupportedFeatureError),
    (lm15.NotConfiguredError, LMNotConfiguredError),
    (lm15.ConfigurationError, LMConfigurationError),
    (lm15.ToolDerivationError, LMConfigurationError),
    (lm15.TransportError, LMTransportError),
    (lm15.ProviderError, LMProviderError),
)


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
    wrapped = target(
        message=exc.message,
        model=getattr(exc, "model", None) or model,
        provider=exc.provider or provider,
        provider_code=exc.provider_code,
        status=exc.status,
        request_id=exc.request_id,
        retry_after=exc.retry_after,
    )
    for name in (
        "partial", "part_index", "env_keys", "credential_hint", "path", "lock_path",
        "providers", "candidates", "rules_tried", "catalog_searched", "cleanup_errors",
    ):
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
