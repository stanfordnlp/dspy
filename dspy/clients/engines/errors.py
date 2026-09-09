"""Error translation shared by native, LiteLLM and legacy engines."""

import sys

from dspy.clients._litellm import is_litellm_context_window_error
from dspy.utils.exceptions import (
    ContextWindowExceededError,
    LMAuthError,
    LMBillingError,
    LMConfigurationError,
    LMError,
    LMInvalidRequestError,
    LMNotConfiguredError,
    LMProviderError,
    LMRateLimitError,
    LMServerError,
    LMTimeoutError,
    LMTransportError,
    LMUnexpectedError,
    LMUnsupportedFeatureError,
    LMUnsupportedModelError,
)


def wrap_error(exc: Exception, *, model: str, provider: str | None = None) -> LMError:
    """Preserve an existing DSPy error; translate backend failures otherwise."""
    if isinstance(exc, LMError):
        return exc

    from dspy import lm15

    if isinstance(exc, lm15.LM15Error):
        mappings = (
            (lm15.ContextLengthError, ContextWindowExceededError),
            (lm15.AuthError, LMAuthError),
            (lm15.BillingError, LMBillingError),
            (lm15.RateLimitError, LMRateLimitError),
            (lm15.UnsupportedModelError, LMUnsupportedModelError),
            (lm15.InvalidRequestError, LMInvalidRequestError),
            (lm15.TimeoutError, LMTimeoutError),
            (lm15.LockTimeoutError, LMTimeoutError),
            (lm15.StreamAssemblyError, LMUnexpectedError),
            (lm15.ServerError, LMServerError),
            (lm15.CapabilityError, LMUnsupportedFeatureError),
            (lm15.NotConfiguredError, LMNotConfiguredError),
            (lm15.ConfigurationError, LMConfigurationError),
            (lm15.TransportError, LMTransportError),
            (lm15.ProviderError, LMProviderError),
        )
        error_class = next((target for source, target in mappings if isinstance(exc, source)), LMUnexpectedError)
        wrapped = error_class(
            message=exc.message, model=model, provider=exc.provider or provider,
            provider_code=exc.provider_code, status=exc.status,
            request_id=exc.request_id, retry_after=exc.retry_after,
        )
        for name in ("partial", "part_index", "env_keys", "credential_hint", "path", "lock_path"):
            if hasattr(exc, name):
                setattr(wrapped, name, getattr(exc, name))
        return wrapped

    status = _exception_status(exc)
    provider = getattr(exc, "llm_provider", None) or provider
    model = getattr(exc, "model", None) or model
    message = _exception_message(exc)
    metadata = {
        "model": model,
        "provider": provider,
        "provider_code": _exception_provider_code(exc),
        "status": status,
        "request_id": _exception_request_id(exc),
        "retry_after": _exception_retry_after(exc),
    }

    if is_litellm_context_window_error(exc):
        return ContextWindowExceededError(message=message or "Context window exceeded", **metadata)

    exc_cls = _lm_error_class_from_litellm_exception(exc) or _lm_error_class_from_status(status)
    return exc_cls(message, **metadata)


def _safe_litellm_exception_class(name: str) -> type[Exception] | None:
    # Classifying a native or custom-plugin failure must not import LiteLLM.
    cls = getattr(sys.modules.get("litellm"), name, None)
    return cls if isinstance(cls, type) and issubclass(cls, Exception) else None


def _lm_error_class_from_litellm_exception(exc: Exception) -> type[LMError] | None:
    message = _exception_message(exc).lower()
    class_name = type(exc).__name__.lower()
    if _exception_status(exc) is None and any(
        phrase in message for phrase in ("api key", "apikey", "credentials", "environment variable")
    ):
        return LMNotConfiguredError
    if "timeout" in class_name or "timed out" in message or "timeout" in message:
        return LMTimeoutError
    if "connection" in class_name or "network" in message or "connection" in message:
        return LMTransportError

    mappings = [
        ("AuthenticationError", LMAuthError),
        ("RateLimitError", LMRateLimitError),
        ("NotFoundError", LMUnsupportedModelError),
        ("UnsupportedParamsError", LMUnsupportedFeatureError),
        ("UnprocessableEntityError", LMInvalidRequestError),
        ("ContentPolicyViolationError", LMInvalidRequestError),
        ("BadRequestError", LMInvalidRequestError),
        ("InvalidRequestError", LMInvalidRequestError),
        ("InternalServerError", LMServerError),
        ("ServiceUnavailableError", LMServerError),
        ("APIConnectionError", LMTransportError),
        ("APIResponseValidationError", LMProviderError),
        ("BudgetExceededError", LMBillingError),
        ("RouterRateLimitError", LMRateLimitError),
    ]
    for litellm_name, dspy_cls in mappings:
        litellm_cls = _safe_litellm_exception_class(litellm_name)
        if litellm_cls is not None and isinstance(exc, litellm_cls):
            return dspy_cls
    return None


def _lm_error_class_from_status(status: int | None) -> type[LMError]:
    if status in (401, 403):
        return LMAuthError
    if status == 402:
        return LMBillingError
    if status == 404:
        return LMUnsupportedModelError
    if status == 408:
        return LMTimeoutError
    if status == 429:
        return LMRateLimitError
    if status is not None and 400 <= status < 500:
        return LMInvalidRequestError
    if status is not None and status >= 500:
        return LMServerError
    return LMUnexpectedError if status is None else LMProviderError


# Best-effort LiteLLM/provider exception metadata extraction.
#
# LiteLLM exception metadata is not exposed as a single stable typed shape across providers, exception classes, and
# LiteLLM versions. Keep the defensive getattr-based extraction localized here so the rest of DSPy sees structured
# DSPyError metadata.
def _exception_status(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _exception_message(exc: Exception) -> str:
    message = getattr(exc, "message", None)
    if message is None:
        message = str(exc)
    return str(message)


def _exception_headers(exc: Exception):
    response = getattr(exc, "response", None)
    return getattr(response, "headers", None) or getattr(exc, "headers", None) or {}


def _exception_header(exc: Exception, name: str) -> str | None:
    headers = _exception_headers(exc)
    if not headers:
        return None
    try:
        return headers.get(name) or headers.get(name.lower())
    except AttributeError:
        return None


def _exception_request_id(exc: Exception) -> str | None:
    return (
        _exception_header(exc, "x-request-id")
        or _exception_header(exc, "request-id")
        or _exception_header(exc, "x-amzn-requestid")
        or _exception_header(exc, "x-ms-request-id")
    )


def _exception_retry_after(exc: Exception) -> float | None:
    retry_after = _exception_header(exc, "retry-after")
    try:
        return float(retry_after) if retry_after is not None else None
    except (TypeError, ValueError):
        return None


def _exception_provider_code(exc: Exception) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("code") is not None:
            return str(error["code"])
        if body.get("code") is not None:
            return str(body["code"])
    return None
