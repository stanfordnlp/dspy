"""LiteLLM/SDK exception interpretation, confined to the compatibility engine."""

import sys
from contextlib import contextmanager

from dspy import lm15
from dspy.clients._http import finite_seconds, retry_after_seconds


def _header(headers, name):
    # SDKs use both case-insensitive header objects and plain mappings.
    items = headers.items() if hasattr(headers, "items") else headers or ()
    return next((value for key, value in items if str(key).lower() == name), None)


def _structured_class(provider, code):
    # Refine generic SDK errors with documented provider codes, never prose.
    # In particular an OpenAI quota failure also travels as HTTP 429.
    if provider in {"openai", "openai-chat", "azure", "azure-chat"}:
        return {
            "insufficient_quota": lm15.BillingError,
            "context_length_exceeded": lm15.ContextLengthError,
            "model_not_found": lm15.UnsupportedModelError,
            "DeploymentNotFound": lm15.UnsupportedModelError,
            "invalid_api_key": lm15.AuthError,
        }.get(code)
    if provider == "anthropic":
        return {
            "billing_error": lm15.BillingError,
            "authentication_error": lm15.AuthError,
            "permission_error": lm15.AuthError,
        }.get(code)
    return None


def to_lm15_error(exc: Exception, *, model=None, provider=None) -> Exception:
    """Specific SDK classes and provider codes win over status; never use prose.

    Unknown exceptions without an HTTP status stay unknown. The DSPy engine
    boundary handles those as unexpected failures, preserving the original cause.
    This function is only for errors raised by the compatibility backend, not
    arbitrary custom engines. It never imports LiteLLM to classify an error.
    """
    if isinstance(exc, (lm15.LM15Error, Warning, ImportError)):
        return exc
    module = sys.modules.get("litellm")
    mappings = (
        ("ContextWindowExceededError", lm15.ContextLengthError),
        ("BudgetExceededError", lm15.BillingError),
        ("RouterRateLimitError", lm15.RateLimitError),
        ("UnsupportedParamsError", lm15.UnsupportedFeatureError),
        ("AuthenticationError", lm15.AuthError),
        ("PermissionDeniedError", lm15.AuthError),
        ("Timeout", lm15.TimeoutError),
        ("APIConnectionError", lm15.TransportError),
        ("APIResponseValidationError", lm15.ProviderError),
        ("RateLimitError", lm15.RateLimitError),
        ("NotFoundError", lm15.UnsupportedModelError),
        ("UnprocessableEntityError", lm15.InvalidRequestError),
        ("ContentPolicyViolationError", lm15.InvalidRequestError),
        ("BadRequestError", lm15.InvalidRequestError),
        ("InvalidRequestError", lm15.InvalidRequestError),
        ("InternalServerError", lm15.ServerError),
        ("ServiceUnavailableError", lm15.ServerError),
        ("BadGatewayError", lm15.ServerError),
    )
    target = None
    for name, canonical in mappings:
        cls = getattr(module, name, None)
        if isinstance(cls, type) and isinstance(exc, cls):
            target = canonical
            break
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(response, "status_code", None)
    if isinstance(status, str) and status.isdecimal():
        status = int(status)
    if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
        status = None
    if target is None and status is None:
        return exc
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None) or {}
    body = getattr(exc, "body", None)
    detail = body.get("error", body) if isinstance(body, dict) else {}
    code = detail.get("code") if isinstance(detail, dict) else None
    if code is None and isinstance(detail, dict):
        code = detail.get("type")
    code = str(code) if code is not None else None
    provider = getattr(exc, "llm_provider", None) or provider
    if target in (None, lm15.ProviderError, lm15.InvalidRequestError, lm15.RateLimitError, lm15.ServerError):
        target = _structured_class(provider, code) or target
    message = str(getattr(exc, "message", None) or exc)
    metadata = {
        "provider": provider,
        "provider_code": code,
        "request_id": getattr(exc, "request_id", None) or next((
            value for name in ("x-request-id", "request-id", "x-amzn-requestid", "x-amz-request-id", "x-ms-request-id")
            if (value := _header(headers, name)) is not None
        ), None),
        "retry_after": finite_seconds(getattr(exc, "retry_after", None)),
    }
    if metadata["retry_after"] is None:
        metadata["retry_after"] = retry_after_seconds(_header(headers, "retry-after"))
    if target is None:
        from dspy._vendor.lm15.errors import map_http_error

        wrapped = map_http_error(status, message, **metadata)
    else:
        wrapped = target(message, status=status, **metadata)
    # lm15's generic errors have no model constructor argument. Keep the SDK's
    # model label for the existing DSPy public metadata projection.
    wrapped.model = getattr(exc, "model", None) or model
    return wrapped


@contextmanager
def litellm_errors(*, model=None, provider=None):
    try:
        yield
    except Exception as exc:
        mapped = to_lm15_error(exc, model=model, provider=provider)
        if mapped is exc:
            raise
        raise mapped from exc
