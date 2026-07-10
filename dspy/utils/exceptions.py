from __future__ import annotations

from typing import Any

from dspy.signatures.signature import Signature


class DSPyError(Exception):
    """Base class for DSPy errors with structured metadata.

    Args:
        message: Human-readable error message.
        code: Stable DSPy error code. Defaults to the class code.
        model: Model identifier involved in the failure.
        provider: Provider or backend that returned the error.
        provider_code: Provider-specific error code, when available.
        status: HTTP status code, when the error came from an HTTP response.
        request_id: Provider request ID, when available.
        retry_after: Suggested retry delay in seconds, when available.
    """

    default_code: str | None = None

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        provider_code: str | None = None,
        status: int | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
    ):
        self.message = message
        self.code = code or self.default_code
        self.model = model
        self.provider = provider
        self.provider_code = provider_code
        self.status = status
        self.request_id = request_id
        self.retry_after = retry_after

        prefix = f"[{model}] " if model else ""
        super().__init__(f"{prefix}{message}" if message else prefix.rstrip())


class LMError(DSPyError):
    """Base class for language model errors.

    Catch this class to handle any failure raised while configuring or calling
    an LM. Concrete subclasses identify whether the failure came from local
    configuration, transport, provider authentication, rate limits, invalid
    requests, unsupported features, or provider server errors.
    """

    default_code = "lm_error"


class LMTransportError(LMError):
    """The LM request failed before the provider returned a response.

    This commonly represents network, DNS, TLS, connection-reset, or similar
    client-side transport failures.
    """

    default_code = "transport"


class LMConfigurationError(LMError):
    """The LM or provider client is not configured correctly."""

    default_code = "configuration"


class LMNotConfiguredError(LMConfigurationError):
    """The LM is missing required provider configuration or credentials."""

    default_code = "not_configured"


class LMUnsupportedFeatureError(LMError):
    """The LM, provider, or DSPy provider wrapper does not support a requested feature.

    Args:
        message: Human-readable error message.
        features: Feature names that were requested but unavailable, such as
            `"finetuning"`, `"reinforce"`, or `"structured_outputs"`.
        issues: Optional detailed reasons the requested feature could not be
            used.
        **kwargs: Structured error metadata accepted by `DSPyError`.
    """

    default_code = "unsupported_feature"

    def __init__(
        self,
        message: str = "",
        *,
        features: list[str] | None = None,
        issues: list[str] | None = None,
        **kwargs: Any,
    ):
        self.features = list(features or [])
        self.issues = list(issues or [])
        super().__init__(message, **kwargs)


class LMProviderError(LMError):
    """The provider returned an error response.

    Provider errors include structured metadata when available, such as HTTP
    `status`, provider `request_id`, provider-specific `provider_code`, and
    `retry_after` for rate limits.
    """

    default_code = "provider"


class LMUnexpectedError(LMError):
    """An unexpected failure occurred at the LM provider boundary.

    DSPy raises this when an exception is raised while calling the LM backend,
    but the exception does not match a known provider error class and does not
    include an HTTP status code. This keeps adapter fallback behavior from
    treating unknown LM-boundary failures as parse failures while avoiding
    over-classifying them as provider response errors.
    """

    default_code = "unexpected"


class LMAuthError(LMProviderError):
    """The provider rejected the request because authentication failed."""

    default_code = "auth"


class LMBillingError(LMProviderError):
    """The provider rejected the request because billing or quota failed."""

    default_code = "billing"


class LMRateLimitError(LMProviderError):
    """The provider rate-limited the request.

    Check the `retry_after` attribute for a provider-suggested retry delay when
    one is available.
    """

    default_code = "rate_limit"


class LMInvalidRequestError(LMProviderError):
    """The provider rejected the request shape or resource."""

    default_code = "invalid_request"


class ContextWindowExceededError(LMInvalidRequestError):
    """Raised when the prompt exceeds the model's context window.

    Any LM subclass should raise this error, or a subclass of it, when the
    request fails because the input is too long for the model. Adapters and
    some modules rely on catching this specific type to decide whether a
    fallback retry is appropriate.

    Args:
        model: The model identifier that rejected the request.
        message: Description of the error. Defaults to `"Context window exceeded"`.
        **kwargs: Structured error metadata such as `provider`, `status`, or
            `request_id`.
    """

    default_code = "context_window_exceeded"

    def __init__(
        self,
        *,
        model: str | None = None,
        message: str = "Context window exceeded",
        **kwargs: Any,
    ):
        super().__init__(message, model=model, **kwargs)


class LMUnsupportedModelError(LMInvalidRequestError):
    """The requested model is unavailable or unsupported by the provider."""

    default_code = "unsupported_model"


class LMTimeoutError(LMProviderError):
    """The provider request timed out."""

    default_code = "timeout"


class LMServerError(LMProviderError):
    """The provider failed while handling the request."""

    default_code = "server"


_RETRYABLE_LM_ERRORS = (LMRateLimitError, LMTimeoutError, LMServerError, LMTransportError)


def is_retryable_lm_error(error: Exception) -> bool:
    """Return whether an LM error is generally safe to retry.

    DSPy's built-in `LM` delegates provider retries to LiteLLM, but callers and
    higher-level orchestration code can use this helper to classify wrapped LM
    failures after provider retries are exhausted. Retryability is advisory:
    callers should still respect provider policy and `retry_after` when present.

    Args:
        error: The exception to classify.
    """
    return isinstance(error, _RETRYABLE_LM_ERRORS)


class AdapterParseError(DSPyError):
    """Raised when an adapter cannot parse an LM response into signature outputs.

    Args:
        adapter_name: Name of the adapter that failed to parse the response.
        signature: DSPy signature whose output fields were expected.
        lm_response: Raw LM response text or representation being parsed.
        message: Optional additional context about the parse failure.
        parsed_result: Partial parsed result, if any.
    """

    default_code = "adapter_parse_error"

    def __init__(
        self,
        adapter_name: str,
        signature: Signature,
        lm_response: str,
        message: str | None = None,
        parsed_result: dict[str, Any] | None = None,
    ):
        self.adapter_name = adapter_name
        self.signature = signature
        self.lm_response = lm_response
        self.parsed_result = parsed_result

        message = f"{message}\n\n" if message else ""
        message = (
            f"{message}"
            f"Adapter {adapter_name} failed to parse the LM response. \n\n"
            f"LM Response: {lm_response} \n\n"
            f"Expected to find output fields in the LM response: [{', '.join(signature.output_fields.keys())}] \n\n"
        )

        if parsed_result is not None:
            message += f"Actual output fields parsed from the LM response: [{', '.join(parsed_result.keys())}] \n\n"

        super().__init__(message)
