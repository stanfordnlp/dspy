from __future__ import annotations

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
    """Base class for language model errors."""

    default_code = "lm_error"


class LMTransportError(LMError):
    """The LM request failed before the provider returned a response."""

    default_code = "transport"


class LMConfigurationError(LMError):
    """The LM or provider client is not configured correctly."""

    default_code = "configuration"


class LMNotConfiguredError(LMConfigurationError):
    """The LM is missing required provider configuration or credentials."""

    default_code = "not_configured"


class LMUnsupportedFeatureError(LMError):
    """The LM does not support a requested feature."""

    default_code = "unsupported_feature"

    def __init__(
        self,
        message: str = "",
        *,
        features: list[str] | None = None,
        issues: list[str] | None = None,
        **kwargs,
    ):
        self.features = list(features or [])
        self.issues = list(issues or [])
        super().__init__(message, **kwargs)


class LMProviderError(LMError):
    """The provider returned an error response."""

    default_code = "provider"


class LMAuthError(LMProviderError):
    """The provider rejected the request because authentication failed."""

    default_code = "auth"


class LMBillingError(LMProviderError):
    """The provider rejected the request because billing or quota failed."""

    default_code = "billing"


class LMRateLimitError(LMProviderError):
    """The provider rate-limited the request."""

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
        **kwargs,
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


RETRYABLE_LM_ERRORS = (LMRateLimitError, LMTimeoutError, LMServerError, LMTransportError)


class AdapterParseError(Exception):
    """Exception raised when adapter cannot parse the LM response."""

    def __init__(
        self,
        adapter_name: str,
        signature: Signature,
        lm_response: str,
        message: str | None = None,
        parsed_result: str | None = None,
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
