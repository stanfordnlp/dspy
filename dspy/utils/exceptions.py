from __future__ import annotations

import traceback
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

    Catch this class for failures at DSPy's engine and capability boundaries.
    Concrete subclasses identify local configuration, transport, authentication,
    rate limits, invalid requests, unsupported features, and provider failures.
    Invalid Python API arguments still raise TypeError/ValueError; missing
    dependencies raise ImportError. Cancellation and warning policies propagate.
    """

    default_code = "lm_error"


class LMTransportError(LMError):
    """The LM request failed before the provider returned a response.

    This commonly represents network, DNS, TLS, connection-reset, or similar
    client-side transport failures.
    """

    default_code = "transport"


class LMLockTimeoutError(LMError):
    """Local credential-lock contention, not a provider or authentication failure."""

    default_code = "lock_timeout"

    def __init__(self, message: str = "", *, path: str = "", lock_path: str = "", **kwargs: Any):
        self.path = path
        self.lock_path = lock_path
        super().__init__(message, **kwargs)


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

    DSPy raises this for an unclassified engine failure and preserves its
    original exception as the cause. Only the owning engine interprets SDK
    errors: arbitrary message text or status-like attributes do not make a
    custom engine failure retryable. This is not a provider-response error or
    a model-output parsing error.
    """

    default_code = "unexpected"


class LMStreamAssemblyError(LMUnexpectedError):
    """An incomplete or invalid stream cannot be accepted as a successful response.

    ``partial`` carries salvageable content when available. It is not a success
    and must not be cached. Unknown usage remains unknown.
    """

    default_code = "stream_assembly"

    def __init__(self, message: str = "", *, partial=None, part_index: int | None = None, **kwargs: Any):
        self.partial = partial
        self.part_index = part_index
        super().__init__(message, **kwargs)


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

    Custom engines raise dspy.lm15.ContextLengthError; DSPy projects it to
    this public type. Legacy 3.4 LM subclasses may still raise this error
    directly. Modules such as ReAct catch it to shorten an overlong history;
    it does not trigger a generation retry or adapter-format fallback.

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


_RETRYABLE_LM_ERRORS = (LMRateLimitError, LMTimeoutError, LMServerError, LMTransportError, LMLockTimeoutError)


def is_retryable_lm_error(error: Exception) -> bool:
    """Return whether an LM error is generally safe to retry.

    DSPy owns retries; managed engines perform one attempt. This classification
    describes transient failures, not proof that the provider did no work.
    Never replay after visible stream output or a completed generation, and
    respect provider policy and valid `retry_after` hints. Network retries can
    still repeat a request already processed or billed by the provider.

    Args:
        error: The exception to classify.
    """
    return isinstance(error, _RETRYABLE_LM_ERRORS)


def format_error_for_lm(error: BaseException, *, traceback_frames: int = 0) -> str:
    """Format an exception as a string to be fed back to an LM.

    Modules that surface execution failures to the LM (`ReAct`, `ProgramOfThought`,
    `RLM`) share this formatter, keeping only their surrounding wrapper text
    (e.g. `"[Error] "` prefixes) at the call site.

    Args:
        error: The exception to format.
        traceback_frames: Maximum number of stack frames to include. When 0
            (the default), only `str(error)` is returned; when positive, a
            newline-prefixed traceback summary limited to that many frames is
            returned.
    """
    if traceback_frames <= 0:
        return str(error)
    return "\n" + "".join(
        traceback.format_exception(type(error), error, error.__traceback__, limit=traceback_frames)
    ).strip()


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
