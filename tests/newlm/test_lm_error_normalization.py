import pytest

import dspy
from dspy.utils.exceptions import LMError


class NativeContextError(Exception):
    pass


class NormalizingLM(dspy.BaseLM):
    def __init__(self):
        super().__init__(model="test/context", cache=False)

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        raise NativeContextError("too long")

    def normalize_error(self, error: Exception, request: dspy.LMRequest) -> Exception:
        if isinstance(error, NativeContextError):
            return dspy.ContextWindowExceededError(
                model=request.model,
                provider="test",
                message="The request is too long.",
            )
        return error


class UnnormalizedLM(dspy.BaseLM):
    def __init__(self):
        super().__init__(model="test/raw", cache=False)

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        raise NativeContextError("too long")


def test_context_window_error_is_a_normalized_lm_error():
    error = dspy.ContextWindowExceededError(
        model="test/model",
        provider="test-provider",
        status=400,
        request_id="req_123",
    )

    assert isinstance(error, LMError)
    assert error.code == "context_window_exceeded"
    assert error.model == "test/model"
    assert error.provider == "test-provider"
    assert error.status == 400
    assert error.request_id == "req_123"
    assert str(error) == "[test/model] Context window exceeded"


def test_language_model_normalizes_errors():
    lm = NormalizingLM()

    with pytest.raises(dspy.ContextWindowExceededError) as exc_info:
        lm("hello")

    assert exc_info.value.model == "test/context"
    assert exc_info.value.provider == "test"


def test_language_model_leaves_unknown_errors_unchanged():
    lm = UnnormalizedLM()

    with pytest.raises(NativeContextError):
        lm("hello")
