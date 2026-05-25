import dspy
from dspy.utils.exceptions import (
    AdapterParseError,
    ContextWindowExceededError,
    DSPyError,
    LMError,
    LMInvalidRequestError,
)


def test_lm_errors_are_exported_from_dspy():
    assert dspy.DSPyError is not None
    assert dspy.LMError is LMError
    assert dspy.LMUnexpectedError is not None
    assert dspy.AdapterParseError is AdapterParseError
    assert dspy.RETRYABLE_LM_ERRORS == (
        dspy.LMRateLimitError,
        dspy.LMTimeoutError,
        dspy.LMServerError,
        dspy.LMTransportError,
    )


def test_retryable_lm_errors_classification():
    assert isinstance(dspy.LMRateLimitError(), dspy.RETRYABLE_LM_ERRORS)
    assert isinstance(dspy.LMTimeoutError(), dspy.RETRYABLE_LM_ERRORS)
    assert isinstance(dspy.LMServerError(), dspy.RETRYABLE_LM_ERRORS)
    assert isinstance(dspy.LMTransportError(), dspy.RETRYABLE_LM_ERRORS)
    assert not isinstance(dspy.LMAuthError(), dspy.RETRYABLE_LM_ERRORS)
    assert not isinstance(dspy.LMInvalidRequestError(), dspy.RETRYABLE_LM_ERRORS)
    assert not isinstance(dspy.LMUnexpectedError(), dspy.RETRYABLE_LM_ERRORS)


def test_lm_error_metadata():
    error = dspy.LMRateLimitError(
        "rate limited",
        model="openai/gpt-4o",
        provider="openai",
        status=429,
        request_id="req-123",
        retry_after=2.5,
    )

    assert error.code == "rate_limit"
    assert error.model == "openai/gpt-4o"
    assert error.provider == "openai"
    assert error.status == 429
    assert error.request_id == "req-123"
    assert error.retry_after == 2.5
    assert str(error) == "[openai/gpt-4o] rate limited"


def test_context_window_exceeded_error_defaults():
    error = ContextWindowExceededError()
    assert isinstance(error, LMInvalidRequestError)
    assert isinstance(error, LMError)
    assert error.code == "context_window_exceeded"
    assert error.model is None
    assert str(error) == "Context window exceeded"


def test_context_window_exceeded_error_with_model():
    error = ContextWindowExceededError(model="openai/gpt-4o")
    assert error.model == "openai/gpt-4o"
    assert str(error) == "[openai/gpt-4o] Context window exceeded"


def test_context_window_exceeded_error_with_message():
    error = ContextWindowExceededError(model="openai/gpt-4o", message="Input is 200k tokens, limit is 128k")
    assert error.model == "openai/gpt-4o"
    assert str(error) == "[openai/gpt-4o] Input is 200k tokens, limit is 128k"


def test_context_window_exceeded_error_message_without_model():
    error = ContextWindowExceededError(message="Too many tokens")
    assert error.model is None
    assert str(error) == "Too many tokens"


def test_adapter_parse_error_basic():
    adapter_name = "ChatAdapter"
    signature = dspy.make_signature("question->answer1, answer2")
    lm_response = "[[ ## answer1 ## ]]\nanswer1"

    error = AdapterParseError(adapter_name=adapter_name, signature=signature, lm_response=lm_response)

    assert isinstance(error, DSPyError)
    assert error.adapter_name == adapter_name
    assert error.signature == signature
    assert error.lm_response == lm_response

    error_message = str(error)
    assert error_message == (
        "Adapter ChatAdapter failed to parse the LM response. \n\n"
        "LM Response: [[ ## answer1 ## ]]\nanswer1 \n\n"
        "Expected to find output fields in the LM response: [answer1, answer2] \n\n"
    )


def test_adapter_parse_error_with_message():
    adapter_name = "ChatAdapter"
    signature = dspy.make_signature("question->answer1, answer2")
    lm_response = "[[ ## answer1 ## ]]\nanswer1"
    message = "Critical error, please fix!"

    error = AdapterParseError(adapter_name=adapter_name, signature=signature, lm_response=lm_response, message=message)

    assert error.adapter_name == adapter_name
    assert error.signature == signature
    assert error.lm_response == lm_response

    error_message = str(error)
    assert error_message == (
        "Critical error, please fix!\n\n"
        "Adapter ChatAdapter failed to parse the LM response. \n\n"
        "LM Response: [[ ## answer1 ## ]]\nanswer1 \n\n"
        "Expected to find output fields in the LM response: [answer1, answer2] \n\n"
    )


def test_adapter_parse_error_with_parsed_result():
    adapter_name = "ChatAdapter"
    signature = dspy.make_signature("question->answer1, answer2")
    lm_response = "[[ ## answer1 ## ]]\nanswer1"
    parsed_result = {"answer1": "value1"}

    error = AdapterParseError(
        adapter_name=adapter_name, signature=signature, lm_response=lm_response, parsed_result=parsed_result
    )

    error_message = str(error)
    assert error_message == (
        "Adapter ChatAdapter failed to parse the LM response. \n\n"
        "LM Response: [[ ## answer1 ## ]]\nanswer1 \n\n"
        "Expected to find output fields in the LM response: [answer1, answer2] \n\n"
        "Actual output fields parsed from the LM response: [answer1] \n\n"
    )
