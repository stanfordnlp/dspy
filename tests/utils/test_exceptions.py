import dspy
from dspy.utils.exceptions import AdapterParseError, ContextWindowExceededError


def test_context_window_exceeded_error_defaults():
    error = ContextWindowExceededError()
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
