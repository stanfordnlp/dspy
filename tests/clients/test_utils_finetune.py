import pytest

from dspy.clients.utils_finetune import (
    TrainDataFormat,
    find_data_error_chat,
    find_data_error_chat_message,
    find_data_errors_completion,
    validate_data_format,
)


def test_validate_data_format_rejects_unsupported_format_under_optimized_python():
    with pytest.raises(ValueError, match="Data format grpo_chat is not supported"):
        validate_data_format([], TrainDataFormat.GRPO_CHAT)


def test_finetune_validation_helpers_return_errors_for_non_dict_inputs():
    expected = "Not a dictionary -- found data type: <class 'list'>"

    assert find_data_errors_completion([]) == expected
    assert find_data_error_chat([]) == expected
    assert find_data_error_chat_message([]) == expected
