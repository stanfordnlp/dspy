import pytest
from unittest.mock import patch
import dspy
from dspy.adapters import RetryAdapter, ChatAdapter, JSONAdapter
from dspy.signatures.signature import Signature
from dspy.utils.dummies import DummyLM


class MockSignature(Signature):
    question: str = dspy.InputField()
    answer: int = dspy.OutputField()


@pytest.mark.parametrize(
    "demos",
    [
        [],
        [dspy.Example({"question": "6 x 7", "answer": 42})],
    ],
)
@pytest.mark.parametrize(
    "max_retries",
    [
        0,
        3,
    ],
)
@pytest.mark.parametrize(
    "n",
    [
        1,
        3,
    ],
)
def test_adapter_max_retry(demos, max_retries, n):
    main_adapter = ChatAdapter()
    fallback_adapter = JSONAdapter()
    adapter = RetryAdapter(main_adapter=main_adapter, fallback_adapter=fallback_adapter, max_retries=max_retries)
    lm = DummyLM([{"answer": "42"}] * (n * 2 + max_retries))
    inputs = {"question": "6 x 7"}

    with dspy.context(lm=lm):
        with (
            patch.object(
                main_adapter,
                "parse",
                side_effect=ValueError("error"),
            ) as mock_main_parse,
            patch.object(
                main_adapter,
                "format",
                wraps=main_adapter.format,
            ) as mock_main_format,
            patch.object(
                fallback_adapter,
                "parse",
                side_effect=ValueError("error"),
            ) as mock_fallback_parse,
        ):
            with pytest.raises(ValueError, match="Failed to parse LM outputs for maximum retries"):
                adapter(lm, {"n": n}, MockSignature, demos, inputs)

    assert mock_main_parse.call_count == n + max_retries
    assert mock_fallback_parse.call_count == n
    assert lm.call_count == max_retries + 2

    assert mock_main_format.call_count == max_retries + 1
    _, kwargs = mock_main_format.call_args
    assert kwargs["inputs"]["previous_response"] == "[[ ## answer ## ]]\n42"
    assert kwargs["inputs"]["error_message"] == "error"
    assert kwargs["inputs"]["question"] == "6 x 7"


def test_adapter_fallback():
    main_adapter = JSONAdapter()
    fallback_adapter = ChatAdapter()
    adapter = RetryAdapter(main_adapter=main_adapter, fallback_adapter=fallback_adapter, max_retries=1)
    lm = DummyLM([{"answer": "42"}] * 3)
    inputs = {"question": "6 x 7"}

    with dspy.context(lm=lm):
        with (
            patch.object(
                main_adapter,
                "parse",
                side_effect=ValueError("error"),
            ) as mock_main_parse,
        ):
            result = adapter(lm, {}, MockSignature, [], inputs)

    assert result == [{"answer": 42}]
    assert mock_main_parse.call_count == 1
