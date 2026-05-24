from typing import Any, ClassVar
from unittest import mock

import pytest

import dspy
from tests.adapters.conftest import format_messages_and_lm_kwargs


class AsyncNativeToolLM:
    model = "openai/gpt-5-nano"
    model_type = "chat"
    kwargs: ClassVar[dict[str, Any]] = {}
    supported_params = frozenset()
    supports_function_calling = True
    supports_reasoning = False
    supports_response_schema = False

    def __init__(self, output):
        self.output = output
        self.messages = None
        self.call_kwargs = None

    async def acall(self, messages, **kwargs):
        self.messages = messages
        self.call_kwargs = kwargs
        return [self.output]


def test_two_step_adapter_format_exact_messages_for_simple_signature_with_demo():
    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    adapter = dspy.TwoStepAdapter(dspy.utils.DummyLM([{"answer": "x"}]))
    messages, lm_kwargs = format_messages_and_lm_kwargs(adapter, QA, [{"question": "Q1", "answer": "A1"}], {"question": "Q2"})

    expected_messages = [{"role": "system",
      "content": "You are a helpful assistant that can solve tasks based on user input.\n"
                 "As input, you will be provided with:\n"
                 "1. `question` (str):\n"
                 "Your outputs must contain:\n"
                 "1. `answer` (str):\n"
                 "You should lay out your outputs in detail so that your answer can be understood by "
                 "another agent\n"
                 "Specific instructions: Given the fields `question`, produce the fields `answer`."},
     {"role": "user", "content": "question: Q1"},
     {"role": "assistant", "content": "answer: A1"},
     {"role": "user", "content": "question: Q2"}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_two_step_adapter_format_exact_messages_with_typed_outputs():
    class TypedSignature(dspy.Signature):
        question: str = dspy.InputField()
        count: int = dspy.OutputField()
        answer: str = dspy.OutputField()

    adapter = dspy.TwoStepAdapter(dspy.utils.DummyLM([{"count": 1, "answer": "x"}]))
    messages, lm_kwargs = format_messages_and_lm_kwargs(adapter, TypedSignature, [], {"question": "Q"})

    expected_messages = [{"role": "system",
      "content": "You are a helpful assistant that can solve tasks based on user input.\n"
                 "As input, you will be provided with:\n"
                 "1. `question` (str):\n"
                 "Your outputs must contain:\n"
                 "1. `count` (int): \n"
                 "2. `answer` (str):\n"
                 "You should lay out your outputs in detail so that your answer can be understood by "
                 "another agent\n"
                 "Specific instructions: Given the fields `question`, produce the fields `count`, "
                 "`answer`."},
     {"role": "user", "content": "question: Q"}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_two_step_adapter_call():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField(desc="The math question to solve")
        solution: str = dspy.OutputField(desc="Step by step solution")
        answer: float = dspy.OutputField(desc="The final numerical answer")

    program = dspy.Predict(TestSignature)

    mock_main_lm = mock.MagicMock(spec=dspy.LM)
    mock_main_lm.return_value = ["text from main LM"]
    mock_main_lm.kwargs = {"temperature": 1.0}
    mock_main_lm.model = "openai/gpt-4o-mini"

    mock_extraction_lm = mock.MagicMock(spec=dspy.LM)
    mock_extraction_lm.return_value = [
        """
[[ ## solution ## ]] result
[[ ## answer ## ]] 12
[[ ## completed ## ]]
"""
    ]
    mock_extraction_lm.kwargs = {"temperature": 1.0}
    mock_extraction_lm.model = "openai/gpt-4o"

    dspy.configure(lm=mock_main_lm, adapter=dspy.TwoStepAdapter(extraction_model=mock_extraction_lm))

    result = program(question="What is 5 + 7?")

    assert result.answer == 12

    # main LM call
    mock_main_lm.assert_called_once()
    _, call_kwargs = mock_main_lm.call_args
    assert len(call_kwargs["messages"]) == 2

    # assert first message
    assert call_kwargs["messages"][0]["role"] == "system"
    content = call_kwargs["messages"][0]["content"]
    assert "1. `question` (str)" in content
    assert "1. `solution` (str)" in content
    assert "2. `answer` (float)" in content

    # assert second message
    assert call_kwargs["messages"][1]["role"] == "user"
    content = call_kwargs["messages"][1]["content"]
    assert "question:" in content.lower()
    assert "What is 5 + 7?" in content

    # extraction LM call
    mock_extraction_lm.assert_called_once()
    _, call_kwargs = mock_extraction_lm.call_args
    assert len(call_kwargs["messages"]) == 2

    # assert first message
    assert call_kwargs["messages"][0]["role"] == "system"
    content = call_kwargs["messages"][0]["content"]
    assert "`text` (str)" in content
    assert "`solution` (str)" in content
    assert "`answer` (float)" in content

    # assert second message
    assert call_kwargs["messages"][1]["role"] == "user"
    content = call_kwargs["messages"][1]["content"]
    assert "text from main LM" in content


@pytest.mark.asyncio
async def test_two_step_adapter_async_call():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField(desc="The math question to solve")
        solution: str = dspy.OutputField(desc="Step by step solution")
        answer: float = dspy.OutputField(desc="The final numerical answer")

    program = dspy.Predict(TestSignature)

    mock_main_lm = mock.MagicMock(spec=dspy.LM)
    mock_main_lm.acall.return_value = ["text from main LM"]
    mock_main_lm.kwargs = {"temperature": 1.0}
    mock_main_lm.model = "openai/gpt-4o-mini"

    mock_extraction_lm = mock.MagicMock(spec=dspy.LM)
    mock_extraction_lm.acall.return_value = [
        """
[[ ## solution ## ]] result
[[ ## answer ## ]] 12
[[ ## completed ## ]]
"""
    ]
    mock_extraction_lm.kwargs = {"temperature": 1.0}
    mock_extraction_lm.model = "openai/gpt-4o"

    with dspy.context(lm=mock_main_lm, adapter=dspy.TwoStepAdapter(extraction_model=mock_extraction_lm)):
        result = await program.acall(question="What is 5 + 7?")

    assert result.answer == 12

    # main LM call
    mock_main_lm.acall.assert_called_once()
    _, call_kwargs = mock_main_lm.acall.call_args
    assert len(call_kwargs["messages"]) == 2

    # assert first message
    assert call_kwargs["messages"][0]["role"] == "system"
    content = call_kwargs["messages"][0]["content"]
    assert "1. `question` (str)" in content
    assert "1. `solution` (str)" in content
    assert "2. `answer` (float)" in content

    # assert second message
    assert call_kwargs["messages"][1]["role"] == "user"
    content = call_kwargs["messages"][1]["content"]
    assert "question:" in content.lower()
    assert "What is 5 + 7?" in content

    # extraction LM call
    mock_extraction_lm.acall.assert_called_once()
    _, call_kwargs = mock_extraction_lm.acall.call_args
    assert len(call_kwargs["messages"]) == 2

    # assert first message
    assert call_kwargs["messages"][0]["role"] == "system"
    content = call_kwargs["messages"][0]["content"]
    assert "`text` (str)" in content
    assert "`solution` (str)" in content
    assert "`answer` (float)" in content

    # assert second message
    assert call_kwargs["messages"][1]["role"] == "user"
    content = call_kwargs["messages"][1]["content"]
    assert "text from main LM" in content


@pytest.mark.asyncio
async def test_two_step_adapter_async_call_uses_normalized_request_pipeline_with_tools():
    def search(query: str) -> str:
        """Search for documents."""
        return query

    class ToolSignature(dspy.Signature):
        question: str = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        answer: str = dspy.OutputField()
        tool_calls: dspy.ToolCalls = dspy.OutputField()

    main_lm = AsyncNativeToolLM(
        {
            "text": "The answer is in the documents.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"query": "documents"}'},
                }
            ],
        }
    )

    extraction_lm = mock.MagicMock(spec=dspy.LM)
    extraction_lm.acall.return_value = [
        """
[[ ## answer ## ]] extracted answer
[[ ## completed ## ]]
"""
    ]
    extraction_lm.kwargs = {"temperature": 1.0}
    extraction_lm.model = "openai/gpt-4o"

    adapter = dspy.TwoStepAdapter(extraction_model=extraction_lm, use_native_function_calling=True)
    lm_kwargs = {"temperature": 0.2}

    result = await adapter.acall(
        main_lm,
        lm_kwargs,
        ToolSignature,
        [],
        {"question": "What should I search?", "tools": [dspy.Tool(search)]},
    )

    assert lm_kwargs == {"temperature": 0.2}
    assert main_lm.call_kwargs["temperature"] == 0.2
    assert main_lm.call_kwargs["tools"][0]["function"]["name"] == "search"
    assert "tools:" not in main_lm.messages[-1]["content"]
    assert result[0]["answer"] == "extracted answer"
    assert result[0]["tool_calls"] == dspy.ToolCalls.from_dict_list(
        [{"name": "search", "args": {"query": "documents"}}]
    )


def test_two_step_adapter_parse():
    class ComplexSignature(dspy.Signature):
        input_text: str = dspy.InputField()
        tags: list[str] = dspy.OutputField(desc="List of relevant tags")
        confidence: float = dspy.OutputField(desc="Confidence score")

    first_response = "main LM response"

    mock_extraction_lm = mock.MagicMock(spec=dspy.LM)
    mock_extraction_lm.return_value = [
        """
        {
            "tags": ["AI", "deep learning", "neural networks"],
            "confidence": 0.87
        }
    """
    ]
    mock_extraction_lm.kwargs = {"temperature": 1.0}
    mock_extraction_lm.model = "openai/gpt-4o"
    adapter = dspy.TwoStepAdapter(mock_extraction_lm)
    dspy.configure(adapter=adapter, lm=mock_extraction_lm)

    result = adapter.parse(ComplexSignature, first_response)

    assert result["tags"] == ["AI", "deep learning", "neural networks"]
    assert result["confidence"] == 0.87


def test_two_step_adapter_parse_errors():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    first_response = "main LM response"

    mock_extraction_lm = mock.MagicMock(spec=dspy.LM)
    mock_extraction_lm.return_value = ["invalid response"]
    mock_extraction_lm.kwargs = {"temperature": 1.0}
    mock_extraction_lm.model = "openai/gpt-4o"

    adapter = dspy.TwoStepAdapter(mock_extraction_lm)

    with pytest.raises(ValueError, match="Failed to parse response"):
        adapter.parse(TestSignature, first_response)
