import pytest

import dspy
from tests.adapters.conftest import format_messages_and_lm_kwargs
from tests.test_utils.engines import recording_lm


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


def _assert_two_step_requests(main_lm, extraction_lm):
    # main LM call
    [main_request] = main_lm.engine.requests
    assert len(main_request.messages) == 1
    assert "1. `question` (str)" in main_request.system
    assert "1. `solution` (str)" in main_request.system
    assert "2. `answer` (float)" in main_request.system
    assert main_request.messages[0].role == "user"
    assert "question:" in main_request.messages[0].text.lower()
    assert "What is 5 + 7?" in main_request.messages[0].text

    # extraction LM call
    [extraction_request] = extraction_lm.engine.requests
    assert len(extraction_request.messages) == 1
    assert "`text` (str)" in extraction_request.system
    assert "`solution` (str)" in extraction_request.system
    assert "`answer` (float)" in extraction_request.system
    assert extraction_request.messages[0].role == "user"
    assert "text from main LM" in extraction_request.messages[0].text


EXTRACTED = """
[[ ## solution ## ]] result
[[ ## answer ## ]] 12
[[ ## completed ## ]]
"""


def test_two_step_adapter_call():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField(desc="The math question to solve")
        solution: str = dspy.OutputField(desc="Step by step solution")
        answer: float = dspy.OutputField(desc="The final numerical answer")

    program = dspy.Predict(TestSignature)
    main_lm = recording_lm(["text from main LM"], model="openai/gpt-4o-mini", temperature=1.0)
    extraction_lm = recording_lm([EXTRACTED], model="openai/gpt-4o", temperature=1.0)

    dspy.configure(lm=main_lm, adapter=dspy.TwoStepAdapter(extraction_model=extraction_lm))

    result = program(question="What is 5 + 7?")

    assert result.answer == 12
    _assert_two_step_requests(main_lm, extraction_lm)


@pytest.mark.asyncio
async def test_two_step_adapter_async_call():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField(desc="The math question to solve")
        solution: str = dspy.OutputField(desc="Step by step solution")
        answer: float = dspy.OutputField(desc="The final numerical answer")

    program = dspy.Predict(TestSignature)
    main_lm = recording_lm(["text from main LM"], model="openai/gpt-4o-mini", temperature=1.0)
    extraction_lm = recording_lm([EXTRACTED], model="openai/gpt-4o", temperature=1.0)

    with dspy.context(lm=main_lm, adapter=dspy.TwoStepAdapter(extraction_model=extraction_lm)):
        result = await program.acall(question="What is 5 + 7?")

    assert result.answer == 12
    _assert_two_step_requests(main_lm, extraction_lm)


def test_two_step_adapter_parse():
    class ComplexSignature(dspy.Signature):
        input_text: str = dspy.InputField()
        tags: list[str] = dspy.OutputField(desc="List of relevant tags")
        confidence: float = dspy.OutputField(desc="Confidence score")

    first_response = "main LM response"

    # The chat-format parse fails on this JSON reply; the JSON fallback reads it.
    reply = """
        {
            "tags": ["AI", "deep learning", "neural networks"],
            "confidence": 0.87
        }
    """
    extraction_lm = recording_lm([reply, reply], model="openai/gpt-4o", temperature=1.0)
    adapter = dspy.TwoStepAdapter(extraction_lm)
    dspy.configure(adapter=adapter, lm=extraction_lm)

    result = adapter.parse(ComplexSignature, first_response)

    assert result["tags"] == ["AI", "deep learning", "neural networks"]
    assert result["confidence"] == 0.87


def test_two_step_adapter_parse_errors():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    first_response = "main LM response"
    # The extraction LM never answers in the chat format, so the chat parse and
    # its JSON fallback both fail.
    extraction_lm = recording_lm(["invalid response", "invalid response"], model="openai/gpt-4o", temperature=1.0)

    adapter = dspy.TwoStepAdapter(extraction_lm)

    with pytest.raises(dspy.AdapterParseError, match="Failed to parse response"):
        adapter.parse(TestSignature, first_response)
