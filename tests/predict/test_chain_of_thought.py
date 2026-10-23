from unittest import mock

import pytest

import dspy
from dspy import ChainOfThought
from dspy.utils import DummyLM
from dspy.utils.exceptions import AdapterParseError
from tests.test_utils.engines import make_response


def test_initialization_with_string_signature():
    lm = DummyLM([{"plan": "find the number after 1", "answer": "2"}])
    dspy.configure(lm=lm)
    predict = ChainOfThought("question -> answer")
    assert list(predict.predict.signature.output_fields.keys()) == [
        "reasoning",
        "answer",
    ]
    assert predict(question="What is 1+1?").answer == "2"


@pytest.mark.asyncio
async def test_async_chain_of_thought():
    lm = DummyLM([{"plan": "find the number after 1", "answer": "2"}])
    with dspy.context(lm=lm):
        program = ChainOfThought("question -> answer")
        result = await program.acall(question="What is 1+1?")
        assert result.answer == "2"


def test_chain_of_thought_with_native_reasoning():
    """Native-capable models need no explicit reasoning or plan output."""

    lm = dspy.LM(model="anthropic/claude-3-7-sonnet-20250219", cache=False)
    dspy.settings.configure(lm=lm)

    with (
        mock.patch.object(dspy.LM, "supports_reasoning", new_callable=mock.PropertyMock, return_value=True),
        mock.patch("dspy.clients.engines.LM15Engine.complete") as mock_completion,
    ):
        mock_completion.return_value = make_response(
            "[[ ## answer ## ]]\nParis\n[[ ## completed ## ]]",
            thinking="Paris is the capital of France.",
            model="anthropic/claude-3-7-sonnet-20250219",
        )

        cot = ChainOfThought("question -> answer")
        result = cot(question="What is the capital of France?")
        assert result.answer == "Paris"
        assert isinstance(result.reasoning, dspy.Reasoning)
        assert result.reasoning == "Paris is the capital of France."

        request = mock_completion.call_args.args[0]
        assert request.config.reasoning.effort == "low"
        assert "[[ ## reasoning ## ]]" not in str(request.messages)
        assert "[[ ## plan ## ]]" not in str(request.messages)


def test_chain_of_thought_with_manual_reasoning():
    """Test ChainOfThought with manual reasoning where LM doesn't support native reasoning."""
    lm = dspy.LM(model="openai/gpt-4o-mini")
    dspy.settings.configure(lm=lm)

    with mock.patch("dspy.clients.engines.LM15Engine.complete") as mock_completion:
        mock_completion.return_value = make_response(
            "[[ ## plan ## ]]\nRecall the capital of France.\n"
            "[[ ## answer ## ]]\nParis\n[[ ## completion ## ]]",
            model="openai/gpt-4o-mini",
        )

        cot = ChainOfThought("question -> answer")
        result = cot(question="What is the capital of France?")
        assert result.answer == "Paris"
        assert result.reasoning == "Recall the capital of France."


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("adapter_type", [dspy.ChatAdapter, dspy.JSONAdapter, dspy.XMLAdapter])
@pytest.mark.parametrize("mode", ["native", "omitted", "unsupported", "disabled", "disabled_string"])
async def test_default_reasoning_modes(mode, adapter_type, async_mode):
    class Model(DummyLM):
        @property
        def supports_reasoning(self):
            return mode != "unsupported"

        @property
        def supported_params(self):
            return ["response_format", "reasoning_effort"]

        @property
        def supports_response_schema(self):
            return True

    native = mode in ("native", "omitted")
    adapter = adapter_type()
    responses = [{"answer": "17"}, {"answer": "29"}]
    if not native:
        for response in responses:
            response["plan"] = "Use the supplied observations."
    # Some models return native thinking even when effort is disabled; preserve the explicit field.
    lm = Model(responses, reasoning=mode in ("native", "disabled", "disabled_string"), adapter=adapter)
    lm.kwargs["reasoning_effort"] = "high"
    program = ChainOfThought("question -> answer", n=2)
    config = {"reasoning_effort": None} if mode == "disabled" else {}
    if mode == "disabled_string":
        config = {"reasoning_effort": "none"}

    with dspy.context(lm=lm, adapter=adapter):
        result = await program.acall(question="Use observations", config=config) if async_mode else program(question="Use observations", config=config)

    expected = "Some reasoning" if mode == "native" else None if native else "Use the supplied observations."
    assert result.reasoning == expected
    assert result.completions.reasoning == [expected, expected]
    assert result.completions.answer == ["17", "29"]
    if adapter_type is dspy.JSONAdapter:
        schema = lm.history[0]["request"].config.response_format["schema"]
        assert set(schema["properties"]) == ({"answer"} if native else {"plan", "answer"})
        if not native:
            assert schema["properties"]["plan"]["type"] == "string"


@pytest.mark.parametrize("adapter_type", [dspy.ChatAdapter, dspy.JSONAdapter, dspy.XMLAdapter])
def test_reasoning_fallback_preserves_demos_and_history(tmp_path, adapter_type):
    class Task(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        plan: str = dspy.OutputField()
        answer: str = dspy.OutputField()

    program = ChainOfThought(Task)
    program.predict.demos = [dspy.Example(question="Demo", reasoning="Demo reasoning", plan="Demo plan", answer="13")]
    program.save(tmp_path / "program.json")
    loaded = ChainOfThought(Task)
    loaded.load(tmp_path / "program.json")
    history = dspy.History(messages=[{"question": "Previous", "reasoning": "Previous reasoning", "plan": "Previous plan", "answer": "19"}])
    adapter = adapter_type()
    lm = DummyLM([{"plan_2": "New reasoning", "plan": "New plan", "answer": "23"}], adapter=adapter)
    with dspy.context(lm=lm, adapter=adapter):
        result = loaded(question="Current", history=history)
    assert result.reasoning == "New reasoning"
    assert result.plan == "New plan"
    messages = "\n".join(m["content"] for m in lm.history[0]["messages"])
    assert "plan_2" in messages
    assert "Demo reasoning" in messages and "Previous reasoning" in messages and "Previous plan" in messages
    assert loaded.predict.demos[0]["reasoning"] == "Demo reasoning"
    assert history.messages[0]["reasoning"] == "Previous reasoning"


@pytest.mark.parametrize("custom", [False, True])
def test_explicit_rationale_is_not_renamed(custom):
    field = dspy.OutputField(desc="Custom working") if custom else None
    if field is not None:
        field.annotation = str
    program = ChainOfThought("question -> answer", rationale_field=field, rationale_field_type=str)
    with dspy.context(lm=DummyLM([{"reasoning": "Working", "answer": "31"}])):
        result = program(question="Question")
    assert result.reasoning == "Working"
    assert isinstance(result.reasoning, str)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("collision", [False, True])
@pytest.mark.parametrize("adapter", [
    dspy.ChatAdapter(use_json_adapter_fallback=False),
    dspy.JSONAdapter(),
    dspy.XMLAdapter(use_json_adapter_fallback=False),
])
async def test_parse_failure_preserves_predictor_signature(adapter, collision, async_mode):
    program = ChainOfThought("question -> plan, answer" if collision else "question -> answer")
    response = {"plan_2": "Working", "plan": "Task plan"} if collision else {"plan": "Working"}
    lm = DummyLM([response], adapter=adapter)

    with dspy.context(lm=lm, adapter=adapter), pytest.raises(AdapterParseError) as exc_info:
        if async_mode:
            await program.acall(question="Question")
        else:
            program(question="Question")

    error = exc_info.value
    assert error.signature is program.predict.signature
    assert error.parsed_result == ({"reasoning": "Working", "plan": "Task plan"} if collision else {"reasoning": "Working"})
    assert error.lm_response == lm.history[-1]["outputs"][0]
    assert error.adapter_name == type(adapter).__name__
