import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

import dspy
from dspy.utils.dummies import DummyLM
from dspy.utils.otel import OtelCallback


@pytest.fixture(autouse=True)
def reset_settings():
    original_settings = dspy.settings.copy()

    yield

    dspy.configure(**original_settings)


@pytest.fixture()
def tracing():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider


def get_span(spans, name):
    matches = [span for span in spans if span.name == name]
    assert len(matches) == 1, f"Expected exactly one span named {name!r}, got {[s.name for s in spans]}"
    return matches[0]


def test_chain_of_thought_emits_nested_spans(tracing):
    exporter, provider = tracing
    lm = DummyLM([{"reasoning": "France has a capital.", "answer": "Paris"}])
    dspy.configure(lm=lm, callbacks=[OtelCallback(tracer_provider=provider)])

    dspy.ChainOfThought("question -> answer")(question="What is the capital of France?")

    spans = exporter.get_finished_spans()
    cot_span = get_span(spans, "ChainOfThought")
    predict_span = get_span(spans, "Predict")
    format_span = get_span(spans, "ChatAdapter.format")
    chat_span = get_span(spans, "chat dummy")
    parse_span = get_span(spans, "ChatAdapter.parse")

    # All spans belong to one trace, rooted at the ChainOfThought span.
    assert {span.context.trace_id for span in spans} == {cot_span.context.trace_id}
    assert cot_span.parent is None
    assert predict_span.parent.span_id == cot_span.context.span_id
    for span in (format_span, chat_span, parse_span):
        assert span.parent.span_id == predict_span.context.span_id

    assert cot_span.attributes["dspy.module.type"] == "ChainOfThought"
    module_input = json.loads(cot_span.attributes["dspy.module.input"])
    assert module_input["kwargs"]["question"] == "What is the capital of France?"
    assert "Paris" in cot_span.attributes["dspy.module.output"]

    assert predict_span.attributes["dspy.module.signature"] == "question -> reasoning, answer"

    assert chat_span.kind == SpanKind.CLIENT
    assert chat_span.attributes["gen_ai.operation.name"] == "chat"
    assert chat_span.attributes["gen_ai.request.model"] == "dummy"
    input_messages = json.loads(chat_span.attributes["gen_ai.input.messages"])
    assert [message["role"] for message in input_messages] == ["system", "user"]
    output_messages = json.loads(chat_span.attributes["gen_ai.output.messages"])
    assert output_messages[0]["role"] == "assistant"
    assert "Paris" in output_messages[0]["parts"][0]["content"]

    assert format_span.attributes["dspy.adapter.method"] == "format"
    assert parse_span.attributes["dspy.adapter.method"] == "parse"


def test_capture_content_disabled_omits_payloads(tracing):
    exporter, provider = tracing
    lm = DummyLM([{"answer": "Paris"}])
    dspy.configure(lm=lm, callbacks=[OtelCallback(tracer_provider=provider, capture_content=False)])

    dspy.Predict("question -> answer")(question="What is the capital of France?")

    spans = exporter.get_finished_spans()
    content_attributes = {
        "dspy.module.input",
        "dspy.module.output",
        "dspy.adapter.input",
        "dspy.adapter.output",
        "gen_ai.input.messages",
        "gen_ai.output.messages",
    }
    for span in spans:
        assert not content_attributes & set(span.attributes)

    # Non-content attributes are still present.
    chat_span = get_span(spans, "chat dummy")
    assert chat_span.attributes["gen_ai.request.model"] == "dummy"
    assert get_span(spans, "Predict").attributes["dspy.module.signature"] == "question -> answer"


def test_capture_content_env_var(tracing, monkeypatch):
    _, provider = tracing
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false")
    assert OtelCallback(tracer_provider=provider).capture_content is False

    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
    assert OtelCallback(tracer_provider=provider).capture_content is True

    monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")
    assert OtelCallback(tracer_provider=provider).capture_content is True

    # An explicit argument wins over the environment variable.
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false")
    assert OtelCallback(tracer_provider=provider, capture_content=True).capture_content is True


def test_typed_lm_span_carries_usage(tracing):
    exporter, provider = tracing

    class TypedLM(dspy.BaseLM):
        forward_contract = "typed_lm"

        def forward(self, request):
            return dspy.LMResponse.from_text(
                "hello",
                model=request.model,
                usage={"prompt_tokens": 3, "completion_tokens": 5},
            )

    dspy.configure(callbacks=[OtelCallback(tracer_provider=provider)])
    lm = TypedLM(model="test/typed")

    with dspy.context(experimental=True):
        lm(prompt="hi")

    chat_span = get_span(exporter.get_finished_spans(), "chat typed")
    assert chat_span.attributes["gen_ai.provider.name"] == "test"
    assert chat_span.attributes["gen_ai.request.model"] == "typed"
    assert chat_span.attributes["gen_ai.usage.input_tokens"] == 3
    assert chat_span.attributes["gen_ai.usage.output_tokens"] == 5
    output_messages = json.loads(chat_span.attributes["gen_ai.output.messages"])
    assert output_messages == [{"role": "assistant", "parts": [{"type": "text", "content": "hello"}]}]


def test_tool_span_records_exception(tracing):
    exporter, provider = tracing
    dspy.configure(callbacks=[OtelCallback(tracer_provider=provider)])

    def divide(numerator: int, denominator: int) -> float:
        """Divide two numbers."""
        return numerator / denominator

    tool = dspy.Tool(divide)
    with pytest.raises(ZeroDivisionError):
        tool(numerator=1, denominator=0)

    tool_span = get_span(exporter.get_finished_spans(), "execute_tool divide")
    assert tool_span.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool_span.attributes["gen_ai.tool.name"] == "divide"
    assert json.loads(tool_span.attributes["gen_ai.tool.call.arguments"]) == {"numerator": 1, "denominator": 0}
    assert tool_span.status.status_code == StatusCode.ERROR
    assert tool_span.events[0].name == "exception"


def test_tool_span_records_result(tracing):
    exporter, provider = tracing
    dspy.configure(callbacks=[OtelCallback(tracer_provider=provider)])

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    dspy.Tool(add)(a=1, b=2)

    tool_span = get_span(exporter.get_finished_spans(), "execute_tool add")
    assert json.loads(tool_span.attributes["gen_ai.tool.call.result"]) == 3


@pytest.mark.asyncio
async def test_async_calls_emit_nested_spans(tracing):
    exporter, provider = tracing
    lm = DummyLM([{"answer": "Paris"}])
    dspy.configure(lm=lm, callbacks=[OtelCallback(tracer_provider=provider)])

    await dspy.Predict("question -> answer").acall(question="What is the capital of France?")

    spans = exporter.get_finished_spans()
    predict_span = get_span(spans, "Predict")
    chat_span = get_span(spans, "chat dummy")
    assert predict_span.parent is None
    assert chat_span.parent.span_id == predict_span.context.span_id


def test_evaluate_spans_nest_across_threads(tracing):
    exporter, provider = tracing
    lm = DummyLM([{"answer": "Paris"}, {"answer": "Rome"}, {"answer": "Berlin"}])
    dspy.configure(lm=lm, callbacks=[OtelCallback(tracer_provider=provider)])

    devset = [
        dspy.Example(question="capital of France?", answer="Paris").with_inputs("question"),
        dspy.Example(question="capital of Italy?", answer="Rome").with_inputs("question"),
        dspy.Example(question="capital of Germany?", answer="Berlin").with_inputs("question"),
    ]
    evaluate = dspy.Evaluate(
        devset=devset,
        metric=lambda example, prediction, trace=None: 1.0,
        num_threads=2,
    )
    result = evaluate(dspy.Predict("question -> answer"))

    spans = exporter.get_finished_spans()
    evaluate_span = get_span(spans, "Evaluate")
    assert evaluate_span.attributes["dspy.evaluate.num_examples"] == 3
    assert evaluate_span.attributes["dspy.evaluate.score"] == float(result.score)

    # Spans created inside worker threads stay in the Evaluate trace.
    predict_spans = [span for span in spans if span.name == "Predict"]
    assert len(predict_spans) == 3
    for span in predict_spans:
        assert span.context.trace_id == evaluate_span.context.trace_id
        assert span.parent.span_id == evaluate_span.context.span_id


def test_enable_otel_tracing_is_idempotent(tracing):
    _, provider = tracing

    callback = dspy.enable_otel_tracing(tracer_provider=provider)
    assert dspy.enable_otel_tracing(tracer_provider=provider) is callback
    assert [cb for cb in dspy.settings.callbacks if isinstance(cb, OtelCallback)] == [callback]


def test_unsupported_semconv_version(tracing):
    _, provider = tracing
    with pytest.raises(ValueError, match="Unsupported semconv_version"):
        OtelCallback(tracer_provider=provider, semconv_version=2)
