"""ReActV2 under streamify with a backend that streams native tool calls.

Pinned after cmpnd-ai/breaka-your-lm reported the streamed run failing on
Fireworks (2026-09-16). These tests establish what DSPy's streaming path does
with two kinds of model output: a native tool-call stream finishes like the
plain run, and a call written as JSON text fails the same way with or without
streamify. They do not establish what that live model produced; that needs
the captured request and response.
"""

import json

import pytest

import dspy
from dspy._vendor.lm15.types import (
    StreamDeltaEvent,
    StreamEndEvent,
    StreamStartEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallPart,
)
from dspy.lm15 import Message, Response, Usage


class ScriptedToolEngine:
    """Turn 1 calls add(3, 4); turn 2 calls submit(answer=35)."""

    def __init__(self, *, tool_call_as_text=False):
        self.turns = 0
        self.modes = []
        self.tool_call_as_text = tool_call_as_text

    def _turn(self):
        self.turns += 1
        return ("add", {"a": 3, "b": 4}) if self.turns == 1 else ("submit", {"answer": 35})

    def complete(self, request):
        self.modes.append("complete")
        name, args = self._turn()
        if self.tool_call_as_text:
            return Response(id="r", model=request.model, message=Message.assistant(json.dumps({"name": name, "arguments": args})),
                            finish_reason="stop", usage=Usage())
        return Response(id="r", model=request.model,
                        message=Message(role="assistant", parts=(ToolCallPart(id="c1", name=name, input=args),)),
                        finish_reason="tool_call", usage=Usage())

    def stream(self, request):
        self.modes.append("stream")
        name, args = self._turn()
        yield StreamStartEvent(id="r", model=request.model)
        if self.tool_call_as_text:
            yield StreamDeltaEvent(TextDelta(text=json.dumps({"name": name, "arguments": args})))
            yield StreamEndEvent(finish_reason="stop", usage=Usage())
        else:
            yield StreamDeltaEvent(ToolCallDelta(input=json.dumps(args), id="c1", name=name))
            yield StreamEndEvent(finish_reason="tool_call", usage=Usage())


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def _program():
    return dspy.ReActV2("question -> answer: int", tools=[add, multiply], max_iters=5)


def _drain(stream):
    final = None
    for item in stream:
        if isinstance(item, dspy.Prediction):
            final = item
    return final


def test_react_v2_streams_native_tool_calls_like_the_plain_run():
    plain_engine = ScriptedToolEngine()
    with dspy.context(lm=dspy.LM("scripted/tools", engine=plain_engine, cache=False, num_retries=0)):
        plain = _program()(question="What is (3 + 4) * 5?")
    assert plain.answer == 35 and plain.termination_reason == "submit" and plain_engine.modes == ["complete", "complete"]

    streamed_engine = ScriptedToolEngine()
    with dspy.context(lm=dspy.LM("scripted/tools", engine=streamed_engine, cache=False, num_retries=0)):
        stream = dspy.streamify(_program(), async_streaming=False)
        final = _drain(stream(question="What is (3 + 4) * 5?"))
    assert final.answer == 35 and final.termination_reason == "submit"
    assert streamed_engine.modes == ["stream", "stream"]


def test_tool_call_written_as_text_fails_the_same_way_with_or_without_streamify():
    # The model answers with the call as JSON text instead of a native tool
    # call: no adapter can read it, so the loop ends in a parse error on
    # both paths. Streamify only skips the JSON-adapter retry, because output
    # was already streamed; it does not change what the model did.
    question = "What is (3 + 4) * 5?"
    with dspy.context(lm=dspy.LM("scripted/tools", engine=ScriptedToolEngine(tool_call_as_text=True),
                                 cache=False, num_retries=0)):
        with pytest.raises(dspy.utils.exceptions.AdapterParseError) as plain:
            _program()(question=question)
    with dspy.context(lm=dspy.LM("scripted/tools", engine=ScriptedToolEngine(tool_call_as_text=True),
                                 cache=False, num_retries=0)):
        stream = dspy.streamify(_program(), async_streaming=False)
        with pytest.raises(BaseException) as streamed:
            _drain(stream(question=question))
    inner = streamed.value
    while hasattr(inner, "exceptions"):  # streamify wraps failures in an exception group (3.10 has no name for it)
        inner = inner.exceptions[0]
    assert type(inner) is type(plain.value)
    assert '"name": "submit"' in str(inner) and '"name": "submit"' in str(plain.value)
