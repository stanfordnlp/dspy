"""Repair wire JSON, never a tool's literal ``partial_json`` parameter."""

import json
from types import SimpleNamespace

import pytest

import dspy
from dspy import lm15
from dspy.clients.engines.lm15_engine import AsyncLM15Engine, LM15Engine
from dspy.clients.lm15_boundary import response_value
from tests.clients.test_engine_errors import AsyncEngine, Sink, invoke
from tests.test_utils.engines import RecordingEngine, make_response, recording_lm

ARGUMENTS = ['{"query":"cats",}', json.dumps({"partial_json": '{"query":"cats",}'}),
             '{"query":"dogs"}']
EXPECTED = [{"query": "cats"}, {"partial_json": '{"query":"cats",}'}, {"query": "dogs"}]


def body_for(model_type):
    calls = [{"id": f"c{i}", "type": "function", "function": {"name": "search", "arguments": raw}}
             for i, raw in enumerate(ARGUMENTS)]
    if model_type == "chat":
        return {"id": "r", "model": "gpt-4o", "choices": [{"message": {
            "role": "assistant", "content": "", "tool_calls": calls}, "finish_reason": "tool_calls"}]}
    return {"id": "r", "model": "gpt-4o", "status": "completed", "output": [
        {"type": "function_call", "id": f"item{i}", "call_id": call["id"], **call["function"]}
        for i, call in enumerate(calls)]}


def read_raw(body, model_type, request):
    from dspy._vendor.lm15.providers.base import HttpResponse

    if model_type == "chat":
        return lm15.response_from_openai_chat(body, model=request.model)
    return lm15.OpenAILM(api_key="test").parse_response(
        request, HttpResponse(status=200, reason="OK", headers=[], body=json.dumps(body).encode()))


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", ["chat", "responses"])
@pytest.mark.parametrize("reader", ["sdk", "native", "async_native"])
async def test_readers_repair_only_malformed_wire_arguments(model_type, reader, monkeypatch):
    request = lm15.Request(model="openai/gpt-4o", messages=(lm15.Message.user("hi"),))
    body = body_for(model_type)
    original = read_raw(body, model_type, request)
    if reader == "sdk":
        response = response_value(body, model_type, request)
    else:
        engine = (AsyncLM15Engine if reader == "async_native" else LM15Engine)(model_type=model_type)

        async def acomplete(request):
            return original

        provider = SimpleNamespace(complete=acomplete if reader == "async_native" else lambda request: original)
        monkeypatch.setattr(engine, "_target", lambda request: (provider, request))
        response = await engine.complete(request) if reader == "async_native" else engine.complete(request)
    assert [call.input for call in response.tool_calls] == EXPECTED
    assert response.provider_data == original.provider_data
    assert response.usage == original.usage
    assert response.message.continuation == original.message.continuation
    assert response.finish_reason == original.finish_reason


@pytest.mark.parametrize("raw", ['["cats",', '["cats"]', '"cats"'])
def test_nonobject_arguments_are_not_replaced(raw):
    request = lm15.Request(model="openai/gpt-4o", messages=(lm15.Message.user("hi"),))
    body = body_for("chat")
    body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = raw
    original = read_raw(body, "chat", request)
    response = response_value(body, "chat", request)
    assert response.tool_calls[0] == original.tool_calls[0]


def test_canonical_partial_json_without_wire_provenance_is_literal():
    response = make_response(tool_calls=[("c0", "search", EXPECTED[1])])
    lm = recording_lm([response])
    request = lm15.Request(model=lm.model, messages=(lm15.Message.user("hi"),))
    assert lm(request).tool_calls[0].input == EXPECTED[1]


class Search(dspy.Signature):
    question: str = dspy.InputField()
    tools: list[dspy.Tool] = dspy.InputField()
    tool_calls: dspy.ToolCalls = dspy.OutputField()


@pytest.mark.asyncio
@pytest.mark.parametrize("two_step", [False, True])
async def test_adapter_consumers_use_repaired_reader_response(two_step):
    class Reader(RecordingEngine):
        def complete(self, request):
            return response_value(body_for("chat"), "chat", request)

    engine = Reader()
    lm = dspy.LM("openai/gpt-4o", engine=engine, async_engine=AsyncEngine(engine), cache=False)
    adapter = (dspy.TwoStepAdapter(recording_lm(["[[ ## tool_calls ## ]]\n[]\n[[ ## completed ## ]]"]))
               if two_step else dspy.ChatAdapter(use_native_function_calling=True))
    result = await adapter.acall(lm, {}, Search, [], {"question": "hi", "tools": [dspy.Tool(
        lambda query: query, name="search")]})
    assert [call.args for call in result[0]["tool_calls"].tool_calls] == EXPECTED


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_interleaved_stream_repairs_by_call_identity(asynchronous):
    class Streaming(RecordingEngine):
        def stream(self, request):
            yield lm15.StreamStartEvent(model=request.model)
            for offset in (0, 8):
                for index in (2, 0, 1):
                    raw = ARGUMENTS[index]
                    yield lm15.StreamDeltaEvent(delta=lm15.ToolCallDelta(
                        part_index=index + 3, id=f"c{index}", name="search",
                        input=raw[:8] if offset == 0 else raw[8:]))
            yield lm15.StreamEndEvent(finish_reason="tool_call")

    engine = Streaming()
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=False)
    request = lm15.Request(model=lm.model, messages=(lm15.Message.user("hi"),))
    with dspy.context(send_stream=Sink()):
        response = await invoke(lm, asynchronous, request)
    assert [(call.id, call.input) for call in response.tool_calls] == list(zip(["c0", "c1", "c2"], EXPECTED, strict=False))
