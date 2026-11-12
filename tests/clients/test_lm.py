"""Public LM behavior through native HTTP and canonical custom engines."""

import json
import warnings
from unittest import mock

import pydantic
import pytest

import dspy
from dspy.lm15 import Config, Message, Request, Usage
from dspy.utils.usage_tracker import track_usage
from tests.clients.test_lm_engine_execution import DualResponse, native_transport, wire
from tests.test_utils.engines import make_response as make_lm_response
from tests.test_utils.engines import recording_lm


def responses_wire(output=None, usage=True):
    return DualResponse(status=200, body=json.dumps({
        "id": "r", "model": "gpt-4o-mini", "status": "completed",
        "output": output if output is not None else [{"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "OK", "annotations": []}]}],
        "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3} if usage else None,
    }).encode())


@pytest.mark.parametrize("engine", ["auto", "lm15"])
@pytest.mark.parametrize("model", ["openai/gpt-4o-mini", "azure/deployment"])
def test_chat_lms_can_be_queried(engine, model, monkeypatch):
    transport = native_transport(monkeypatch, [wire("Hi!")])
    lm = dspy.LM(model, model_type="chat", engine=engine, api_base="https://endpoint.test", api_key="fake", cache=False)
    assert lm("query") == ["Hi!"]
    assert len(transport.requests) == 1
    assert transport.requests[0].url.startswith("https://endpoint.test/")
    lm.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_dspy_cache(asynchronous, monkeypatch, tmp_path):
    transport = native_transport(monkeypatch, [wire("answer"), wire("new answer")])
    original = dspy.cache
    dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True, disk_cache_dir=tmp_path)
    lm = dspy.LM("openai/gpt-4o-mini", model_type="chat")
    try:
        with track_usage() as usage:
            assert (await lm.acall("query") if asynchronous else lm("query")) == ["answer"]
        assert len(usage.usage_data) == 1
        assert len(dspy.cache.memory_cache) == 1
        assert next(iter(dspy.cache.memory_cache)) in dspy.cache.disk_cache
        with track_usage() as cached_usage:
            assert (await lm.acall("query") if asynchronous else lm("query")) == ["answer"]
        assert not cached_usage.usage_data
        assert len(transport.requests) == 1
        assert (await lm.acall("new query") if asynchronous else lm("new query")) == ["new answer"]
        assert len(transport.requests) == len(dspy.cache.memory_cache) == 2
    finally:
        await lm.aclose()
        dspy.cache.disk_cache.close()
        dspy.cache = original


def test_disabled_cache_skips_cache_key(monkeypatch):
    native_transport(monkeypatch, [wire()])
    original = dspy.cache
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    try:
        with mock.patch.object(dspy.cache, "cache_key", wraps=dspy.cache.cache_key) as key:
            assert dspy.LM("openai/gpt-4o-mini", model_type="chat")("hello") == ["hello"]
        key.assert_not_called()
    finally:
        dspy.cache = original


def test_rollout_id_bypasses_cache(monkeypatch):
    transport = native_transport(monkeypatch, [wire(), wire(), wire()])
    lm = dspy.LM("openai/gpt-4o-mini", model_type="chat")
    for prompt, options, fresh in [
        ("query", {"rollout_id": 1}, True), ("query", {"rollout_id": 1}, False),
        ("query", {"rollout_id": 2}, True), ("no rid", {}, True), ("no rid", {"rollout_id": None}, False),
    ]:
        with track_usage() as usage:
            assert lm(prompt, **options) == ["hello"]
        assert bool(usage.usage_data) is fresh
    assert len(transport.requests) == 3
    assert all("rollout_id" not in json.loads(r.body) for r in transport.requests)


def test_zero_temperature_rollout_warns_once(monkeypatch):
    native_transport(monkeypatch, [wire(), wire()])
    lm = dspy.LM("openai/gpt-4o-mini", temperature=0)
    with pytest.warns(UserWarning, match="rollout_id has no effect"):
        lm("query", rollout_id=1)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        lm("query", rollout_id=2)
    assert not records


def test_rollout_id_with_default_temperature_does_not_warn(monkeypatch):
    native_transport(monkeypatch, [wire()])
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        dspy.LM("openai/gpt-5-nano", rollout_id=1)("query")
    assert not records


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_lm_preserves_existing_lm_error_without_self_cause(asynchronous):
    error = dspy.LMRateLimitError("rate limited", model="custom")
    lm = recording_lm([error], num_retries=0)
    with pytest.raises(dspy.LMRateLimitError) as caught:
        await lm.acall("question") if asynchronous else lm("question")
    assert caught.value is error
    assert caught.value.__cause__ is None


def test_exponential_backoff_retry(monkeypatch):
    transport = native_transport(monkeypatch, [wire(status=429) for _ in range(4)])
    delays = []
    with mock.patch("dspy.clients.execution.time", mock.Mock(sleep=delays.append)):
        with pytest.raises(dspy.LMRateLimitError):
            dspy.LM("openai/gpt-4o-mini", cache=False, num_retries=3)("question")
    assert delays == [1, 2, 4]
    assert len(transport.requests) == 4


@pytest.mark.parametrize("model,reasoning", [
    ("openai/o1", True), ("openai/o1-mini", True), ("openai/o1-2023-01-01", True),
    ("openai/o3", True), ("openai/o3-mini-2023-01-01", True), ("openai/gpt-5", True),
    ("openai/gpt-5-mini", True), ("openai/gpt-5-nano", True), ("azure/gpt-5-chat", False),
    ("openai/gpt-4", False), ("anthropic/claude-2", False), ("openai/gpt-5-chat", False),
])
def test_reasoning_model_token_parameter(model, reasoning):
    lm = dspy.LM(model, temperature=1.0 if reasoning else 0.7, max_tokens=16000 if reasoning else 1000)
    field = "max_completion_tokens" if reasoning else "max_tokens"
    assert lm.kwargs[field] == (16000 if reasoning else 1000)
    assert ("max_tokens" if reasoning else "max_completion_tokens") not in lm.kwargs


@pytest.mark.parametrize("model", ["openai/o1", "openai/gpt-5-nano", "openai/gpt-5-mini"])
def test_reasoning_model_requirements(model):
    with pytest.raises(dspy.LMConfigurationError, match="reasoning models require"):
        dspy.LM(model, temperature=0.7, max_tokens=1000)
    assert dspy.LM(model, temperature=1.0, max_tokens=16000).kwargs["max_completion_tokens"] == 16000
    assert dspy.LM(model).kwargs == {"temperature": None, "max_completion_tokens": None}


def test_base_lm_init_uses_lm_defaults_and_isolates_callback_list():
    callbacks = [object()]
    lm = dspy.BaseLM("custom-model", callbacks=callbacks)
    assert lm.kwargs == {"temperature": None, "max_tokens": None}
    assert lm.num_retries == 3
    assert lm.callbacks == callbacks and lm.callbacks is not callbacks


def test_base_lm_runs_its_engine_and_returns_list_outputs():
    lm = recording_lm(["Hi!"], model="custom-model")
    assert lm("Query") == ["Hi!"]
    assert lm.engine.requests[0].messages[0].text == "Query"


def test_base_lm_rejects_engines_that_do_not_return_responses():
    class BadEngine:
        def complete(self, request):
            return ["not a response"]
    lm = dspy.LM("custom-model", engine=BadEngine(), cache=False, num_retries=0)
    with pytest.raises(dspy.LMUnexpectedError, match=r"must return dspy\.lm15\.Response"):
        lm("Query")


def test_base_lm_tracks_usage_for_custom_engines():
    lm = recording_lm([make_lm_response("Hi!", usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2))],
                      model="custom-model")
    with track_usage() as tracker:
        lm("Query")
    usage = tracker.get_total_tokens()["custom-model"]
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (1, 1, 2)


def test_base_lm_copy_is_shallow_runtime_copy_with_isolated_dspy_state():
    class CustomLM(dspy.BaseLM):
        pass
    callback, client = object(), object()
    lm = CustomLM("custom-model", callbacks=[callback], temperature=0.1)
    lm.client = client
    lm.extra_state = {"mutable": []}
    lm.history = [{"prompt": "original"}]
    copied = lm.copy(temperature=0.2, rollout_id=1)
    assert copied is not lm and copied.client is client
    assert copied.extra_state is lm.extra_state
    assert copied.history == [] and copied.history is not lm.history
    assert copied.callbacks == [callback] and copied.callbacks is not lm.callbacks
    assert copied.kwargs == {"temperature": 0.2, "max_tokens": None, "rollout_id": 1}
    assert lm.kwargs == {"temperature": 0.1, "max_tokens": None}


def test_lm_copy_reasoning_model_validation_and_token_normalization():
    """Verify that copying an LM to a reasoning model validates parameters and normalizes tokens (fixes #10577)."""
    # 1. Invalid temperature on copy to reasoning model must raise LMConfigurationError
    lm_chat = dspy.LM("openai/gpt-4o", temperature=0.2)
    with pytest.raises(
        dspy.LMConfigurationError,
        match=r"reasoning models require passing temperature=1\.0 or None",
    ):
        lm_chat.copy(model="openai/gpt-5")

    # 2. Token normalization from max_tokens to max_completion_tokens on copy
    lm_valid = dspy.LM("openai/gpt-4o", max_tokens=16000)
    copied_reasoning = lm_valid.copy(model="openai/gpt-5")
    assert copied_reasoning.kwargs["max_completion_tokens"] == 16000
    assert "max_tokens" not in copied_reasoning.kwargs

    # 3. Token normalization back to max_tokens when copying from reasoning to standard chat model
    copied_back = copied_reasoning.copy(model="openai/gpt-4o")
    assert copied_back.kwargs["max_tokens"] == 16000
    assert "max_completion_tokens" not in copied_back.kwargs

    # 4. Clearing token limit on reasoning model with max_tokens=None or max_completion_tokens=None
    cleared_reasoning_1 = copied_reasoning.copy(max_tokens=None)
    assert "max_tokens" not in cleared_reasoning_1.kwargs
    assert "max_completion_tokens" not in cleared_reasoning_1.kwargs

    cleared_reasoning_2 = copied_reasoning.copy(max_completion_tokens=None)
    assert "max_tokens" not in cleared_reasoning_2.kwargs
    assert "max_completion_tokens" not in cleared_reasoning_2.kwargs

    # 5. Clearing token limit on standard chat model with max_tokens=None or max_completion_tokens=None
    cleared_chat_1 = copied_back.copy(max_tokens=None)
    assert "max_tokens" not in cleared_chat_1.kwargs
    assert "max_completion_tokens" not in cleared_chat_1.kwargs

    cleared_chat_2 = copied_back.copy(max_completion_tokens=None)
    assert "max_tokens" not in cleared_chat_2.kwargs
    assert "max_completion_tokens" not in cleared_chat_2.kwargs

    # 6. Copying default model with explicit max_completion_tokens preserves the explicit limit
    lm_default = dspy.LM("openai/gpt-4o")
    copied_explicit = lm_default.copy(model="openai/gpt-5", max_completion_tokens=16000)
    assert copied_explicit.kwargs["max_completion_tokens"] == 16000
    assert "max_tokens" not in copied_explicit.kwargs

    # 7. Copying default reasoning model to standard model with explicit max_tokens preserves the explicit limit
    lm_default_reasoning = dspy.LM("openai/gpt-5")
    copied_explicit_chat = lm_default_reasoning.copy(model="openai/gpt-4o", max_tokens=2000)
    assert copied_explicit_chat.kwargs["max_tokens"] == 2000
    assert "max_completion_tokens" not in copied_explicit_chat.kwargs


def test_dump_and_load_state():
    lm = dspy.LM("openai/gpt-4o-mini", model_type="chat", temperature=1, max_tokens=100, num_retries=10,
                 launch_kwargs={"temperature": 1}, train_kwargs={"temperature": 5})
    assert lm.dump_state() == {
        "_dspy_lm_class": "dspy.clients.lm.LM", "model": "openai/gpt-4o-mini", "model_type": "chat",
        "temperature": 1, "max_tokens": 100, "num_retries": 10, "cache": True, "finetuning_model": None,
        "launch_kwargs": {"temperature": 1}, "train_kwargs": {"temperature": 5},
    }
    restored = dspy.LM.load_state(lm.dump_state())
    assert isinstance(restored, dspy.LM) and restored.dump_state() == lm.dump_state()


@pytest.mark.parametrize("alias", [False, True])
def test_reasoning_model_load_state_round_trips_canonical_state(alias):
    lm = dspy.LM("openai/gpt-5-nano", temperature=1.0, max_tokens=16000, cache=False, num_retries=1)
    state = lm.dump_state()
    assert "max_completion_tokens" not in state and state["max_tokens"] == 16000
    if alias:
        state["max_completion_tokens"] = state.pop("max_tokens")
    restored = dspy.BaseLM.load_state(state)
    assert isinstance(restored, dspy.LM)
    assert restored.kwargs["max_completion_tokens"] == 16000
    assert restored.dump_state() == lm.dump_state()


def test_dump_state_preserves_enabled_developer_role():
    lm = dspy.LM("openai/gpt-4o-mini", use_developer_role=True)
    assert lm.dump_state()["use_developer_role"] is True
    assert dspy.LM.load_state(lm.dump_state()).use_developer_role is True


def test_dump_state_ignores_internal_class_marker_kwarg():
    lm = dspy.LM("openai/gpt-4o-mini", **{"_dspy_lm_class": "malicious.module.LM"})
    assert lm.dump_state()["_dspy_lm_class"] == "dspy.clients.lm.LM"
    assert lm.kwargs["_dspy_lm_class"] == "malicious.module.LM"


def test_lm_load_state_forwards_allow_custom_lm_class(monkeypatch):
    calls = []
    original = dspy.BaseLM.load_state.__func__

    def spy(cls, state, *, allow_custom_lm_class=False):
        calls.append(allow_custom_lm_class)
        return original(cls, state, allow_custom_lm_class=allow_custom_lm_class)

    monkeypatch.setattr(dspy.BaseLM, "load_state", classmethod(spy))
    dspy.LM.load_state(dspy.LM("openai/gpt-4o-mini").dump_state(), allow_custom_lm_class=True)
    assert calls == [True]


def test_logprobs_included_when_requested(monkeypatch):
    scores = {"content": [{"token": "answer", "logprob": -0.2, "top_logprobs": []}]}
    body = {"model": "gpt-4o-mini", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "answer"}, "logprobs": scores}]}
    transport = native_transport(monkeypatch, [DualResponse(status=200, body=json.dumps(body).encode())])
    lm = dspy.LM("openai/gpt-4o-mini", model_type="chat", logprobs=True, cache=False)
    output = lm("question")
    assert output[0]["text"] == "answer"
    assert output[0]["logprobs"]["content"][0]["logprob"] == -0.2
    assert json.loads(transport.requests[0].body)["logprobs"] is True


@pytest.mark.parametrize("disabled,expected", [(False, 5), (True, 0)])
def test_history_limits_and_disable(disabled, expected, monkeypatch):
    transport = native_transport(monkeypatch, [wire()])
    lm = dspy.LM("openai/gpt-4o-mini")
    with dspy.context(max_history_size=5, disable_history=disabled):
        for _ in range(10):
            lm("query")
    assert len(lm.history) == expected
    assert len(transport.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("usage", [False, True])
async def test_responses_api_reasoning_cache_and_usage(asynchronous, usage, monkeypatch):
    output = [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
              for text in ("part one. ", "part two.")]
    output.append({"type": "reasoning", "id": "reason", "summary": [{"type": "summary_text", "text": "thought"}]})
    transport = native_transport(monkeypatch, [responses_wire(output, usage)])
    lm = dspy.LM("openai/gpt-4o-mini", model_type="responses", api_key="fake")
    try:
        for cached in (False, True):
            with track_usage() as tracker:
                result = await lm.acall("query") if asynchronous else lm("query")
            assert result == [{"text": "part one. part two.", "reasoning_content": "thought"}]
            assert bool(tracker.get_total_tokens()) is (usage and not cached)
        assert lm.history[-1]["usage"] == {}
        assert len(transport.requests) == 1
    finally:
        await lm.aclose()


@pytest.mark.parametrize("model_type", ["auto", "responses"])
def test_lm_replaces_system_with_developer_role(model_type, monkeypatch):
    from dspy.adapters import Prompt
    from dspy.clients.requests import build_request

    transport = native_transport(monkeypatch, [responses_wire()])
    lm = dspy.LM("openai/gpt-4o-mini", model_type=model_type, api_key="fake", use_developer_role=True)
    request = build_request(lm, Prompt(system="hi", messages=(Message.user("q"),)), {})
    assert request.system is None and request.messages[0].role == "developer"
    assert request.messages[0].text == "hi"
    lm(request)
    assert json.loads(transport.requests[0].body)["input"][0]["role"] == "developer"


@pytest.mark.parametrize("extra", [{}, {"caller": None, "namespace": None},
                                    {"caller": {"type": "direct"}, "namespace": "collaboration"}])
def test_responses_api_tool_calls(extra, monkeypatch):
    item = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather",
            "arguments": '{"city":"Paris"}', "status": "completed", **extra}
    native_transport(monkeypatch, [responses_wire([item])])
    lm = dspy.LM("openai/gpt-4o-mini", model_type="responses", api_key="fake", cache=False)
    assert lm("query") == [{"text": None, "tool_calls": [{"type": "function", "id": "call_1", "function": {
        "name": "get_weather", "arguments": json.dumps({"city": "Paris"})}}]}]


@pytest.mark.parametrize("model_type", ["chat", "responses"])
def test_reasoning_controls_reach_native_wire(model_type, monkeypatch):
    transport = native_transport(monkeypatch, [responses_wire() if model_type == "responses" else wire()])
    lm = dspy.LM("openai/gpt-5", api_key="fake", model_type=model_type, reasoning_effort="low", cache=False)
    lm("hi", reasoning={"effort": "high"})
    sent = json.loads(transport.requests[0].body)
    if model_type == "responses":
        assert sent["reasoning"] == {"effort": "high", "summary": "auto"}
        assert "reasoning_effort" not in sent
    else:
        assert sent["reasoning_effort"] == "high"


def test_api_key_not_saved_in_json(tmp_path):
    predict = dspy.Predict("question -> answer")
    predict.lm = dspy.LM("openai/gpt-4o-mini", temperature=1.0, max_tokens=100, api_key="fake-secret")
    path = tmp_path / "program.json"
    predict.save(path)
    state = json.loads(path.read_text())
    assert "api_key" not in state["lm"]
    assert state["lm"]["model"] == "openai/gpt-4o-mini"
    assert state["lm"]["temperature"] == 1.0 and state["lm"]["max_tokens"] == 100


def test_responses_api_with_image_input(monkeypatch):
    from dspy.lm15 import ImagePart

    transport = native_transport(monkeypatch, [responses_wire()])
    lm = dspy.LM("openai/gpt-4o-mini", model_type="responses", api_key="fake", cache=False)
    request = Request(model=lm.model, messages=(Message.user(["describe", ImagePart(data="QUJD", media_type="image/png")]),))
    assert lm(request).text == "OK"
    content = json.loads(transport.requests[0].body)["input"][0]["content"]
    assert {"type": "input_image", "image_url": "data:image/png;base64,QUJD"} in content


@pytest.mark.parametrize("model_type", ["chat", "responses"])
def test_pydantic_response_format_reaches_native_wire(model_type, monkeypatch):
    class Answer(pydantic.BaseModel):
        answer: str

    transport = native_transport(monkeypatch, [responses_wire() if model_type == "responses" else wire()])
    lm = dspy.LM("openai/gpt-4o-mini", model_type=model_type, api_key="fake", cache=False)
    lm("query", response_format=Answer)
    sent = json.loads(transport.requests[0].body)
    fmt = sent["text"]["format"] if model_type == "responses" else sent["response_format"]["json_schema"]
    assert fmt["strict"] is True
    assert fmt["schema"] == {**Answer.model_json_schema(), "additionalProperties": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("order", [("chat", "responses"), ("responses", "chat")])
async def test_cache_keeps_chat_and_responses_calls_separate(asynchronous, order, monkeypatch):
    transport = native_transport(monkeypatch, [wire("chat answer") if kind == "chat" else responses_wire() for kind in order])
    lms = {kind: dspy.LM("openai/gpt-4o-mini", model_type=kind, api_key="fake") for kind in order}
    expected = {"chat": ["chat answer"], "responses": [{"text": "OK"}]}
    try:
        for _ in range(2):
            for kind in order:
                lm = lms[kind]
                assert (await lm.acall("same") if asynchronous else lm("same")) == expected[kind]
        assert len(transport.requests) == 2
    finally:
        for lm in lms.values():
            await lm.aclose()


def _weather_tool():
    return {"type": "function", "function": {"name": "get_weather", "description": "Get weather.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("choice", [
    {"type": "function", "name": "get_weather"}, {"type": "function", "function": {"name": "get_weather"}},
    {"type": "allowed_tools", "allowed_tools": {"mode": "required", "tools": [{"type": "function", "function": {"name": "get_weather"}}]}},
])
async def test_responses_named_choice_accepts_tool_spellings(asynchronous, typed, choice, monkeypatch):
    transport = native_transport(monkeypatch, [responses_wire()])
    lm = dspy.LM("openai/gpt-4o-mini", api_key="fake", model_type="responses", cache=False)
    tool = dspy.lm15.FunctionTool(**_weather_tool()["function"]) if typed else _weather_tool()
    options = {"tools": [tool], "tool_choice": choice}
    try:
        assert (await lm.acall("weather", **options) if asynchronous else lm("weather", **options)) == [{"text": "OK"}]
        sent = json.loads(transport.requests[0].body)
        assert sent["tool_choice"] == {"type": "function", "name": "get_weather"}
        assert sent["tools"][0]["name"] == "get_weather"
    finally:
        await lm.aclose()


def test_lm_responses_engine_writes_lm15_tools_and_documents(monkeypatch):
    from dspy.lm15 import BuiltinTool, DocumentPart, FunctionTool, ToolChoice

    transport = native_transport(monkeypatch, [responses_wire()])
    lm = dspy.LM("openai/gpt-4o-mini", api_key="fake", model_type="responses", cache=False)
    request = Request(model=lm.model, messages=(Message.user(["Read", DocumentPart(data="JVBERi0xLjQK", media_type="application/pdf")]),),
                      tools=(BuiltinTool("web_search", config={"search_context_size": "low"}), FunctionTool(**_weather_tool()["function"])),
                      config=Config(tool_choice=ToolChoice(mode="required", allowed=("get_weather",))))
    lm(request)
    sent = json.loads(transport.requests[0].body)
    assert sent["tools"][0]["type"].startswith("web_search")
    assert sent["tools"][1]["name"] == "get_weather"
    assert sent["tool_choice"] == {"type": "function", "name": "get_weather"}
    assert sent["input"][0]["content"][1]["file_data"] == "data:application/pdf;base64,JVBERi0xLjQK"


def test_lm_reads_responses_api_tool_spellings_from_options():
    from dspy.clients.requests import build_request
    from dspy.lm15 import BuiltinTool, FunctionTool

    lm = dspy.LM("openai/gpt-4o-mini", model_type="responses")
    flat = {"type": "function", "name": "get_time", "parameters": {"type": "object", "properties": {}}, "strict": False}
    request = build_request(lm, "hi", {"tools": [flat, {"type": "web_search_preview"}, _weather_tool()],
                                       "tool_choice": {"type": "function", "name": "get_time"}})
    assert {(type(t), t.name) for t in request.tools} == {
        (FunctionTool, "get_time"), (BuiltinTool, "web_search_preview"), (FunctionTool, "get_weather"),
    }
    assert request.config.tool_choice.allowed == ("get_time",)
    hosted = build_request(lm, "hi", {"tools": [{"type": "web_search_preview"}], "tool_choice": {"type": "web_search_preview"}})
    assert hosted.config.tool_choice.allowed == ("web_search_preview",)


@pytest.mark.parametrize("model", ["openai/", "anthropic/", ""])
def test_empty_model_id_is_refused_at_construction(model):
    with pytest.raises(ValueError, match="model"):
        dspy.LM(model, api_key="k")


def test_responses_only_refusal_names_model_type():
    from dspy.clients.execution import _hint_responses_api

    exc = dspy.LMInvalidRequestError("Function tools are not supported in /v1/chat/completions; use /v1/responses", model="m")
    lm = dspy.LM("openai/gpt-5.6-luna", model_type="responses")
    _hint_responses_api(lm, exc)
    assert "DSPy:" not in str(exc)
    lm = dspy.LM("openai/gpt-5.6-luna", model_type="chat")
    _hint_responses_api(lm, exc)
    _hint_responses_api(lm, exc)
    assert str(exc).count("DSPy:") == 1
    assert "model_type='responses'" in str(exc) and "/v1/responses" in str(exc)


@pytest.mark.parametrize("model,provider,family", [
    ("gpt-6-luna", "openai", "responses"),
    ("openai/gpt-6.1-sol", "openai", "responses"),
    ("azure/opaque-deployment", "azure", "responses"),
    ("anthropic/claude-sonnet-5", "anthropic", "chat"),
    ("gemini/gemini-3-flash-preview", "gemini", "chat"),
    ("openrouter/openai/gpt-6-luna", "openrouter", "chat"),
    ("openai-chat:gpt-6-luna", "openai-chat", "chat"),
    ("unlisted-proxy/model", None, "chat"),
])
def test_auto_api_family_uses_native_provider_route(model, provider, family):
    from dspy.clients.backend_selection import select_backend

    lm = dspy.LM(model)
    assert lm.model_type == "auto"
    selection = select_backend(lm)
    assert selection.model_type == family
    assert (selection.resolution.provider if selection.resolution else None) == provider
    assert selection.native == (provider is not None)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_auto_family_preserves_direct_outputs_cache_copy_and_state(asynchronous, monkeypatch):
    transport = native_transport(monkeypatch, [responses_wire(), responses_wire(), wire("chat"), wire("loaded"),
                                             responses_wire(), wire("copied")])
    lm = dspy.LM("openai/gpt-6-luna", api_key="fake")

    async def call(target, prompt):
        return await target.acall(prompt) if asynchronous else target(prompt)

    try:
        assert await call(lm, "auto family cache contract") == ["OK"]
        assert await call(lm, "auto family cache contract") == ["OK"]
        assert len(transport.requests) == 1
        state = lm.dump_state()
        assert state["model_type"] == "auto"
        restored = dspy.BaseLM.load_state(state).copy(api_key="fake")
        assert restored.model_type == "auto"
        assert await call(restored, "restored auto") == ["OK"]
        chat = lm.copy(model_type="chat")
        assert await call(chat, "auto family cache contract") == ["chat"]
        legacy = dspy.BaseLM.load_state({**state, "model_type": "chat"}).copy(api_key="fake")
        assert await call(legacy, "loaded explicit chat") == ["loaded"]
        response = await call(lm, Request(model=lm.model, messages=(Message.user("typed auto"),)))
        assert isinstance(response, dspy.lm15.Response) and response.text == "OK"
        copied = lm.copy(model="openai-chat:gpt-6-luna")
        assert copied.model_type == "auto"
        assert await call(copied, "copied model") == ["copied"]
        assert [request.url.rsplit("/", 1)[-1] for request in transport.requests] == [
            "responses", "responses", "completions", "completions", "responses", "completions",
        ]
    finally:
        await lm.aclose()


@pytest.mark.parametrize("model_type,path", [(None, "/responses"), ("chat", "/chat/completions")])
def test_react_luna_keeps_tools_and_reasoning_on_selected_family(model_type, path, monkeypatch):
    # Synthetic provider replies: prove DSPy's public program/wire mapping,
    # not that a live deployment accepts this combination.
    if model_type == "chat":
        reply = DualResponse(status=400, body=json.dumps({"error": {"message":
            "Function tools with reasoning_effort are not supported for gpt-6-luna in /v1/chat/completions",
            "type": "invalid_request_error"}}).encode())
    else:
        reply = responses_wire([{"type": "function_call", "call_id": "submit_1", "name": "submit",
                                 "arguments": '{"answer":"23"}'}])
    transport = native_transport(monkeypatch, [reply])
    options = {} if model_type is None else {"model_type": model_type}
    lm = dspy.LM("openai/gpt-6-luna", api_key="fake", cache=False, num_retries=0, **options)
    with dspy.context(lm=lm):
        program = dspy.ReAct("question -> answer", tools=[], max_iters=1)
        if model_type == "chat":
            with pytest.raises(dspy.LMInvalidRequestError, match="Function tools with reasoning_effort"):
                program(question="What is 17 plus 6?")
        else:
            assert program(question="What is 17 plus 6?").answer == "23"
    assert len(transport.requests) == 1  # Never retry on another API or lower effort.
    request = transport.requests[0]
    body = json.loads(request.body)
    assert request.url.endswith(path)
    if model_type == "chat":
        assert body["tools"][0]["function"]["name"] == "submit"
        assert body["reasoning_effort"] == "low"
    else:
        assert body["tools"][0]["name"] == "submit"
        assert body["reasoning"]["effort"] == "low"
    lm.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("engine,extra", [("auto", {"extra_headers": {"x-test": "compat"}}),
                                          ("litellm", {}), ("auto", {"api_version": "2025-01-01-preview"})])
@pytest.mark.parametrize("model,wire_model,family", [
    ("gpt-6-luna", "openai/gpt-6-luna", "responses"),
    ("openai:gpt-6-luna", "openai/gpt-6-luna", "responses"),
    ("azure/opaque-deployment", "azure/opaque-deployment", "responses"),
    ("openrouter/openai/gpt-6-luna", "openrouter/openai/gpt-6-luna", "chat"),
    ("ollama_chat/llama3.1", "ollama_chat/llama3.1", "chat"),
    ("unlisted-proxy/model", "unlisted-proxy/model", "chat"),
])
async def test_auto_family_survives_compatibility(engine, extra, model, wire_model, family, asynchronous, monkeypatch):
    from types import SimpleNamespace

    import dspy.clients.engines.litellm_engine as compatibility

    calls = []

    def complete(api, **kwargs):
        calls.append((api, kwargs))
        return json.loads((responses_wire() if api == "responses" else wire()).read())

    async def aresponses(**kwargs):
        return complete("responses", **kwargs)

    async def acompletion(**kwargs):
        return complete("chat", **kwargs)

    monkeypatch.setattr(compatibility, "get_litellm", lambda **kw: SimpleNamespace(
        responses=lambda **kw: complete("responses", **kw), completion=lambda **kw: complete("chat", **kw),
        aresponses=aresponses, acompletion=acompletion, text_completion=None, atext_completion=None,
    ))
    lm = dspy.LM(model, engine=engine, api_key="fake", api_base="https://private-proxy.invalid/v1",
                 reasoning_effort="high", cache=False, **extra)
    try:
        result = await lm.acall("hello") if asynchronous else lm("hello")
        assert result == (["OK"] if family == "responses" else ["hello"])
        assert len(calls) == 1
        api, sent = calls[0]
        assert api == family
        assert sent["model"] == wire_model
        assert sent["api_base"] == "https://private-proxy.invalid/v1"
        assert sent["reasoning"]["effort"] == "high" if family == "responses" else sent["reasoning_effort"] == "high"
        assert lm.history[-1]["model"] == model
    finally:
        await lm.aclose()


def test_auto_native_only_responses_route_does_not_invent_compatibility_destination(monkeypatch):
    transport = native_transport(monkeypatch, [])
    lm = dspy.LM("moonshotai-responses:kimi-k3", api_key="fake", extra_headers={"x-test": "compat"})
    with pytest.raises(dspy.LMUnsupportedFeatureError, match="no LiteLLM mapping"):
        lm("hi")
    assert not transport.requests


@pytest.mark.parametrize("engine", ["auto", "litellm"])
def test_auto_native_spelling_keeps_react_capabilities_on_compatibility(engine, monkeypatch):
    litellm = pytest.importorskip("litellm")
    calls = []

    def responses(**kwargs):
        calls.append(kwargs)
        return json.loads(responses_wire([{"type": "function_call", "call_id": "s", "name": "submit",
                                          "arguments": '{"answer":"23"}'}]).read())

    monkeypatch.setattr(litellm, "responses", responses)
    lm = dspy.LM("openai:o3-mini", engine=engine, api_key="fake", extra_headers={"x-test": "compat"},
                 cache=False, num_retries=0)
    with dspy.context(lm=lm):
        result = dspy.ReAct("question -> answer", tools=[], max_iters=1)(question="What is 17 plus 6?")
    assert result.answer == "23"
    assert len(calls) == 1
    assert calls[0]["model"] == "openai/o3-mini"
    assert calls[0]["tools"][0]["name"] == "submit"
    assert calls[0]["reasoning"]["effort"] == "low"
    lm.close()
