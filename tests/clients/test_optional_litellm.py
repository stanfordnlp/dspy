"""Exercise optional compatibility through the public LM contract, without live providers."""

import pytest

import dspy
from dspy.lm15 import Message, Request, request_from_openai_chat
from tests.clients.test_lm_engine_execution import native_transport, wire
from tests.test_utils.server import read_test_server_request_logs


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("engine", ["auto", "litellm"])
async def test_optional_chat_candidates_and_cache(engine, asynchronous, monkeypatch, openai_test_server):
    pytest.importorskip("litellm")
    transport = native_transport(monkeypatch, [])
    url, log_path = openai_test_server
    options = {"extra_headers": {"x-test": "compatibility"}} if engine == "auto" else {}
    lm = dspy.LM("openai/gpt-4o-mini", engine=engine, api_base=url, api_key="fake", num_retries=0, **options)
    try:
        for _ in range(2):
            result = await lm.acall("hello", n=2, seed=11) if asynchronous else lm("hello", n=2, seed=11)
            assert result == ["Hi!", "Hi!"]
        requests = read_test_server_request_logs(log_path)
        assert [body["seed"] for body in requests] == [11, 12]
        assert all(body["n"] == 1 for body in requests)
        assert lm.history[0]["usage"]["total_tokens"] == 40
        assert lm.history[1]["usage"] == {}
        assert not transport.requests
    finally:
        await lm.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_auto_disabled_timeout_uses_compatibility(asynchronous, monkeypatch, openai_test_server):
    pytest.importorskip("litellm")
    import httpx

    transport = native_transport(monkeypatch, [])
    url, log_path = openai_test_server
    lm = dspy.LM("openai/gpt-4o-mini", api_base=url, api_key="fake", cache=False, num_retries=0,
                 timeout=httpx.Timeout(None))
    try:
        assert (await lm.acall("hello") if asynchronous else lm("hello")) == ["Hi!"]
        assert len(read_test_server_request_logs(log_path)) == 1
        assert not transport.requests
    finally:
        await lm.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("engine,model,model_type", [
    ("auto", "openai/gpt-4o-mini", "text"),
    ("litellm", "openai/gpt-4o-mini", "responses"),
    ("auto", "cohere/command-r", "chat"),
])
async def test_optional_endpoints_through_predict(engine, model, model_type, asynchronous, monkeypatch):
    pytest.importorskip("litellm")
    transport = native_transport(monkeypatch, [])
    # Exercise LiteLLM's own mock-response mode, not a replacement DSPy engine.
    lm = dspy.LM(model, engine=engine, model_type=model_type, api_key="fake", cache=False, num_retries=0,
                 mock_response="[[ ## answer ## ]]\nParis\n[[ ## completed ## ]]")
    try:
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
            predict = dspy.Predict("question -> answer")
            result = await predict.acall(question="Capital of France?") if asynchronous else predict(question="Capital of France?")
        assert result.answer == "Paris"
        assert not transport.requests
    finally:
        await lm.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("part", [
    {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}},
    {"type": "file", "file": {"file_id": "file-123"}},
])
async def test_auto_representation_fallback_preserves_media(part, asynchronous, monkeypatch, openai_test_server):
    pytest.importorskip("litellm")
    transport = native_transport(monkeypatch, [])
    url, log_path = openai_test_server
    lm = dspy.LM("openai/gpt-4o-mini", api_base=url, api_key="fake", cache=False, num_retries=0)
    request = request_from_openai_chat({"model": lm.model, "messages": [{"role": "user", "content": [part]}]})
    try:
        response = await lm.acall(request) if asynchronous else lm(request)
        assert response.text == "Hi!"
        assert response.usage.total_tokens == 20
        assert read_test_server_request_logs(log_path)[0]["messages"][0]["content"] == [part]
        assert not transport.requests
    finally:
        await lm.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status,error", [(401, dspy.LMAuthError), (429, dspy.LMRateLimitError)])
async def test_native_failure_never_replays_through_litellm(status, error, asynchronous, monkeypatch, openai_test_server):
    transport = native_transport(monkeypatch, [wire(status=status)])
    url, log_path = openai_test_server
    lm = dspy.LM("openai/gpt-4o-mini", api_base=url, api_key="fake", cache=False, num_retries=0)
    try:
        with pytest.raises(error):
            await lm.acall("hello") if asynchronous else lm("hello")
        assert len(transport.requests) == 1
        assert read_test_server_request_logs(log_path) == []
    finally:
        await lm.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status,error", [(400, dspy.LMInvalidRequestError), (429, dspy.LMRateLimitError)])
async def test_compatibility_errors_keep_public_types(status, error, asynchronous, openai_test_server):
    pytest.importorskip("litellm")
    url, log_path = openai_test_server
    lm = dspy.LM("openai/gpt-4o-mini", engine="litellm", api_base=url, api_key="fake", num_retries=0)
    try:
        for _ in range(2):
            with pytest.raises(error):
                await lm.acall(str(status)) if asynchronous else lm(str(status))
        assert len(read_test_server_request_logs(log_path)) == 2  # Failures are never cached.
        assert lm.history == []
    finally:
        await lm.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_compatibility_chat_stream(asynchronous, openai_test_server):
    pytest.importorskip("litellm")
    from dspy.clients.engines import AsyncLiteLLMEngine, LiteLLMEngine

    url, log_path = openai_test_server
    engine = (AsyncLiteLLMEngine if asynchronous else LiteLLMEngine)(api_base=url, api_key="fake")
    request = Request(model="openai/gpt-4o-mini", messages=(Message.user("hello"),))
    try:
        events = [event async for event in engine.stream(request)] if asynchronous else list(engine.stream(request))
        assert events[0].type == "start"
        assert "".join(event.delta.text for event in events if event.type == "delta") == '{"output_text": "Hello!"}'
        assert events[-1].type == "end"
        assert events[-1].finish_reason == "stop"
        assert len(read_test_server_request_logs(log_path)) == 1
    finally:
        await engine.aclose() if asynchronous else engine.close()
