"""Offline checks for the coordinated LM engine path."""

import json

import pytest

import dspy
from dspy._vendor.lm15.testing import FakeResponse, FakeTransport
from dspy.lm15 import Config, Message, Request, Response, RouterConfig
from dspy.utils.callback import BaseCallback
from dspy.utils.usage_tracker import track_usage


class DualResponse(FakeResponse):
    async def __aenter__(self):
        source = self

        class AsyncResponse:
            def __getattr__(self, name):
                return getattr(source, name)

            async def read(self):
                return source.read()

            async def __aiter__(self):
                for chunk in source:
                    yield chunk

        return AsyncResponse()

    async def __aexit__(self, *exc):
        pass


def wire(text="hello", status=200):
    body = {"id": "response-1", "model": "gpt-4o-mini", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
    ], "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}
    if status != 200:
        body = {"error": {"message": "try later", "type": "authentication_error" if status == 401 else "rate_limit_error"}}
    return DualResponse(status=status, body=json.dumps(body).encode())


def native_transport(monkeypatch, responses):
    import dspy.clients.execution as execution

    transport = FakeTransport(responses)
    monkeypatch.setattr(execution, "RouterConfig", lambda **kwargs: RouterConfig(
        **{**kwargs, "api_keys": {"openai-chat": "fake", **(kwargs.get("api_keys") or {})}, "transport": transport},
    ))
    return transport


def test_default_native_cache_and_typed_request(monkeypatch):
    transport = native_transport(monkeypatch, [wire(), wire("typed")])
    lm = dspy.LM("openai/gpt-4o-mini", model_type="chat")
    with track_usage() as usage:
        assert lm("hello") == ["hello"]
        assert lm("hello") == ["hello"]
    assert len(transport.requests) == 1
    assert usage.get_total_tokens()[lm.model]["total_tokens"] == 3
    assert lm.history[-1]["usage"] == {}
    request = Request(model=lm.model, messages=(Message.user("explicit"),), config=Config(max_tokens=12))
    response = lm(request)
    assert isinstance(response, Response)
    assert response.text == "typed"
    assert lm.history[-1]["request"] is request
    assert len(transport.requests) == 2


@pytest.mark.asyncio
async def test_async_native_cache_and_history(monkeypatch):
    transport = native_transport(monkeypatch, [wire("async")])
    lm = dspy.LM("openai/gpt-4o-mini", model_type="chat")
    try:
        with track_usage() as usage:
            assert await lm.acall("hello") == ["async"]
            assert await lm.acall("hello") == ["async"]
        assert len(transport.requests) == 1
        assert len(lm.history) == 2
        assert usage.get_total_tokens()[lm.model]["total_tokens"] == 3
    finally:
        await lm.aclose()


def test_native_n_fans_out_but_bookkeeping_is_one_call(monkeypatch):
    transport = native_transport(monkeypatch, [wire("first"), wire("second")])

    class Trace(BaseCallback):
        def __init__(self):
            self.events = []

        def on_lm_start(self, call_id, instance, inputs):
            self.events.append(("start", call_id))

        def on_lm_end(self, call_id, outputs, exception):
            self.events.append(("end", call_id))

    trace = Trace()
    lm = dspy.LM("openai/gpt-4o-mini", model_type="chat", callbacks=[trace])
    with track_usage() as usage:
        assert lm("hello", n=2) == ["first", "second"]
    assert len(transport.requests) == 2
    assert len(lm.history) == 1
    assert trace.events[1] == ("end", trace.events[0][1])
    assert len(trace.events) == 2
    assert usage.get_total_tokens()[lm.model]["total_tokens"] == 6
    assert lm("hello", n=2) == ["first", "second"]
    assert len(transport.requests) == 2


def test_native_retries_without_backend_fallback(monkeypatch):
    import dspy.clients.execution as execution

    transport = native_transport(monkeypatch, [wire(status=429), wire("recovered")])
    delays = []
    monkeypatch.setattr(execution.time, "sleep", delays.append)
    lm = dspy.LM("openai/gpt-4o-mini", model_type="chat", num_retries=1, cache=False)
    assert lm("hello") == ["recovered"]
    assert len(transport.requests) == 2
    assert delays == [1]


def test_provider_auth_error_does_not_switch_backend(monkeypatch):
    transport = native_transport(monkeypatch, [wire(status=401)])
    with pytest.raises(dspy.LMAuthError):
        dspy.LM("openai/gpt-4o-mini", cache=False)("hello")
    assert len(transport.requests) == 1


@pytest.mark.parametrize("options", [
    {"audio": {"voice": "alloy", "format": "wav"}},
    {"questions": {"flag": {"type": "noul", "instructions": "Is this correct?"}}},
])
def test_unrepresentable_ordinary_options_refuse_before_io(monkeypatch, options):
    transport = native_transport(monkeypatch, [])
    lm = dspy.LM("openai/gpt-4o-mini", engine="lm15", cache=False)
    with pytest.raises(dspy.LMUnsupportedFeatureError, match="provider options"):
        lm("hello", **options)
    assert not transport.requests


def test_new_native_cache_uses_restricted_plain_data(monkeypatch, tmp_path):
    transport = native_transport(monkeypatch, [wire()])
    dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=False, disk_cache_dir=tmp_path, restrict_pickle=True)
    try:
        lm = dspy.LM("openai/gpt-4o-mini", model_type="chat")
        assert lm("hello") == ["hello"]
        assert lm("hello") == ["hello"]
        assert len(transport.requests) == 1
    finally:
        dspy.cache.disk_cache.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", [None, "chat"])
async def test_native_streaming_reaches_existing_listener(model_type, monkeypatch):
    text_parts = ("<answer>", "native", "</answer>")
    if model_type == "chat":
        payloads = [{"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
                    for text in text_parts]
        payloads.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}})
    else:
        payloads = [{"type": "response.created", "response": {"id": "r", "model": "gpt-4o-mini"}}]
        payloads.extend({"type": "response.output_text.delta", "output_index": 0, "delta": text} for text in text_parts)
        payloads.append({"type": "response.completed", "response": {
            "id": "r", "status": "completed", "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "".join(text_parts)},
            ]}], "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        }})
    chunks = [f"data: {json.dumps(payload)}\n\n".encode() for payload in payloads]
    chunks.append(b"data: [DONE]\n\n")
    transport = native_transport(monkeypatch, [FakeResponse(status=200, body=b"", chunks=chunks)])
    options = {} if model_type is None else {"model_type": model_type}
    lm = dspy.LM("openai/gpt-4o-mini", api_key="fake", **options)
    predict = dspy.Predict("question -> answer")
    streamer = dspy.streamify(predict, stream_listeners=[dspy.streaming.StreamListener("answer")])
    with dspy.context(lm=lm):
        values = [value async for value in streamer(question="hello")]
        cached = [value async for value in streamer(question="hello")]
    assert values[-1].answer == "native"
    assert "".join(value.chunk for value in values if isinstance(value, dspy.streaming.StreamResponse)) == "native"
    assert len(cached) == 1 and cached[0].answer == "native"
    assert len(transport.requests) == 1
    assert transport.requests[0].url.endswith("/chat/completions" if model_type == "chat" else "/responses")


@pytest.mark.asyncio
async def test_partial_stream_is_not_retried_and_reports_callback_error():
    from dspy.lm15 import StreamDeltaEvent, StreamStartEvent, TextDelta

    class Engine:
        def __init__(self):
            self.calls = 0

        def complete(self, request):
            raise AssertionError("Expected a streaming call")

        def stream(self, request):
            self.calls += 1
            yield StreamStartEvent(model=request.model)
            yield StreamDeltaEvent(TextDelta("partial"))
            raise dspy.LMTransportError("stream broke")

    class Sink:
        async def send(self, chunk):
            pass

    class Trace(BaseCallback):
        error = None

        def on_lm_end(self, call_id, outputs, exception):
            self.error = exception

    engine = Engine()
    trace = Trace()
    lm = dspy.LM("custom", engine=engine, callbacks=[trace], num_retries=3)
    import anyio

    with dspy.context(send_stream=Sink()):
        with pytest.raises(dspy.LMTransportError, match="stream broke"):
            await anyio.to_thread.run_sync(lambda: lm("hello"))
    assert engine.calls == 1
    assert isinstance(trace.error, dspy.LMTransportError)
    assert lm.history == []


def test_dummy_uses_canonical_engine_without_changing_script_behavior():
    from dspy.utils.dummies import DummyLM

    lm = DummyLM([{"answer": "one"}, {"answer": "two"}])
    with dspy.context(lm=lm):
        predict = dspy.Predict("question -> answer")
        assert predict(question="same").answer == "one"
        assert predict(question="same").answer == "two"
    assert len(lm.history) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("provider", ["vertex", "vertex-anthropic"])
@pytest.mark.parametrize("identity", ["environment", "container", "invalid-source"])
async def test_native_vertex_aws_federation(monkeypatch, asynchronous, provider, identity):
    import datetime as dt
    from urllib.parse import unquote

    from dspy._vendor.lm15.cloud.chains import ChainContext
    from dspy.clients.engines.lm15_engine import AsyncLM15Engine, LM15Engine

    audience = "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/aws"
    info = {
        "type": "external_account", "audience": audience,
        "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
        "credential_source": {
            "environment_id": "aws2" if identity == "invalid-source" else "aws1",
            "regional_cred_verification_url":
                "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15",
        },
        "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/test:generateAccessToken",
    }
    env = {"GOOGLE_APPLICATION_CREDENTIALS": "/offline/wif.json", "AWS_REGION": "us-east-2",
           "AWS_DEFAULT_REGION": "eu-west-1", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/v2/credentials/task"}
    if identity != "container":
        env.update(AWS_ACCESS_KEY_ID="AKIDENV", AWS_SECRET_ACCESS_KEY="fake-secret")
    auth_calls = []

    def http(method, url, headers, body, timeout):
        auth_calls.append((method, url))
        if url == "http://169.254.170.2/v2/credentials/task":
            assert identity == "container"
            reply = {"AccessKeyId": "ASIATASK", "SecretAccessKey": "fake-secret", "Token": "task-session"}
        elif url == "https://sts.googleapis.com/v1/token":
            exchange = json.loads(body)
            assert exchange["audience"] == audience
            assert exchange["subjectTokenType"] == info["subject_token_type"]
            signed = json.loads(unquote(exchange["subjectToken"]))
            assert signed["method"] == "POST"
            assert signed["url"] == "https://sts.us-east-2.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15"
            signing_headers = {h["key"]: h["value"] for h in signed["headers"]}
            key = "ASIATASK" if identity == "container" else "AKIDENV"
            assert f"Credential={key}/20261113/us-east-2/sts/aws4_request" in signing_headers["Authorization"]
            assert signing_headers["x-goog-cloud-target-resource"] == audience
            assert signing_headers.get("x-amz-security-token") == ("task-session" if identity == "container" else None)
            reply = {"access_token": "federated", "expires_in": 3600, "token_type": "Bearer"}
        else:
            assert url == info["service_account_impersonation_url"]
            assert headers["authorization"] == "Bearer federated"
            reply = {"accessToken": "sa-token", "expireTime": "2099-01-01T00:00:00Z"}
        return 200, {}, json.dumps(reply).encode()

    monkeypatch.setattr(ChainContext, "online", classmethod(lambda cls, env: cls(
        env=env, files={"/offline/wif.json": json.dumps(info)}, http=http,
        now=lambda: dt.datetime(2026, 11, 13, 12, tzinfo=dt.timezone.utc),
    )))
    reply = ({"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]}
             if provider == "vertex" else {"role": "assistant", "content": [{"type": "text", "text": "ok"}],
                                           "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 2}})
    transport = FakeTransport([DualResponse(200, json.dumps(reply).encode())])
    config = RouterConfig(env=env, credentials={provider: "workload"}, transport=transport,
                          settings={provider: {"project": "offline-project", "location": "us-central1"}})
    model = "gemini-2.5-flash" if provider == "vertex" else "claude-sonnet-4-5"
    lm = dspy.LM(f"{provider}:{model}", cache=False, num_retries=0,
                 engine=LM15Engine(config, model_type="auto"), async_engine=AsyncLM15Engine(config, model_type="auto"))
    try:
        if identity == "invalid-source":
            with pytest.raises(dspy.LMNotConfiguredError):
                await lm.acall("hello") if asynchronous else lm("hello")
            assert not auth_calls and not transport.requests and not lm.history
        else:
            assert (await lm.acall("hello") if asynchronous else lm("hello")) == ["ok"]
            assert len(auth_calls) == (3 if identity == "container" else 2)
            assert len(transport.requests) == 1 and len(lm.history) == 1
            sent = transport.requests[0]
            assert {k.lower(): v for k, v in sent.headers}["authorization"] == "Bearer sa-token"
            assert "/projects/offline-project/locations/us-central1/" in sent.url
    finally:
        await lm.aclose()
