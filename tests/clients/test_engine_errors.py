"""The public engine-error contract, including failures before inference starts."""

import asyncio
import inspect
from datetime import datetime, timezone
from types import SimpleNamespace

import anyio
import litellm
import pytest

import dspy
from dspy import lm15
from dspy._vendor.lm15._authlock import CredentialLockTimeout
from dspy._vendor.lm15.authkit import DeviceCodeExpiredError
from dspy._vendor.lm15.router import MissingCredentialError
from dspy.clients.engines import AsyncLiteLLMEngine, LiteLLMEngine
from dspy.clients.engines.litellm_errors import to_lm15_error
from dspy.clients.errors import wrap_error
from dspy.utils.callback import BaseCallback

EXPECTED = {
    lm15.LM15Error: dspy.LMUnexpectedError,
    lm15.TransportError: dspy.LMTransportError,
    lm15.LockTimeoutError: dspy.LMLockTimeoutError,
    lm15.StreamAssemblyError: dspy.LMStreamAssemblyError,
    lm15.ConfigurationError: dspy.LMConfigurationError,
    lm15.NotConfiguredError: dspy.LMNotConfiguredError,
    lm15.UnknownModelError: dspy.LMConfigurationError,
    lm15.AmbiguousModelError: dspy.LMConfigurationError,
    lm15.CapabilityError: dspy.LMUnsupportedFeatureError,
    lm15.UnsupportedFeatureError: dspy.LMUnsupportedFeatureError,
    lm15.ProviderError: dspy.LMProviderError,
    lm15.AuthError: dspy.LMAuthError,
    lm15.BillingError: dspy.LMBillingError,
    lm15.RateLimitError: dspy.LMRateLimitError,
    lm15.InvalidRequestError: dspy.LMInvalidRequestError,
    lm15.ContextLengthError: dspy.ContextWindowExceededError,
    lm15.UnsupportedModelError: dspy.LMUnsupportedModelError,
    lm15.TimeoutError: dspy.LMTimeoutError,
    lm15.ServerError: dspy.LMServerError,
    lm15.ToolDerivationError: dspy.LMConfigurationError,
    CredentialLockTimeout: dspy.LMLockTimeoutError,
    DeviceCodeExpiredError: dspy.LMAuthError,
    MissingCredentialError: dspy.LMNotConfiguredError,
}


class Engine:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return lm15.Response(None, request.model, lm15.Message.assistant("ok"), "stop", lm15.Usage(input_tokens=2, output_tokens=1))

    def stream(self, request):
        yield from lm15.response_to_events(self.complete(request))


class AsyncEngine:
    def __init__(self, sync):
        self.sync = sync

    async def complete(self, request):
        return self.sync.complete(request)

    async def stream(self, request):
        for event in self.sync.stream(request):
            yield event


class Sink:
    async def send(self, chunk):
        pass


async def invoke(lm, asynchronous, *args, **kwargs):
    if asynchronous:
        return await lm.acall(*args, **kwargs)
    return await anyio.to_thread.run_sync(lambda: lm(*args, **kwargs))


def test_all_exported_canonical_error_classes_have_an_explicit_policy():
    exported = {value for name in lm15.__all__ if inspect.isclass(value := getattr(lm15, name))
                and issubclass(value, lm15.LM15Error)}
    assert exported <= set(EXPECTED)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("source,target", EXPECTED.items(), ids=[cls.__name__ for cls in EXPECTED])
async def test_public_projection_preserves_metadata_and_cause(source, target, asynchronous, streaming):
    original = source("original failure", provider="fake", provider_code="code", status=429,
                      request_id="request-1", retry_after=7.0)
    engine = Engine(original)
    errors = []

    class Callback(BaseCallback):
        def on_lm_end(self, call_id, outputs, exception):
            errors.append(exception)

    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=False,
                 num_retries=0, callbacks=[Callback()])
    request = lm15.Request(model=lm.model, messages=(lm15.Message.user("hello"),))
    with dspy.context(send_stream=Sink() if streaming else None):
        with pytest.raises(target) as caught:
            await invoke(lm, asynchronous, request)
    error = caught.value
    assert type(error) is target
    assert error.__cause__ is original
    assert error.message == original.message
    assert error.model == lm.model
    assert (error.provider, error.provider_code, error.status, error.request_id, error.retry_after) == (
        "fake", "code", 429, "request-1", 7.0,
    )
    assert errors == [error]
    assert engine.calls == 1
    assert not lm.history


def test_routing_lock_and_partial_diagnostics_survive():
    partial = Engine().complete(lm15.Request(model="custom", messages=(lm15.Message.user("hi"),)))
    for exc, fields in [
        (lm15.AmbiguousModelError("choose", model="ambiguous", providers=("a", "b")), {"providers": ("a", "b"), "model": "ambiguous"}),
        (lm15.UnknownModelError("unknown", rules_tried=("rule",), catalog_searched=True), {"rules_tried": ("rule",), "catalog_searched": True}),
        (lm15.LockTimeoutError("locked", path="keys", lock_path="keys.lock"), {"path": "keys", "lock_path": "keys.lock"}),
        (lm15.StreamAssemblyError("broken", partial=partial, part_index=3), {"partial": partial, "part_index": 3}),
        (lm15.AuthError("denied", env_keys=("KEY",), credential_hint="log in"), {"env_keys": ("KEY",), "credential_hint": "log in"}),
    ]:
        projected = wrap_error(exc, model="custom")
        for name, value in fields.items():
            assert getattr(projected, name) == value
    assert not isinstance(wrap_error(lm15.LockTimeoutError(), model="custom"), dspy.LMProviderError)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("text", ["network invariant failed", "timeout must be positive", "credentials parser bug"])
async def test_unexpected_engine_errors_are_not_guessed_or_retried(asynchronous, text, monkeypatch):
    original = RuntimeError(text)
    original.status_code = 503  # An arbitrary plugin attribute is not SDK evidence.
    engine = Engine(original)
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=False, num_retries=3)
    monkeypatch.setattr("dspy.clients.lm._get_litellm", lambda: pytest.fail("Native errors must not load LiteLLM"))
    with pytest.raises(dspy.LMUnexpectedError) as caught:
        await invoke(lm, asynchronous, "hello")
    assert caught.value.__cause__ is original
    assert engine.calls == 1


@pytest.mark.parametrize("error", [dspy.LMAuthError("existing"), DeprecationWarning("stop"), ImportError("missing SDK")])
def test_already_public_and_python_control_errors_keep_identity(error):
    assert wrap_error(error, model="custom") is error


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_cancellation_is_not_translated(asynchronous):
    original = asyncio.CancelledError("cancelled")
    engine = Engine(original)
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=False, num_retries=3)
    with pytest.raises(asyncio.CancelledError):
        await invoke(lm, asynchronous, "hello")
    assert engine.calls == 1


@pytest.mark.parametrize("error,target", [
    (litellm.BadRequestError("timeout must be positive", model="wire", llm_provider="openai"), lm15.InvalidRequestError),
    (litellm.AuthenticationError("network access denied", model="wire", llm_provider="openai"), lm15.AuthError),
    (litellm.UnsupportedParamsError("timeout unsupported", model="wire", llm_provider="openai"), lm15.UnsupportedFeatureError),
    (litellm.BudgetExceededError(current_cost=2, max_budget=1), lm15.BillingError),
    (litellm.APIConnectionError("failed", model="wire", llm_provider="openai"), lm15.TransportError),
    (litellm.Timeout("failed", model="wire", llm_provider="openai"), lm15.TimeoutError),
])
def test_sdk_classes_override_messages_and_status(error, target):
    mapped = to_lm15_error(error)
    assert isinstance(mapped, target)
    if target in (lm15.InvalidRequestError, lm15.AuthError, lm15.UnsupportedFeatureError, lm15.BillingError):
        assert not dspy.is_retryable_lm_error(wrap_error(mapped, model="custom"))


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_litellm_engine_raises_canonical_errors_and_public_lm_projects_once(asynchronous, monkeypatch):
    sdk = litellm.BadRequestError("timeout must be positive", model="wire", llm_provider="openai")

    def fail(**kwargs):
        raise sdk

    async def afail(**kwargs):
        raise sdk

    monkeypatch.setattr(litellm, "completion", fail)
    monkeypatch.setattr(litellm, "acompletion", afail)
    request = lm15.Request(model="openai/test", messages=(lm15.Message.user("hi"),))
    with pytest.raises(lm15.InvalidRequestError) as canonical:
        if asynchronous:
            await AsyncLiteLLMEngine().complete(request)
        else:
            LiteLLMEngine().complete(request)
    assert canonical.value.__cause__ is sdk
    lm = dspy.LM(request.model, engine="litellm", cache=False, num_retries=0)
    with pytest.raises(dspy.LMInvalidRequestError) as public:
        await invoke(lm, asynchronous, request)
    assert isinstance(public.value.__cause__, lm15.InvalidRequestError)
    assert public.value.__cause__.__cause__ is sdk
    assert public.value.model == "wire"


def test_billing_code_on_generic_429_is_not_retryable():
    sdk = litellm.RateLimitError("quota", model="wire", llm_provider="openai")
    sdk.body = {"error": {"code": "insufficient_quota"}}
    canonical = to_lm15_error(sdk)
    assert isinstance(canonical, lm15.BillingError)
    assert canonical.status == 429
    assert not dspy.is_retryable_lm_error(wrap_error(canonical, model="custom"))
    sdk.llm_provider = "unrelated-provider"
    assert isinstance(to_lm15_error(sdk), lm15.RateLimitError)  # no cross-provider code guesses


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("error_class", [lm15.RateLimitError, lm15.TimeoutError, lm15.TransportError, lm15.ServerError, lm15.LockTimeoutError])
async def test_only_transient_failures_retry_within_budget(asynchronous, error_class):
    engine = Engine(error_class("transient", retry_after=0))
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=False, num_retries=1)
    with pytest.raises(dspy.LMError):
        await invoke(lm, asynchronous, "hello")
    assert engine.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_invalid_request_controls_fail_before_engine_execution(asynchronous):
    engine = Engine()
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=False)
    request = lm15.Request(model=lm.model, messages=(lm15.Message.user("hi"),), config=lm15.Config(extensions={"n": 2}))
    with pytest.raises(ValueError, match="execution controls"):
        await invoke(lm, asynchronous, request)
    assert engine.calls == 0


def test_headers_are_case_insensitive_and_body_metadata_wins(monkeypatch):
    import dspy.clients._http as http

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2025, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(http, "datetime", Clock)
    error = litellm.RateLimitError("wait", model="wire", llm_provider="openai")
    error.response = SimpleNamespace(status_code=429, headers={
        "ReTrY-AfTeR": "Wed, 01 Jan 2025 00:00:09 GMT", "X-Request-ID": "header-id",
    })
    assert to_lm15_error(error).retry_after == 9.0
    assert to_lm15_error(error).request_id == "header-id"
    error.retry_after = 0.0
    error.request_id = "body-id"
    assert to_lm15_error(error).retry_after == 0.0
    assert to_lm15_error(error).request_id == "body-id"


@pytest.mark.parametrize("hint", [float("nan"), float("inf"), -1, "bad", True, object()])
def test_invalid_retry_hint_uses_backoff_without_masking_error(hint):
    from dspy.clients.execution import _delay

    assert _delay(SimpleNamespace(retry_after=hint), 0) == 1
    assert _delay(SimpleNamespace(retry_after=hint), 1000) == 60
    assert _delay(SimpleNamespace(retry_after=120), 0) == 120


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_custom_request_refusal_is_projected_before_adapter_fallback(asynchronous, monkeypatch):
    engine = Engine()
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=False)
    adapter = dspy.ChatAdapter()
    monkeypatch.setattr(adapter, "_make_json_adapter_fallback", lambda: pytest.fail("Setup failures are not parse failures"))
    args = (lm, {"prediction": {"type": "content", "content": "x"}}, dspy.Signature("question -> answer"), [], {"question": "hi"})
    with pytest.raises(dspy.LMUnsupportedFeatureError) as caught:
        if asynchronous:
            await adapter.acall(*args)
        else:
            adapter(*args)
    assert isinstance(caught.value.__cause__, lm15.UnsupportedFeatureError)
    assert engine.calls == 0


@pytest.mark.parametrize("code", ["auth", "billing", "rate_limit", "invalid_request", "context_length", "timeout", "server", "unsupported_model", "unsupported_feature", "not_configured", "unknown_model", "ambiguous_model", "transport", "lock_timeout", "stream_assembly", "provider"])
def test_every_canonical_error_event_uses_the_same_projection(code):
    from dspy._vendor.lm15.errors import error_class_for_code
    from dspy.clients.engines.stream_guard import error_from_event

    event = lm15.StreamErrorEvent(lm15.ErrorDetail(code=code, message="failure"))
    canonical = error_from_event(event, provider="fake")
    assert canonical.code == code
    assert type(wrap_error(canonical, model="custom")) is EXPECTED[error_class_for_code(code)]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_native_http_request_id_reaches_public_error(asynchronous, streaming, monkeypatch):
    from tests.clients.test_lm_engine_execution import DualResponse, native_transport

    transport = native_transport(monkeypatch, [DualResponse(
        status=429, body=b'{"error":{"type":"rate_limit_error","message":"wait"}}',
        headers=[("X-Request-ID", "request-1"), ("rEtRy-AfTeR", "3")],
    )])
    lm = dspy.LM("openai/gpt-4o-mini", engine="lm15", cache=False, num_retries=0)
    try:
        with dspy.context(send_stream=Sink() if streaming else None), pytest.raises(dspy.LMRateLimitError) as caught:
            await invoke(lm, asynchronous, "hello")
        assert caught.value.request_id == "request-1"
        assert caught.value.retry_after == 3.0
        assert caught.value.status == 429
        assert len(transport.requests) == 1
    finally:
        await lm.aclose()


def test_capability_routing_failure_is_projected_without_backend_fallback(monkeypatch):
    from dspy.clients.engines import LM15Engine

    error = lm15.AuthError("denied")

    def fail(*args):
        raise error

    monkeypatch.setattr(LM15Engine, "resolve", fail)
    monkeypatch.setattr("dspy.clients.lm._get_litellm", lambda: pytest.fail("No backend switch on auth errors"))
    with pytest.raises(dspy.LMAuthError) as caught:
        _ = dspy.LM("openai/test").supports_function_calling
    assert caught.value.__cause__ is error
