"""Migration warnings target callers, not DSPy's temporary adapter internals."""

import asyncio
import inspect
import warnings
from pathlib import Path

import pytest

import dspy
from dspy.adapters.baml_adapter import BAMLAdapter
from dspy.clients.call_result import CallResult
from dspy.clients.engines import AsyncLiteLLMEngine, LiteLLMEngine
from dspy.lm15 import Message, Request, Response, Usage
from dspy.utils.callback import BaseCallback
from dspy.utils.dummies import DummyLM

_GUIDE = "https://dspy.ai/community/normalized-lm-api-migration/"


class Engine:
    def __init__(self, text="ok"):
        self.text = text
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return Response(None, request.model, Message.assistant(self.text), "stop", Usage())


class AsyncEngine:
    def __init__(self, sync):
        self.sync = sync

    async def complete(self, request):
        return self.sync.complete(request)


def make_lm(text="ok", *, cache=False, callbacks=None):
    engine = Engine(text)
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=cache, callbacks=callbacks)
    return lm, engine


def migration_warnings(recorded):
    return [item for item in recorded if _GUIDE in str(item.message)]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("callbacks", [False, True])
async def test_dictionary_messages_warn_at_the_call_site_even_on_cache_hits(asynchronous, callbacks):
    lm, engine = make_lm(cache=True, callbacks=[BaseCallback()] if callbacks else None)
    messages = [{"role": "user", "content": "hello"}]
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        for _ in range(2):
            if asynchronous:
                line = inspect.currentframe().f_lineno + 1
                result = await lm.acall(messages=messages)
            else:
                line = inspect.currentframe().f_lineno + 1
                result = lm(messages=messages)
            assert result == ["ok"]

    found = migration_warnings(recorded)
    assert len(found) == 2
    assert all(item.category is DeprecationWarning for item in found)
    assert all(Path(item.filename).resolve() == Path(__file__).resolve() for item in found)
    assert all(item.lineno == line for item in found)
    assert all("OpenAI-style message dictionaries" in str(item.message) for item in found)
    assert all("removal in DSPy 3.5" in str(item.message) for item in found)
    assert all("lm('hello') remains supported" in str(item.message) for item in found)
    assert all(_GUIDE + "#migrating-openai-style-messages" in str(item.message) for item in found)
    assert len(engine.requests) == 1
    assert messages == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_message_objects_cannot_bypass_the_deprecated_keyword(asynchronous):
    lm, engine = make_lm()
    # Canonical Message objects belong in Request.messages, not lm(messages=).
    # Diagnose the keyword before any backend-specific object conversion.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(DeprecationWarning, match="through messages="):
            if asynchronous:
                await lm.acall(messages=[Message.user("hello")])
            else:
                lm(messages=[Message.user("hello")])
    assert not engine.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_prompt_and_canonical_request_calls_stay_quiet(asynchronous):
    lm, engine = make_lm()
    request = Request(model=lm.model, messages=(Message.user("hello"),))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        if asynchronous:
            assert await lm.acall("hello") == ["ok"]
            response = await lm.acall(request)
        else:
            assert lm("hello") == ["ok"]
            response = lm(request)
    assert isinstance(response, Response)
    assert response.text == "ok"
    assert len(engine.requests) == 2
    assert not migration_warnings(recorded)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("adapter_kind,text", [
    ("chat", "[[ ## answer ## ]]\nok"),
    ("json", '{"answer":"ok"}'),
    ("xml", "<answer>ok</answer>"),
    ("baml", '{"answer":"ok"}'),
    ("two_step", "[[ ## answer ## ]]\nok"),
])
async def test_builtin_adapter_calls_stay_quiet_without_hiding_later_user_calls(asynchronous, adapter_kind, text):
    lm, engine = make_lm(text)
    adapter = {
        "chat": dspy.ChatAdapter(use_json_adapter_fallback=False),
        "json": dspy.JSONAdapter(),
        "xml": dspy.XMLAdapter(use_json_adapter_fallback=False),
        "baml": BAMLAdapter(),
        "two_step": dspy.TwoStepAdapter(lm),
    }[adapter_kind]
    program = dspy.Predict("question -> answer")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        with dspy.context(lm=lm, adapter=adapter):
            result = await program.acall(question="hello") if asynchronous else program(question="hello")
        assert result.answer == "ok"
        assert not migration_warnings(recorded)
        assert len(engine.requests) == (2 if adapter_kind == "two_step" else 1)

        messages = [{"role": "user", "content": "another call"}]
        if asynchronous:
            await lm.acall(messages=messages)
        else:
            lm(messages=messages)
    assert len(migration_warnings(recorded)) == 1


def test_dummy_program_calls_stay_quiet():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        with dspy.context(lm=DummyLM([{"answer": "ok"}])):
            assert dspy.Predict("question -> answer")(question="hello").answer == "ok"
    assert not migration_warnings(recorded)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_streamified_programs_stay_quiet(asynchronous):
    from dspy.lm15 import response_to_events

    class StreamingEngine(Engine):
        def stream(self, request):
            return response_to_events(self.complete(request))

    class AsyncStreamingEngine(AsyncEngine):
        async def stream(self, request):
            for event in self.sync.stream(request):
                yield event

    engine = StreamingEngine("[[ ## answer ## ]]\nok")
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncStreamingEngine(engine), cache=False)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
            stream = dspy.streamify(dspy.Predict("question -> answer"), is_async_program=asynchronous)
            values = [value async for value in stream(question="hello")]
    assert values[-1].answer == "ok"
    assert len(engine.requests) == 1
    assert not migration_warnings(recorded)


@pytest.mark.asyncio
async def test_adapter_origin_does_not_leak_to_concurrent_user_call():
    lm, engine = make_lm("[[ ## answer ## ]]\nok")
    entered, release = asyncio.Event(), asyncio.Event()

    class BlockingEngine(AsyncEngine):
        async def complete(self, request):
            if "[[ ## question ## ]]" in request.messages[-1].text:
                entered.set()
                await release.wait()
            return await super().complete(request)

    lm = lm.copy(async_engine=BlockingEngine(engine))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
            task = asyncio.create_task(dspy.Predict("question -> answer").acall(question="hello"))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                await lm.acall(messages=[{"role": "user", "content": "user call"}])
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
            assert task.result().answer == "ok"
    found = migration_warnings(recorded)
    assert len(found) == 1
    assert "OpenAI-style message dictionaries" in str(found[0].message)


def test_callback_call_on_another_lm_is_not_hidden_even_with_same_messages():
    other, _ = make_lm()

    class Callback(BaseCallback):
        def on_lm_start(self, call_id, instance, inputs):
            other(messages=inputs["messages"])

    lm, _ = make_lm("[[ ## answer ## ]]\nok", callbacks=[Callback()])
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
            assert dspy.Predict("question -> answer")(question="hello").answer == "ok"
    found = migration_warnings(recorded)
    assert len(found) == 1
    assert "OpenAI-style message dictionaries" in str(found[0].message)


def test_nested_engine_call_with_different_messages_is_not_hidden():
    lm, engine = make_lm("[[ ## answer ## ]]\nok")
    complete = engine.complete
    nested = False

    def with_nested_call(request):
        nonlocal nested
        if not nested:
            nested = True
            lm(messages=[{"role": "user", "content": "nested user call"}])
        return complete(request)

    engine.complete = with_nested_call
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
            assert dspy.Predict("question -> answer")(question="hello").answer == "ok"
    assert len(migration_warnings(recorded)) == 1
    assert len(engine.requests) == 2


def test_adapter_origin_is_restored_after_failure():
    lm, engine = make_lm()

    def fail(request):
        raise dspy.LMAuthError("denied")

    engine.complete = fail
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
            with pytest.raises(dspy.LMAuthError):
                dspy.Predict("question -> answer")(question="hello")
        assert not migration_warnings(recorded)
        with pytest.raises(dspy.LMAuthError):
            lm(messages=[{"role": "user", "content": "hello"}])
    assert len(migration_warnings(recorded)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_message_warning_as_error_stops_before_engine_execution(asynchronous):
    lm, engine = make_lm()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(DeprecationWarning, match="OpenAI-style"):
            if asynchronous:
                await lm.acall(messages=[{"role": "user", "content": "hello"}])
            else:
                lm(messages=[{"role": "user", "content": "hello"}])
    assert not engine.requests
    assert not lm.history


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_custom_legacy_shortcut_warns_but_canonical_method_does_not(asynchronous):
    class ShortcutEngine(Engine):
        def complete_legacy(self, lm, request, **context):
            return CallResult(outputs=["ok"], response_model=lm.model)

    class AsyncShortcutEngine(AsyncEngine):
        async def complete_legacy(self, lm, request, **context):
            return self.sync.complete_legacy(lm, request, **context)

    engine = ShortcutEngine()
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncShortcutEngine(engine), cache=False)
    with pytest.warns(DeprecationWarning, match=r"complete_legacy\(\).*removal in DSPy 3.5"):
        assert (await lm.acall("hello") if asynchronous else lm("hello")) == ["ok"]
    request = Request(model=lm.model, messages=(Message.user("hello"),))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        response = await lm.acall(request) if asynchronous else lm(request)
    assert response.text == "ok"
    assert not migration_warnings(recorded)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("form", ["named", "explicit", "subclass"])
async def test_litellm_backend_itself_is_not_deprecated(asynchronous, form, monkeypatch):
    def complete(self, lm, request, **context):
        return CallResult(outputs=["ok"], response_model=lm.model)

    async def acomplete(self, lm, request, **context):
        return complete(self, lm, request, **context)

    monkeypatch.setattr(LiteLLMEngine, "complete_legacy", complete)
    monkeypatch.setattr(AsyncLiteLLMEngine, "complete_legacy", acomplete)
    class InheritedEngine(LiteLLMEngine):
        pass

    class AsyncInheritedEngine(AsyncLiteLLMEngine):
        pass

    if form == "named":
        options = {"engine": "litellm"}
    elif form == "subclass":
        options = {"engine": InheritedEngine(), "async_engine": AsyncInheritedEngine()}
    else:
        options = {"engine": LiteLLMEngine(), "async_engine": AsyncLiteLLMEngine()}
    lm = dspy.LM("custom", cache=False, **options)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        assert (await lm.acall("hello") if asynchronous else lm("hello")) == ["ok"]
    assert not migration_warnings(recorded)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_shortcut_warning_as_error_stops_before_execution(asynchronous):
    class ShortcutEngine(Engine):
        def complete_legacy(self, *args, **kwargs):
            raise AssertionError("A warnings-as-errors policy must prevent execution")

    class AsyncShortcutEngine(AsyncEngine):
        async def complete_legacy(self, *args, **kwargs):
            raise AssertionError("A warnings-as-errors policy must prevent execution")

    engine = ShortcutEngine()
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncShortcutEngine(engine), cache=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(DeprecationWarning, match="complete_legacy"):
            if asynchronous:
                await lm.acall("hello")
            else:
                lm("hello")
    assert not engine.requests
    assert not lm.history
