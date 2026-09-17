"""The 3.5 cutoff applies to legacy integration and explicit wrappers alike."""

import inspect
import warnings
from pathlib import Path

import pytest

import dspy
from dspy.clients.engines import AsyncLegacyEngine, LegacyEngine
from dspy.lm15 import Message, Request, Response, Usage
from dspy.utils.callback import BaseCallback
from dspy.utils.dummies import DummyLM


class LegacyLM(dspy.BaseLM):
    forward_contract = "legacy"

    def forward(self, prompt=None, messages=None, **kwargs):
        self.calls = getattr(self, "calls", 0) + 1
        return {
            "model": self.model,
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        }

    async def aforward(self, **kwargs):
        return self.forward(**kwargs)


def migration_warnings(recorded):
    return [item for item in recorded if "https://dspy.ai/community/normalized-lm-api-migration/" in str(item.message)]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("callbacks", [False, True])
async def test_legacy_warning_identifies_the_call_site(asynchronous, typed, callbacks):
    lm = LegacyLM("custom", callbacks=[BaseCallback()] if callbacks else None)
    argument = Request(model=lm.model, messages=(Message.user("hello"),)) if typed else "hello"

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        if asynchronous:
            line = inspect.currentframe().f_lineno + 1
            result = await lm.acall(argument)
        else:
            line = inspect.currentframe().f_lineno + 1
            result = lm(argument)

    found = migration_warnings(recorded)
    assert len(found) == 1
    assert found[0].category is DeprecationWarning
    assert Path(found[0].filename).resolve() == Path(__file__).resolve()
    assert found[0].lineno == line
    message = str(found[0].message)
    assert "complete(Request) -> Response" in message
    assert "supported throughout DSPy 3.4" in message
    assert "scheduled for removal in 3.5" in message
    assert "along with LegacyEngine and AsyncLegacyEngine" in message
    assert "https://dspy.ai/community/normalized-lm-api-migration/#custom-engines-and-legacy-plugins" in message
    if typed:
        assert isinstance(result, Response)
        assert result.text == "ok"
    else:
        assert result == ["ok"]
    assert lm.calls == 1
    assert len(lm.history) == 1


def test_python_warning_filters_control_repetition_and_copies():
    lm = LegacyLM("custom")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        lm("hello")
        lm("hello")
        copied = lm.copy()
        copied("hello")
    assert len(migration_warnings(recorded)) == 3
    assert not hasattr(lm, "_warned_legacy_engine")
    assert not hasattr(copied, "_warned_legacy_engine")

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("default", DeprecationWarning)
        for instance in (lm, lm, copied):
            instance("hello")
    assert len(migration_warnings(recorded)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["direct", "ordinary", "typed"])
async def test_explicit_legacy_engines_warn_at_construction_not_each_execution(asynchronous, mode):
    plugin = LegacyLM("custom")
    request = Request(model=plugin.model, messages=(Message.user("hello"),))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        sync_line = inspect.currentframe().f_lineno + 1
        sync_engine = LegacyEngine(plugin)
        async_line = inspect.currentframe().f_lineno + 1
        async_engine = AsyncLegacyEngine(plugin)
        if mode != "direct":
            lm = dspy.LM(plugin.model, engine=sync_engine, async_engine=async_engine, cache=False, num_retries=0)
            argument = request if mode == "typed" else "hello"
            result = await lm.acall(argument) if asynchronous else lm(argument)
            if mode == "typed":
                assert isinstance(result, Response)
                assert result.text == "ok"
            else:
                assert result == ["ok"]
        else:
            result = await async_engine.complete(request) if asynchronous else sync_engine.complete(request)
            assert result.text == "ok"
    found = migration_warnings(recorded)
    assert len(found) == 2
    assert [item.lineno for item in found] == [sync_line, async_line]
    assert all(Path(item.filename).resolve() == Path(__file__).resolve() for item in found)
    assert all("LegacyEngine and AsyncLegacyEngine are deprecated" in str(item.message) for item in found)
    assert all("scheduled for removal in DSPy 3.5" in str(item.message) for item in found)
    assert plugin.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("base", [dspy.LM, DummyLM])
async def test_builtin_forward_overrides_warn(asynchronous, base):
    class OverriddenLM(base):
        forward = LegacyLM.forward
        aforward = LegacyLM.aforward

    lm = OverriddenLM([]) if base is DummyLM else OverriddenLM("custom", cache=False)
    with pytest.warns(DeprecationWarning, match="scheduled for removal in 3.5"):
        result = await lm.acall("hello") if asynchronous else lm("hello")
    assert result == ["ok"]
    assert lm.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_engine_integrations_do_not_warn(asynchronous):
    class Engine:
        def complete(self, request):
            return Response(None, request.model, Message.assistant("ok"), "stop", Usage())

    class AsyncEngine:
        async def complete(self, request):
            return Engine().complete(request)

    models = [DummyLM([{"answer": "ok"}]), dspy.LM("custom", engine=Engine(), async_engine=AsyncEngine(), cache=False)]
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        for lm in models:
            if asynchronous:
                await lm.acall("hello")
            else:
                lm("hello")
    assert not migration_warnings(recorded)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_removed_typed_contract_does_not_receive_legacy_warning(asynchronous):
    class RemovedLM(LegacyLM):
        forward_contract = "typed_lm"

    lm = RemovedLM("custom")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        with pytest.raises(TypeError, match="typed_lm contract was removed"):
            if asynchronous:
                await lm.acall("hello")
            else:
                lm("hello")
    assert not migration_warnings(recorded)
    assert not hasattr(lm, "calls")


@pytest.mark.parametrize("engine_class", [LegacyEngine, AsyncLegacyEngine])
def test_wrapper_warning_as_error_stops_at_construction(engine_class):
    plugin = LegacyLM("custom")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(DeprecationWarning, match="LegacyEngine and AsyncLegacyEngine are deprecated"):
            engine_class(plugin)
    assert not hasattr(plugin, "calls")


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_program_warns_about_legacy_lm_not_internal_messages(asynchronous):
    class ProgramLM(LegacyLM):
        def forward(self, **kwargs):
            raw = super().forward(**kwargs)
            raw["choices"][0]["message"]["content"] = "[[ ## answer ## ]]\nok"
            return raw

    lm = ProgramLM("custom")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        with dspy.context(lm=lm):
            program = dspy.Predict("question -> answer")
            result = await program.acall(question="hello") if asynchronous else program(question="hello")
    assert result.answer == "ok"
    found = migration_warnings(recorded)
    assert len(found) == 1
    assert "Implementing custom LMs" in str(found[0].message)
    assert lm.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("adapter_class", [dspy.ChatAdapter, dspy.JSONAdapter])
async def test_warning_as_error_does_not_trigger_adapter_fallback(asynchronous, adapter_class):
    starts = []

    class Callback(BaseCallback):
        def on_lm_start(self, call_id, instance, inputs):
            starts.append(call_id)

    class SchemaLM(LegacyLM):
        supported_params = frozenset({"response_format"})
        supports_response_schema = True

    lm = SchemaLM("custom", callbacks=[Callback()])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with dspy.context(lm=lm, adapter=adapter_class()):
            program = dspy.Predict("question -> answer")
            with pytest.raises(DeprecationWarning, match="Implementing custom LMs"):
                if asynchronous:
                    await program.acall(question="hello")
                else:
                    program(question="hello")
    assert len(starts) == 1
    assert not hasattr(lm, "calls")


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_two_step_extraction_preserves_warning_as_error(asynchronous):
    extraction = LegacyLM("custom")
    main = DummyLM([{"answer": "ok"}])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with dspy.context(lm=main, adapter=dspy.TwoStepAdapter(extraction)):
            program = dspy.Predict("question -> answer")
            with pytest.raises(DeprecationWarning, match="Implementing custom LMs"):
                if asynchronous:
                    await program.acall(question="hello")
                else:
                    program(question="hello")
    assert not hasattr(extraction, "calls")
