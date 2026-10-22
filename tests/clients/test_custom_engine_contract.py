"""The custom-engine contract: a borrowed engine owns its connection, the
engine pair is one unit, and a custom engine is saved through its own state.

Pinned from cmpnd-ai/breaka-your-lm results/CUSTOM_ENGINE_FINDINGS.md
(2026-09-16): api_key/api_base/timeout/extra_headers were dropped before a
custom engine saw the request, the constructor and copy() applied different
rules to async_engine, and dump_state refused custom engines with no
documented way out.
"""

import json
import os
import re
import subprocess
import sys

import pytest

import dspy
from dspy.lm15 import Message, Request, Response, Usage


class Echo:
    """A stateful engine that knows how to save and restore itself."""

    def __init__(self, tag="t"):
        self.tag = tag

    def complete(self, request):
        return Response(id=None, model=request.model, message=Message.assistant(f"echo:{self.tag}"),
                        finish_reason="stop", usage=Usage())

    def dump_state(self):
        return {"tag": self.tag}

    @classmethod
    def load_state(cls, state):
        return cls(**state)


class AsyncEcho(Echo):
    async def complete(self, request):
        return Echo.complete(self, request)


class CustomLM(dspy.BaseLM):
    """A custom LM relying on the default reconstruction contract."""


class BadState(Echo):
    """An engine whose state is not a dict."""

    def dump_state(self):
        return ["tag"]


class NonJsonState(Echo):
    """An engine whose state is a dict that JSON cannot carry."""

    def dump_state(self):
        return {"tag": {1, 2}}


class Stateless:
    """An engine with no save/load protocol."""

    def complete(self, request):
        return Echo().complete(request)


# ─── connection settings belong to the engine ─────────────────────────


@pytest.mark.parametrize("setting", [
    {"api_key": "k"}, {"api_base": "http://127.0.0.1:9/v1"}, {"base_url": "http://127.0.0.1:9/v1"},
    {"timeout": 1}, {"extra_headers": {"X-Test": "1"}}, {"headers": {"X-Test": "1"}}, {"api_version": "v1"},
    {"organization": "org"}, {"project": "p"}, {"extra_query": {"a": "b"}}, {"custom_llm_provider": "x"},
])
@pytest.mark.parametrize("lm_class", [CustomLM, dspy.LM])
@pytest.mark.parametrize("with_sync", [False, True])
@pytest.mark.parametrize("on_copy", [False, True])
def test_client_settings_are_refused_with_a_custom_engine(setting, lm_class, with_sync, on_copy):
    with pytest.raises(ValueError, match="custom engine owns its own"):
        if on_copy:
            lm = lm_class("custom/echo", engine=Echo() if with_sync else None, async_engine=AsyncEcho())
            lm.copy(**setting)
        else:
            lm_class("custom/echo", engine=Echo() if with_sync else None, async_engine=AsyncEcho(), **setting)


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [{"api_key": "k"}, {"api_base": "http://127.0.0.1:9/v1"}, {"timeout": 0.001},
                                     {"extra_headers": {"X-Test": "1"}}])
async def test_client_settings_are_refused_on_every_call(setting):
    # The constructor and copy() refused them; a call passed them in and the
    # engine never saw them (greptile on dspy#10441 found the third door open).
    lm = dspy.LM("custom/echo", engine=Echo(), async_engine=AsyncEcho(), cache=True)
    refused = re.escape(f"LM call: {sorted(setting)}")
    with pytest.raises(ValueError, match=refused):
        lm("hi", **setting)
    with pytest.raises(ValueError, match=refused):
        await lm.acall("hi", **setting)
    with pytest.raises(ValueError, match=refused):
        lm(Request(model=lm.model, messages=(Message.user("hi"),)), **setting)
    with pytest.raises(ValueError, match=refused):
        await lm.acall(Request(model=lm.model, messages=(Message.user("hi"),)), **setting)
    assert not lm.history  # refused before any attempt or cache entry
    assert lm("hi") == ["echo:t"]


def test_client_settings_are_refused_on_copy_too():
    lm = dspy.LM("custom/echo", engine=Echo(), cache=False)
    with pytest.raises(ValueError, match=r"LM.copy: \['api_key'\]"):
        lm.copy(api_key="k")
    # And when a copy switches to a custom engine while the LM carries them.
    native = dspy.LM("openai/gpt-4o", api_key="k", cache=False)
    with pytest.raises(ValueError, match="api_key"):
        native.copy(engine=Echo())
    # Clearing them in the same copy is the way through.
    switched = native.copy(engine=Echo(), api_key=None)
    assert "api_key" not in switched.kwargs and isinstance(switched.engine, Echo)


@pytest.mark.parametrize("lm_class", [CustomLM, dspy.LM])
def test_none_is_not_a_setting(lm_class):
    lm = lm_class("custom/echo", engine=Echo(), api_key=None, timeout=None, cache=False)
    assert lm("hi") == ["echo:t"]
    assert lm.copy(api_key=None, timeout=None)("hi") == ["echo:t"]


def test_built_in_engines_keep_taking_client_settings():
    lm = dspy.LM("openai/gpt-4o", engine="lm15", api_key="k", api_base="http://127.0.0.1:9/v1", timeout=3)
    assert lm.kwargs["api_key"] == "k"
    assert lm.copy(api_key="k2").kwargs["api_key"] == "k2"


# ─── the engine pair is one unit ──────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("async_first", [False, True])
async def test_sync_and_async_engines_keep_separate_cached_answers(typed, async_first):
    sync_engine, async_engine = Echo("sync"), AsyncEcho("async")
    lm = dspy.LM("custom/echo", engine=sync_engine, async_engine=async_engine)
    prompt = Request(model=lm.model, messages=(Message.user("hi"),)) if typed else "hi"
    for _ in range(2):
        for asynchronous in (async_first, not async_first):
            result = await lm.acall(prompt) if asynchronous else lm(prompt)
            assert (result.text if typed else result[0]) == ("echo:async" if asynchronous else "echo:sync")
        # Both modes must reuse their own cached answer, not call either engine again.
        sync_engine.tag = async_engine.tag = "changed"


def test_constructor_and_copy_apply_the_same_async_rule():
    with pytest.raises(ValueError, match="async_engine is only used with a custom engine"):
        dspy.LM("custom/echo", engine="auto", async_engine=AsyncEcho())
    lm = dspy.LM("custom/echo", engine=Echo(), async_engine=AsyncEcho(), cache=False)
    with pytest.raises(ValueError, match="async_engine is only used with a custom engine"):
        lm.copy(engine="litellm", async_engine=AsyncEcho())


@pytest.mark.asyncio
@pytest.mark.parametrize("lm_class", [dspy.BaseLM, CustomLM, dspy.LM])
async def test_copy_with_a_new_engine_replaces_the_pair(lm_class):
    lm = lm_class("custom/echo", engine=Echo("a"), async_engine=AsyncEcho("a"), cache=False)
    if lm_class is dspy.LM:
        to_litellm = lm.copy(engine="litellm")
        assert to_litellm.engine == "litellm" and to_litellm.async_engine is None
    other = lm.copy(engine=Echo("b"))
    assert other.engine.tag == "b" and other.async_engine is None
    assert other("hi") == ["echo:b"]
    both = lm.copy(engine=Echo("c"), async_engine=AsyncEcho("d"))
    assert both.engine.tag == "c" and both.async_engine.tag == "d"
    assert both("hi") == ["echo:c"]
    assert await both.acall("hi") == ["echo:d"]
    # A copy that does not mention the engine keeps the whole pair.
    same = lm.copy(temperature=0.5)
    assert same.engine is lm.engine and same.async_engine is lm.async_engine
    assert same("hi") == ["echo:a"]
    assert await same.acall("hi") == ["echo:a"]
    async_only = lm.copy(engine=None, async_engine=AsyncEcho("async"))
    assert async_only.engine is None and async_only.async_engine.tag == "async"
    assert await async_only.acall("hi") == ["echo:async"]
    assert async_only.copy(temperature=0.5).async_engine is async_only.async_engine
    replaced_async = lm.copy(async_engine=AsyncEcho("new-async"))
    assert replaced_async.engine is lm.engine
    assert replaced_async("hi") == ["echo:a"]
    assert await replaced_async.acall("hi") == ["echo:new-async"]
    cleared_async = lm.copy(async_engine=None)
    assert cleared_async.engine is lm.engine and cleared_async.async_engine is None
    assert cleared_async("hi") == ["echo:a"]
    assert lm.engine.tag == "a" and lm.async_engine.tag == "a"
    assert lm("hi") == ["echo:a"]
    assert await lm.acall("hi") == ["echo:a"]
    assert "engine" not in both.kwargs and "async_engine" not in both.kwargs


@pytest.mark.parametrize("lm_class", [dspy.BaseLM, CustomLM])
def test_base_copy_allows_both_engines_absent(lm_class):
    empty = lm_class("custom/echo", cache=False)
    assert empty.copy().engine is None and empty.copy().async_engine is None
    lm = lm_class("custom/echo", engine=Echo(), async_engine=AsyncEcho(), cache=False)
    cleared = lm.copy(engine=None)
    assert cleared.engine is None and cleared.async_engine is None
    assert lm.engine is not None and lm.async_engine is not None
    assert lm.copy(engine=None, async_engine=AsyncEcho()).copy(async_engine=None).async_engine is None


@pytest.mark.asyncio
@pytest.mark.parametrize("lm_class", [dspy.BaseLM, dspy.LM])
@pytest.mark.parametrize("with_sync", [False, True])
async def test_custom_capabilities_prefer_sync_otherwise_async(lm_class, with_sync):
    class FalseyEcho(Echo):
        def __bool__(self):
            return False

    class AsyncFeatures(AsyncEcho):
        supports_function_calling = True
        supports_reasoning = True
        supports_response_schema = True
        supported_params = frozenset({"tools", "response_format"})

        async def complete(self, request):
            return Response(None, request.model, Message.assistant('{"answer":"async"}'), "stop", Usage())

    lm = lm_class("custom/echo", engine=FalseyEcho("sync") if with_sync else None,
                  async_engine=AsyncFeatures(), cache=False)
    # Even falsey sync engines with no capability declarations take precedence;
    # do not union the engines' hints or fall back separately for missing fields.
    assert lm.supports_function_calling is (not with_sync)
    assert lm.supports_reasoning is (not with_sync)
    assert lm.supports_response_schema is (not with_sync)
    assert lm.supported_params == (set() if with_sync else {"tools", "response_format"})
    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
        result = await dspy.Predict("question -> answer").acall(question="Which engine?")
    assert result.answer == "async"
    if with_sync:
        assert lm("hi") == ["echo:sync"]
    else:
        with pytest.raises(dspy.LMUnsupportedFeatureError, match="no engine"):
            lm("hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("lm_class", [dspy.BaseLM, dspy.LM])
async def test_async_counterpart_must_be_an_engine(lm_class):
    with pytest.raises(TypeError, match="async complete"):
        lm_class("custom/echo", engine=Echo(), async_engine=object())
    lm = lm_class("custom/echo", engine=Echo(), async_engine=AsyncEcho("z"), cache=False)
    assert lm("hi") == ["echo:t"]
    assert await lm.acall("hi") == ["echo:z"]


@pytest.mark.parametrize("lm_class", [dspy.BaseLM, dspy.LM])
def test_each_side_of_the_pair_must_be_the_right_kind(lm_class):
    # A sync complete() as the async engine would be awaited as a Response
    # on the first acall; an async complete() as the sync engine would hand
    # execute() a coroutine. Both are refused at construction (greptile on
    # dspy#10441).
    with pytest.raises(TypeError, match=r"async_engine\.complete is not a coroutine function"):
        lm_class("custom/echo", engine=Echo(), async_engine=Echo())
    with pytest.raises(TypeError, match=r"engine\.complete is a coroutine function"):
        lm_class("custom/echo", engine=AsyncEcho())


@pytest.mark.parametrize("lm_class", [dspy.BaseLM, CustomLM, dspy.LM])
def test_copy_checks_each_side_of_the_pair(lm_class):
    lm = lm_class("custom/echo", engine=Echo(), async_engine=AsyncEcho(), cache=False)
    with pytest.raises(TypeError, match="not a coroutine function"):
        lm.copy(async_engine=Echo())
    with pytest.raises(TypeError, match="is a coroutine function"):
        lm.copy(engine=AsyncEcho())
    with pytest.raises(TypeError, match="must implement complete"):
        lm.copy(engine=object())
    with pytest.raises(TypeError, match="must implement async complete"):
        lm.copy(async_engine=object())
    assert lm("hi") == ["echo:t"]


class AsyncCallable:
    async def __call__(self, request):
        return Echo("inherited").complete(request)


class InheritsAsyncCall(AsyncCallable):
    pass


class WithCallableComplete:
    complete = InheritsAsyncCall()


@pytest.mark.asyncio
@pytest.mark.parametrize("lm_class", [dspy.BaseLM, dspy.LM])
async def test_an_inherited_async_call_counts_as_a_coroutine_function(lm_class):
    # Python resolves __call__ through the MRO; so does the kind check.
    lm = lm_class("custom/echo", engine=Echo(), async_engine=WithCallableComplete(), cache=False)
    assert await lm.acall("hi") == ["echo:inherited"]


# ─── saving and loading ───────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("lm_class", [CustomLM, dspy.LM])
@pytest.mark.parametrize("with_sync", [False, True])
async def test_custom_engine_round_trips_through_json_state(with_sync, lm_class):
    lm = lm_class("custom/echo", engine=Echo("a") if with_sync else None,
                 async_engine=AsyncEcho("b"), cache=False, num_retries=0)
    state = lm.dump_state()
    json.dumps(state)  # JSON-serializable
    assert state["engine"] == ({"class": f"{__name__}:Echo", "state": {"tag": "a"}} if with_sync else None)
    assert state["async_engine"] == {"class": f"{__name__}:AsyncEcho", "state": {"tag": "b"}}
    with pytest.raises(ValueError, match="Refusing to import custom serialized engine class"):
        lm_class.load_state(state)
    loaded = lm_class.load_state(state, allow_custom_lm_class=True)
    if with_sync:
        assert isinstance(loaded.engine, Echo) and loaded.engine.tag == "a"
        assert loaded("hi") == ["echo:a"]
    else:
        assert loaded.engine is None
    assert isinstance(loaded._async_engine_spec, AsyncEcho) and loaded._async_engine_spec.tag == "b"
    assert loaded.num_retries == 0
    assert await loaded.acall("hi") == ["echo:b"]


def test_loading_a_custom_engine_is_gated_like_a_custom_lm_class():
    state = dspy.LM("custom/echo", engine=Echo()).dump_state()
    with pytest.raises(ValueError, match="Refusing to import custom serialized engine class"):
        dspy.LM.load_state(state)
    with pytest.raises(ValueError, match="Refusing to import custom serialized engine class"):
        dspy.BaseLM.load_state(state)


@pytest.mark.asyncio
@pytest.mark.parametrize("lm_class", [dspy.BaseLM, CustomLM, dspy.LM])
@pytest.mark.parametrize("with_sync", [False, True])
async def test_program_save_and_load_with_a_custom_engine(tmp_path, lm_class, with_sync):
    program = dspy.Predict("question -> answer")
    program.lm = lm_class("custom/echo", engine=Echo("saved") if with_sync else None,
                          async_engine=AsyncEcho("async-saved"), cache=False)
    path = tmp_path / "program.json"
    program.save(path)
    fresh = dspy.Predict("question -> answer")
    with pytest.raises(ValueError, match="allow_unsafe_lm_state"):
        fresh.load(path)
    fresh.load(path, allow_unsafe_lm_state=True)
    assert type(fresh.lm) is lm_class
    if with_sync:
        assert fresh.lm("hi") == ["echo:saved"]
    else:
        assert fresh.lm.engine is None
    assert await fresh.lm.acall("hi") == ["echo:async-saved"]


@pytest.mark.parametrize("lm_class", [CustomLM, dspy.LM])
def test_engine_without_state_protocol_is_refused_with_the_way_out(lm_class):
    lm = lm_class("custom/echo", engine=Stateless())
    with pytest.raises(TypeError, match=r"dump_state\(\) -> dict.*load_state\(state\)") as info:
        lm.dump_state()
    assert "save the program without this LM" in str(info.value)


@pytest.mark.parametrize("lm_class", [CustomLM, dspy.LM])
def test_function_local_engine_class_is_refused_at_save_time(lm_class):
    class Local(Echo):
        pass

    lm = lm_class("custom/echo", engine=Local())
    with pytest.raises(TypeError, match=r"not importable as .*<locals>.*module level"):
        lm.dump_state()


@pytest.mark.parametrize("lm_class", [CustomLM, dspy.LM])
def test_engine_state_must_be_a_dict(lm_class):
    with pytest.raises(TypeError, match="must return a dict"):
        lm_class("custom/echo", engine=BadState()).dump_state()


@pytest.mark.parametrize("lm_class", [CustomLM, dspy.LM])
def test_engine_state_must_be_json_at_dump_time(lm_class):
    # Refused where the promise is made, not at file-write time.
    with pytest.raises(TypeError, match=r"JSON-serializable.*set"):
        lm_class("custom/echo", engine=NonJsonState()).dump_state()


def test_saved_state_loads_in_a_fresh_process(tmp_path):
    # The promise of a class path is that another process can follow it.
    program = dspy.Predict("question -> answer")
    program.lm = dspy.LM("custom/echo", engine=Echo("portable"), cache=False)
    path = tmp_path / "program.json"
    program.save(path)
    script = (
        "import sys, dspy\n"
        "p = dspy.Predict('question -> answer')\n"
        f"p.load({str(path)!r}, allow_unsafe_lm_state=True)\n"
        "print(type(p.lm.engine).__name__, p.lm.engine.tag)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            cwd=str(tmp_path), env={**os.environ, "PYTHONPATH": ":".join(sys.path)})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Echo portable"


def test_serialized_engine_records_are_validated():
    base = dspy.LM("custom/echo").dump_state()
    with pytest.raises(ValueError, match="selection string or a"):
        dspy.LM.load_state({**base, "engine": {"state": {}}}, allow_custom_lm_class=True)
    with pytest.raises(ValueError, match="module:QualName"):
        dspy.LM.load_state({**base, "engine": {"class": "nonsense", "state": {}}}, allow_custom_lm_class=True)
    with pytest.raises(ValueError, match="without a custom engine"):
        dspy.LM.load_state({**base, "async_engine": {"class": f"{__name__}:AsyncEcho", "state": {}}},
                           allow_custom_lm_class=True)
    with pytest.raises(ImportError, match="cannot be imported"):
        dspy.LM.load_state({**base, "engine": {"class": f"{__name__}:Missing", "state": {}}},
                           allow_custom_lm_class=True)
    with pytest.raises(TypeError, match="no load_state"):
        dspy.LM.load_state({**base, "engine": {"class": f"{__name__}:Stateless", "state": {}}},
                           allow_custom_lm_class=True)


def test_built_in_engine_selection_state_is_unchanged():
    assert "engine" not in dspy.LM("openai/gpt-4o").dump_state()
    assert dspy.LM("openai/gpt-4o", engine="litellm").dump_state()["engine"] == "litellm"
    loaded = dspy.LM.load_state(dspy.LM("openai/gpt-4o", engine="lm15").dump_state())
    assert loaded.engine == "lm15"
