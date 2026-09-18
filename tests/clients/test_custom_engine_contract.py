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
from dspy.lm15 import Message, Response, Usage


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
def test_client_settings_are_refused_with_a_custom_engine(setting):
    with pytest.raises(ValueError, match=re.escape(f"dspy.LM: {sorted(setting)}") + ".*custom engine owns its own"):
        dspy.LM("custom/echo", engine=Echo(), **setting)


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
        lm.forward("hi", **setting)
    with pytest.raises(ValueError, match=refused):
        await lm.aforward("hi", **setting)
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


def test_none_is_not_a_setting():
    lm = dspy.LM("custom/echo", engine=Echo(), api_key=None, timeout=None, cache=False)
    assert lm("hi") == ["echo:t"]


def test_built_in_engines_keep_taking_client_settings():
    lm = dspy.LM("openai/gpt-4o", engine="lm15", api_key="k", api_base="http://127.0.0.1:9/v1", timeout=3)
    assert lm.kwargs["api_key"] == "k"
    assert lm.copy(api_key="k2").kwargs["api_key"] == "k2"


# ─── the engine pair is one unit ──────────────────────────────────────


def test_constructor_and_copy_apply_the_same_async_rule():
    with pytest.raises(ValueError, match="async_engine is only used with a custom engine"):
        dspy.LM("custom/echo", engine="auto", async_engine=AsyncEcho())
    lm = dspy.LM("custom/echo", engine=Echo(), async_engine=AsyncEcho(), cache=False)
    with pytest.raises(ValueError, match="async_engine is only used with a custom engine"):
        lm.copy(engine="litellm", async_engine=AsyncEcho())


def test_copy_with_a_new_engine_replaces_the_pair():
    lm = dspy.LM("custom/echo", engine=Echo("a"), async_engine=AsyncEcho("a"), cache=False)
    to_litellm = lm.copy(engine="litellm")
    assert to_litellm.engine == "litellm" and to_litellm._async_engine_spec is None
    other = lm.copy(engine=Echo("b"))
    assert other.engine.tag == "b" and other._async_engine_spec is None
    both = lm.copy(engine=Echo("c"), async_engine=AsyncEcho("c"))
    assert both.engine.tag == "c" and both._async_engine_spec.tag == "c"
    # A copy that does not mention the engine keeps the whole pair.
    same = lm.copy(temperature=0.5)
    assert same.engine is lm.engine and same._async_engine_spec is lm._async_engine_spec


@pytest.mark.asyncio
async def test_async_counterpart_must_be_an_engine():
    with pytest.raises(TypeError, match="async complete"):
        dspy.LM("custom/echo", engine=Echo(), async_engine=object())
    lm = dspy.LM("custom/echo", engine=Echo(), async_engine=AsyncEcho("z"), cache=False)
    assert await lm.acall("hi") == ["echo:z"]


def test_each_side_of_the_pair_must_be_the_right_kind():
    # A sync complete() as the async engine would be awaited as a Response
    # on the first acall; an async complete() as the sync engine would hand
    # execute() a coroutine. Both are refused at construction (greptile on
    # dspy#10441).
    with pytest.raises(TypeError, match=r"async_engine\.complete is not a coroutine function"):
        dspy.LM("custom/echo", engine=Echo(), async_engine=Echo())
    with pytest.raises(TypeError, match=r"engine\.complete is a coroutine function"):
        dspy.LM("custom/echo", engine=AsyncEcho())
    lm = dspy.LM("custom/echo", engine=Echo(), async_engine=AsyncEcho(), cache=False)
    with pytest.raises(TypeError, match="not a coroutine function"):
        lm.copy(async_engine=Echo())
    with pytest.raises(TypeError, match="is a coroutine function"):
        lm.copy(engine=AsyncEcho())


class AsyncCallable:
    async def __call__(self, request):
        return Echo("inherited").complete(request)


class InheritsAsyncCall(AsyncCallable):
    pass


class WithCallableComplete:
    complete = InheritsAsyncCall()


@pytest.mark.asyncio
async def test_an_inherited_async_call_counts_as_a_coroutine_function():
    # Python resolves __call__ through the MRO; so does the kind check.
    lm = dspy.LM("custom/echo", engine=Echo(), async_engine=WithCallableComplete(), cache=False)
    assert await lm.acall("hi") == ["echo:inherited"]


# ─── saving and loading ───────────────────────────────────────────────


def test_custom_engine_round_trips_through_json_state():
    lm = dspy.LM("custom/echo", engine=Echo("a"), async_engine=AsyncEcho("b"), cache=False, num_retries=0)
    state = lm.dump_state()
    json.dumps(state)  # JSON-serializable
    assert state["engine"] == {"class": f"{__name__}:Echo", "state": {"tag": "a"}}
    assert state["async_engine"] == {"class": f"{__name__}:AsyncEcho", "state": {"tag": "b"}}
    loaded = dspy.LM.load_state(state, allow_custom_lm_class=True)
    assert isinstance(loaded.engine, Echo) and loaded.engine.tag == "a"
    assert isinstance(loaded._async_engine_spec, AsyncEcho) and loaded._async_engine_spec.tag == "b"
    assert loaded.num_retries == 0 and loaded("hi") == ["echo:a"]


def test_loading_a_custom_engine_is_gated_like_a_custom_lm_class():
    state = dspy.LM("custom/echo", engine=Echo()).dump_state()
    with pytest.raises(ValueError, match="Refusing to import custom serialized engine class"):
        dspy.LM.load_state(state)
    with pytest.raises(ValueError, match="Refusing to import custom serialized engine class"):
        dspy.BaseLM.load_state(state)


def test_program_save_and_load_with_a_custom_engine(tmp_path):
    program = dspy.Predict("question -> answer")
    program.lm = dspy.LM("custom/echo", engine=Echo("saved"), cache=False)
    path = tmp_path / "program.json"
    program.save(path)
    fresh = dspy.Predict("question -> answer")
    with pytest.raises(ValueError, match="allow_unsafe_lm_state"):
        fresh.load(path)
    fresh.load(path, allow_unsafe_lm_state=True)
    assert isinstance(fresh.lm.engine, Echo) and fresh.lm.engine.tag == "saved"


def test_engine_without_state_protocol_is_refused_with_the_way_out():
    lm = dspy.LM("custom/echo", engine=Stateless())
    with pytest.raises(TypeError, match=r"dump_state\(\) -> dict.*load_state\(state\)") as info:
        lm.dump_state()
    assert "save the program without this LM" in str(info.value)


def test_function_local_engine_class_is_refused_at_save_time():
    class Local(Echo):
        pass

    lm = dspy.LM("custom/echo", engine=Local())
    with pytest.raises(TypeError, match=r"not importable as .*<locals>.*module level"):
        lm.dump_state()


def test_engine_state_must_be_a_dict():
    with pytest.raises(TypeError, match="must return a dict"):
        dspy.LM("custom/echo", engine=BadState()).dump_state()


def test_engine_state_must_be_json_at_dump_time():
    # Refused where the promise is made, not at file-write time.
    with pytest.raises(TypeError, match=r"JSON-serializable.*set"):
        dspy.LM("custom/echo", engine=NonJsonState()).dump_state()


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
