"""Public behavior captured before replacing the LM backend; no provider calls."""

import copy
import importlib
import json
import zipfile
from pathlib import Path

import litellm
import pytest
from pydantic import BaseModel

import dspy
from dspy.clients.cache import Cache
from dspy.utils.callback import BaseCallback
from dspy.utils.usage_tracker import track_usage

FIXTURES = Path(__file__).parent / "fixtures" / "lm_3_3_0"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
CASES = MANIFEST["cases"]


def plain(value):
    # Read fields, not SDK serializers: older pickled LiteLLM models can carry
    # deferred Pydantic serializers even though their public fields are usable.
    if isinstance(value, BaseModel):
        value = dict(value)
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


class Trace(BaseCallback):
    def __init__(self):
        self.events = []
        self.errors = []

    def on_lm_start(self, call_id, instance, inputs):
        self.events.append(("start", call_id))

    def on_lm_end(self, call_id, outputs, exception):
        self.events.append(("end", call_id))
        self.errors.append(exception)


def response_for(case):
    response = litellm.ModelResponse(**copy.deepcopy(case["provider_response"]))
    response._hidden_params["response_cost"] = 0.01
    return response


@pytest.fixture
def backend(monkeypatch):
    """Only replace the transport boundary, never LM.forward or the cache."""
    calls = []
    state = {"case": CASES[0], "forbid": False}

    def complete(**kwargs):
        if state["forbid"]:
            pytest.fail("A saved cache hit must not call the provider")
        calls.append(copy.deepcopy(kwargs))
        return response_for(state["case"])

    async def acomplete(**kwargs):
        return complete(**kwargs)

    module = importlib.import_module("dspy.clients.lm")
    provider = module._get_litellm()
    monkeypatch.setattr(provider, "completion", complete)
    monkeypatch.setattr(provider, "acompletion", acomplete)
    return calls, state


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("saved", [False, True], ids=["cold", "3.3-disk-hit"])
async def test_calls_match_3_3_outputs_keys_and_bookkeeping(case, saved, backend, monkeypatch, tmp_path):
    calls, state = backend
    state.update(case=case, forbid=saved)
    if saved:
        # Trusted, repository-owned fixtures only. Extract into a disposable
        # directory: reading a disk cache updates SQLite metadata.
        with zipfile.ZipFile(FIXTURES / f"{case['name']}.zip") as archive:
            archive.extractall(tmp_path)
    cache = Cache(True, not saved, str(tmp_path))
    monkeypatch.setattr(dspy, "cache", cache)
    trace = Trace()
    lm = dspy.LM(engine="litellm", **case["init"], callbacks=[trace])
    try:
        with track_usage() as usage:
            outputs = lm(**case["call"]) if case["mode"] == "sync" else await lm.acall(**case["call"])
        expected = "warm" if saved else "cold"
        assert plain(outputs) == case["outputs"]
        assert usage.get_total_tokens() == case[f"{expected}_usage"]
        assert len(lm.history) == 1
        history = lm.history[0]
        assert plain({key: history[key] for key in MANIFEST["history_keys"]}) == case[f"{expected}_history"]
        assert history["response"] is not None
        assert history["timestamp"] and history["uuid"]
        assert len(calls) == (0 if saved else 1)
        assert trace.events[0][0] == "start" and trace.events[1][0] == "end"
        assert len(trace.events) == 2 and trace.events[0][1] == trace.events[1][1]
        assert trace.errors == [None]
        if not saved:
            assert list(cache.memory_cache) == [case["cache_key"]]
        if case["name"].startswith("rich_n"):
            assert len(outputs) == 2
            assert outputs[0]["tool_calls"][0].function.name == "weather"
    finally:
        cache.disk_cache.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_cache_controls_and_usage_are_per_call(mode, backend):
    calls, _ = backend
    trace = Trace()
    lm = dspy.LM("openai/gpt-4o-mini", engine="litellm", temperature=0.7, max_tokens=32, callbacks=[trace])

    async def invoke(**kwargs):
        return lm("Say hello.", **kwargs) if mode == "sync" else await lm.acall("Say hello.", **kwargs)

    with track_usage() as usage:
        first = await invoke()
        assert await invoke() == first
        assert len(calls) == 1
        await invoke(cache=False)
        assert len(calls) == 2
        await invoke()
        assert len(calls) == 2
        await invoke(rollout_id=11)
        await invoke(rollout_id=11)
        assert len(calls) == 3
        await invoke(rollout_id=12)
        assert len(calls) == 4
    assert usage.get_total_tokens()[lm.model]["total_tokens"] == 4 * 20
    assert len(lm.history) == 7
    assert len(trace.events) == 14
    assert trace.errors == [None] * 7
    for start, end in zip(trace.events[::2], trace.events[1::2]):
        assert start[0] == "start" and end == ("end", start[1])
    assert all("rollout_id" not in call for call in calls)


@pytest.mark.parametrize("override", [{"n": 2}, {"temperature": 0.8}, {"max_tokens": 33}])
def test_generation_changes_do_not_reuse_the_original_cache_entry(override, backend):
    calls, _ = backend
    lm = dspy.LM("openai/gpt-4o-mini", engine="litellm", temperature=0.7, max_tokens=32)
    lm("Say hello.")
    lm("Say hello.", **override)
    lm("Say hello.", **override)
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_legacy_plugin_works_through_chat_adapter(mode):
    class LegacyLM(dspy.BaseLM):
        forward_contract = "legacy"

        def forward(self, prompt=None, messages=None, **kwargs):
            self.received = (prompt, messages, kwargs)
            return litellm.ModelResponse(
                model=self.model,
                choices=[{"message": {"role": "assistant", "content":
                          "[[ ## answer ## ]]\nyes\n\n[[ ## completed ## ]]"}, "finish_reason": "stop"}],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )

        async def aforward(self, prompt=None, messages=None, **kwargs):
            return self.forward(prompt, messages, **kwargs)

    lm = LegacyLM("legacy-test")
    adapter = dspy.ChatAdapter()
    kwargs = dict(lm=lm, lm_kwargs={"temperature": 0.4}, signature=dspy.Signature("question -> answer"),
                  demos=[], inputs={"question": "Can you answer?"})
    result = adapter(**kwargs) if mode == "sync" else await adapter.acall(**kwargs)
    assert result == [{"answer": "yes"}]
    prompt, messages, parameters = lm.received
    assert prompt is None
    assert messages[-1]["role"] == "user" and "Can you answer?" in messages[-1]["content"]
    assert parameters["temperature"] == 0.4
    assert len(lm.history) == 1
