"""SandboxLM: the host LM as sandboxed code may call it (dspy.primitives.facade)."""

import pytest

import dspy
from dspy.primitives.facade import SANDBOX_LM_KWARGS, SandboxLM
from dspy.utils.dummies import DummyLM


class _CapableLM(DummyLM):
    """A DummyLM that reports model capabilities, like a real provider-backed dspy.LM."""

    supports_function_calling = True
    supports_reasoning = True
    supports_response_schema = True
    supported_params = frozenset({"temperature", "response_format", "tools"})


def _lm(**kwargs):
    lm = DummyLM([{"answer": "a"}, {"answer": "b"}, {"answer": "c"}])
    lm.kwargs.update(kwargs)
    return lm


def test_generation_parameters_pass_and_the_wrapped_lm_does_the_work():
    lm = _lm()
    sandbox = SandboxLM(lm)

    outputs = sandbox(messages=[{"role": "user", "content": "hi"}], temperature=0.3, max_tokens=5, n=1)

    assert outputs == ["[[ ## answer ## ]]\na"]
    assert len(lm.history) == 1
    assert sandbox.history == []  # the proxy keeps no history of its own


@pytest.mark.parametrize(
    "options",
    [
        {"api_base": "http://evil.example"},
        {"api_key": "sk-x"},
        {"extra_headers": {"Authorization": "Bearer x"}},
        {"model": "openai/gpt-5-pro"},
        {"temperature": 0.1, "base_url": "http://evil.example", "timeout": 1},
    ],
)
def test_routing_credential_and_transport_options_are_rejected_before_any_request(options):
    lm = _lm()
    sandbox = SandboxLM(lm)
    rejected = sorted(options.keys() - SANDBOX_LM_KWARGS)

    with pytest.raises(TypeError, match=rf"may not set LM option\(s\) {rejected!r}".replace("[", r"\[").replace("]", r"\]")):
        sandbox("prompt", **options)
    with pytest.raises(TypeError, match="may not set LM option"):
        sandbox.copy(**options)

    assert lm.history == []


@pytest.mark.asyncio
async def test_acall_applies_the_same_allowlist():
    lm = _lm()
    sandbox = SandboxLM(lm)

    with pytest.raises(TypeError, match=r"may not set LM option\(s\) \['api_key'\]"):
        await sandbox.acall("prompt", api_key="sk-x")
    assert lm.history == []

    outputs = await sandbox.acall("prompt", temperature=0.2)
    assert outputs == ["[[ ## answer ## ]]\na"]


def test_reserve_meters_each_admitted_call_and_a_rejected_call_costs_nothing():
    reserved = []
    sandbox = SandboxLM(_lm(), reserve=reserved.append)

    sandbox("one")
    with pytest.raises(TypeError):
        sandbox("two", api_key="sk-x")
    sandbox("three", temperature=0.5)

    assert reserved == [1, 1]


def test_reserve_failures_stop_the_call():
    lm = _lm()

    def exhausted(n):
        raise RuntimeError("LLM call limit exceeded")

    with pytest.raises(RuntimeError, match="LLM call limit exceeded"):
        SandboxLM(lm, reserve=exhausted)("prompt")
    assert lm.history == []


def test_proxy_mirrors_identity_kwargs_and_capabilities_of_the_wrapped_lm():
    # Adapters and Predict read these off dspy.settings.lm; the proxy must answer for the real model.
    lm = _CapableLM([{"answer": "a"}])
    lm.kwargs.update(temperature=0.7, api_key="sk-secret")
    sandbox = SandboxLM(lm)

    assert (sandbox.model, sandbox.model_type) == (lm.model, lm.model_type)
    assert sandbox.kwargs == lm.kwargs and sandbox.kwargs is not lm.kwargs
    assert sandbox.supports_function_calling and sandbox.supports_reasoning and sandbox.supports_response_schema
    assert sandbox.supported_params == {"temperature", "response_format", "tools"}
    assert not SandboxLM(_lm()).supports_function_calling  # DummyLM reports none


def test_copy_keeps_the_proxy_and_its_meter():
    reserved = []
    sandbox = SandboxLM(_lm(), reserve=reserved.append)

    copied = sandbox.copy(temperature=0.9, rollout_id=3)

    assert isinstance(copied, SandboxLM)
    assert copied.kwargs["temperature"] == 0.9
    copied("prompt")
    assert reserved == [1]


def test_scoped_override_reaches_predictors_and_nested_sub_llm_calls():
    # The way the facade uses it: as dspy.context(lm=...) around a bridged call, so a predictor's LM
    # call and anything nested resolve to the proxy.
    lm = _lm()
    reserved = []
    with dspy.context(lm=SandboxLM(lm, reserve=reserved.append), adapter=dspy.ChatAdapter()):
        assert dspy.Predict("question -> answer")(question="q").answer == "a"
        assert isinstance(dspy.settings.lm, SandboxLM)
    assert reserved == [1]
    assert len(lm.history) == 1
