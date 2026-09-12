"""SandboxLM: the host LM as sandboxed code may call it."""

import pytest

from dspy.primitives.facade import SANDBOX_LM_KWARGS, SandboxLM
from dspy.utils.dummies import DummyLM


class _CapableLM(DummyLM):
    supports_function_calling = True
    supports_response_schema = True
    supported_params = frozenset({"temperature", "response_format", "tools"})


def _lm():
    return DummyLM([{"answer": "a"}, {"answer": "b"}])


def test_generation_parameters_pass_and_the_wrapped_lm_does_the_work():
    lm = _lm()
    outputs = SandboxLM(lm)(messages=[{"role": "user", "content": "hi"}], temperature=0.3, max_tokens=5)
    assert outputs == ["[[ ## answer ## ]]\na"]
    assert len(lm.history) == 1


@pytest.mark.parametrize(
    "options",
    [{"api_base": "http://evil.example"}, {"api_key": "sk-x"}, {"extra_headers": {"X": "y"}}, {"model": "gpt-5-pro"}],
)
def test_routing_credential_and_transport_options_are_rejected_before_any_request(options):
    lm = _lm()
    sandbox = SandboxLM(lm)
    with pytest.raises(TypeError, match="may not set LM option"):
        sandbox("prompt", temperature=0.1, **options)
    with pytest.raises(TypeError, match="may not set LM option"):
        sandbox.copy(**options)
    assert lm.history == []
    assert not options.keys() & SANDBOX_LM_KWARGS


@pytest.mark.asyncio
async def test_acall_applies_the_same_allowlist():
    lm = _lm()
    with pytest.raises(TypeError, match=r"may not set LM option\(s\) \['api_key'\]"):
        await SandboxLM(lm).acall("prompt", api_key="sk-x")
    assert lm.history == []
    assert await SandboxLM(lm).acall("prompt", temperature=0.2) == ["[[ ## answer ## ]]\na"]


def test_reserve_meters_admitted_calls_only():
    reserved = []
    sandbox = SandboxLM(_lm(), reserve=reserved.append)
    sandbox("one")
    with pytest.raises(TypeError):
        sandbox("two", api_key="sk-x")
    sandbox.copy(temperature=0.9)("three")
    sandbox.forward("four")
    with pytest.raises(TypeError):
        sandbox.forward("five", api_base="http://evil.example")
    assert reserved == [1, 1, 1]


def test_proxy_mirrors_identity_kwargs_and_capabilities_of_the_wrapped_lm():
    lm = _CapableLM([{"answer": "a"}])
    lm.kwargs.update(temperature=0.7, api_key="sk-secret", api_base="http://internal.example")
    sandbox = SandboxLM(lm)
    assert (sandbox.model, sandbox.model_type) == (lm.model, lm.model_type)
    assert sandbox.kwargs["temperature"] == 0.7
    assert not {"api_key", "api_base"} & sandbox.kwargs.keys()  # credentials stay on the wrapped LM
    assert sandbox.supports_function_calling and sandbox.supports_response_schema and not sandbox.supports_reasoning
    assert sandbox.supported_params == {"temperature", "response_format", "tools"}
    assert not SandboxLM(_lm()).supports_function_calling


def test_wraps_a_callable_only_lm():
    # RLM accepts any callable as sub_lm; the facade is installed by default, so the proxy must too.
    class PlainLM:
        def __call__(self, prompt):
            return ["plain:" + prompt]

    sandbox = SandboxLM(PlainLM())
    assert sandbox("hi") == ["plain:hi"]
    assert (sandbox.model, sandbox.kwargs, sandbox.supports_function_calling, sandbox.supported_params) == (
        None, {}, False, set()
    )
