"""The DSPy 3.5 LM call contract: one string convenience, otherwise lm15 objects."""

import pytest

import dspy
from dspy.lm15 import Config, Message, Request, Response
from tests.test_utils.engines import make_response, recording_lm


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_string_prompt_is_one_user_message_with_lm_defaults(asynchronous):
    lm = recording_lm(["pong"], temperature=0.3, max_tokens=50)

    outputs = await lm.acall("ping") if asynchronous else lm("ping")

    assert outputs == ["pong"]
    [request] = lm.engine.requests
    assert request.messages == (Message.user("ping"),)
    assert request.system is None
    assert request.config.temperature == 0.3
    assert request.config.max_tokens == 50
    assert lm.history[-1]["prompt"] == "ping"
    assert lm.history[-1]["outputs"] == ["pong"]
    assert isinstance(lm.history[-1]["response"], Response)


def test_string_prompt_accepts_generation_overrides_and_candidates():
    lm = recording_lm(["a", "b"], temperature=0.0)

    outputs = lm("ping", temperature=0.9, n=2, stop=["END"])

    assert outputs == ["a", "b"]
    assert [r.config.temperature for r in lm.engine.requests] == [0.9, 0.9]
    assert lm.engine.requests[0].config.stop == ("END",)
    assert len(lm.history) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_request_returns_one_response(asynchronous):
    lm = recording_lm([make_response("hello", thinking="thought")])
    request = Request(model=lm.model, system="Be brief.", messages=(Message.user("hi"),), config=Config(max_tokens=5))

    response = await lm.acall(request) if asynchronous else lm(request)

    assert isinstance(response, Response)
    assert response.text == "hello"
    assert lm.engine.requests == [request]
    assert lm.history[-1]["request"] is request
    assert lm.history[-1]["messages"][0] == {"role": "system", "content": "Be brief."}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_generate_returns_n_responses_from_separate_requests(asynchronous):
    lm = recording_lm(["one", "two", "three"])
    request = Request(model=lm.model, messages=(Message.user("count"),))

    responses = await lm.agenerate(request, n=3) if asynchronous else lm.generate(request, n=3)

    assert [r.text for r in responses] == ["one", "two", "three"]
    assert len(lm.engine.requests) == 3
    assert len(lm.history) == 1
    assert lm.history[-1]["outputs"] == ["one", "two", "three"]


def test_openai_style_messages_are_refused_with_a_migration_hint():
    lm = recording_lm()
    with pytest.raises(TypeError, match=r"lm\(messages=\[\.\.\.\]\) was removed in DSPy 3\.5"):
        lm(messages=[{"role": "user", "content": "hi"}])
    assert lm.engine.requests == []


def test_request_calls_take_no_generation_options():
    lm = recording_lm()
    request = Request(model=lm.model, messages=(Message.user("hi"),))
    with pytest.raises(TypeError, match=r"Request\.config"):
        lm(request, temperature=0.5)
    with pytest.raises(ValueError, match="Request.model must match"):
        lm(Request(model="other", messages=(Message.user("hi"),)))
    with pytest.raises(TypeError, match="Request or a prompt string"):
        lm(123)


def test_forward_subclasses_are_no_longer_an_integration_point():
    class OldStyle(dspy.BaseLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            return {"choices": [{"message": {"content": "x"}}]}

    lm = OldStyle("old")
    with pytest.raises(dspy.LMUnsupportedFeatureError, match="engine"):
        lm("hi")


def test_removed_legacy_names_are_gone():
    import dspy.clients.engines as engines

    assert not hasattr(engines, "LegacyEngine")
    assert not hasattr(dspy.BaseLM, "forward")
    assert not hasattr(dspy.LM, "forward")
    assert not hasattr(dspy.Tool, "format_as_litellm_function_call")


def test_engine_capabilities_reach_the_lm():
    lm = recording_lm()
    assert lm.supports_function_calling is True
    assert "tools" in lm.supported_params
    plain = dspy.BaseLM("m", engine=type("E", (), {"complete": lambda self, r: None})())
    assert plain.supports_function_calling is False
    assert plain.supported_params == set()


def test_audio_on_a_chat_completions_engine_fails_before_the_wire():
    """The bundled lm15 chat writer carries text and images only; the refusal is loud, not silent."""
    lm = dspy.LM("openai/gpt-4o-mini", engine="litellm", cache=False, num_retries=0)
    with pytest.raises(dspy.LMUnsupportedFeatureError, match="audio part"):
        dspy.Predict("audio: dspy.Audio -> transcript")(audio=dspy.Audio(data="QUJD", audio_format="wav"), lm=lm)
