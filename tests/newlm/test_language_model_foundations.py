import pytest

import dspy


class TextOnlyLM(dspy.LanguageModel):
    def __init__(self):
        super().__init__(model="test/text-only", cache=False)
        self.forward_called = False
        self.requests = []

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        self.forward_called = True
        self.requests.append(request)
        return dspy.LMResponse.from_text("ok", model=request.model)


class StreamingLM(TextOnlyLM):
    def forward_stream(self, request):
        yield dspy.LMStreamEndEvent(response=dspy.LMResponse.from_text("ok", model=request.model))


class NonDeepcopyableClient:
    def __deepcopy__(self, memo):
        raise RuntimeError("Provider clients should not be deep-copied")


class ClientBackedLM(dspy.LanguageModel):
    def __init__(self, client):
        super().__init__(model="test/client-backed", cache=False, temperature=0.1)
        self.client = client

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        return dspy.LMResponse.from_text("ok", model=request.model)


def test_language_model_copy_is_shallow_for_provider_resources_and_isolates_dspy_state():
    callback = object()
    client = NonDeepcopyableClient()
    lm = ClientBackedLM(client)
    lm.callbacks.append(callback)
    lm.history.append({"old": "entry"})

    copied = lm.copy(temperature=0.9, rollout_id=7)

    assert copied is not lm
    assert copied.client is client
    assert copied.history == []
    assert copied.callbacks == [callback]
    assert copied.callbacks is not lm.callbacks
    assert copied.kwargs == {"temperature": 0.9, "rollout_id": 7}
    assert lm.kwargs == {"temperature": 0.1}


def test_call_normalizes_multimodal_request_before_forward():
    lm = TextOnlyLM()

    response = lm("describe", dspy.Image("data:image/png;base64,abc"))

    assert response.text == "ok"
    assert lm.forward_called is True
    assert lm.requests[-1].messages[0].parts == [
        dspy.LMTextPart(text="describe"),
        dspy.LMImagePart(data="abc", media_type="image/png"),
    ]


def test_stream_requires_streaming_implementation():
    lm = TextOnlyLM()

    with pytest.raises(NotImplementedError) as exc_info:
        lm.stream("hello")

    assert "forward_stream" in str(exc_info.value)


def test_stream_allowed_when_forward_stream_is_overridden():
    stream = StreamingLM().stream("hello")

    assert list(stream)[-1].type == "end"
    assert stream.result().text == "ok"


def test_prompt_cache_is_separate_from_dspy_cache_config():
    lm = TextOnlyLM()
    request = lm.normalize_request("hello", cache=False, prompt_cache=True, prompt_cache_key="prefix-1")

    assert request.config.cache.enabled is False
    assert request.config.prompt_cache.enabled is True
    assert request.config.prompt_cache.key == "prefix-1"

