"""Wire contracts shared by SDK inference and training export."""

import pytest

from dspy.clients.lm15_boundary import request_kwargs, response_value
from dspy.lm15 import AudioPart, Config, DocumentPart, ImagePart, Message, Request, TextPart, UnsupportedFeatureError


@pytest.mark.parametrize("count", [0, 5])
def test_text_export_logprobs_round_trip(count):
    # This private export utility remains available; built-in LM text mode does not.
    request = Request(model="openai/test", messages=(Message.user("hello"),), config=Config(logprobs=count))
    wire = request_kwargs(request, "text")
    assert type(wire["logprobs"]) is int
    assert wire["logprobs"] == count
    assert "top_logprobs" not in wire


def test_text_completion_scores_preserve_unknown_tokens_and_offsets():
    request = Request(model="openai/test", messages=(Message.user("hello"),))
    scores = {"tokens": ["hello", " world"], "token_logprobs": [None, -0.25],
              "top_logprobs": [None, {" world": -0.25, " there": -1.5}], "text_offset": [0, 5]}
    response = response_value({"choices": [{"text": "hello world", "finish_reason": "stop", "logprobs": scores}]},
                              "text", request)
    assert response.text == "hello world"
    assert [(token.token, token.logprob) for token in response.logprobs] == [(" world", -0.25)]
    assert [(top.token, top.logprob) for top in response.logprobs[0].top] == [(" world", -0.25), (" there", -1.5)]
    assert not response.logprobs_complete
    assert response.provider_data["choices"][0]["logprobs"] == scores


def test_text_completion_export_preserves_the_output_token_limit():
    request = Request(model="openai/test", messages=(Message.user("hello"),), config=Config(max_tokens=23))
    assert request_kwargs(request, "text") == {"prompt": "hello\n\nBEGIN RESPONSE:", "max_tokens": 23}
    assert request_kwargs(request, "chat")["max_completion_tokens"] == 23


@pytest.mark.parametrize("audio_type,format_", [("audio/wav", "wav"), ("audio/mpeg", "mp3")])
def test_chat_media_preserves_sources_and_part_order(audio_type, format_):
    request = Request(model="openai/test", system="system", messages=(
        Message.user((TextPart(text="first"), ImagePart(url="https://example.com/image.png"),
                      AudioPart(data="YXVkaW8=", media_type=audio_type),
                      DocumentPart(data="cGRm", media_type="application/pdf"),
                      TextPart(text="last"))),
        Message.assistant("seen"),
        Message.user((DocumentPart(file_id="file-123"),)),
    ))
    assert request_kwargs(request, "chat")["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": [
            {"type": "text", "text": "first"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            {"type": "input_audio", "input_audio": {"data": "YXVkaW8=", "format": format_}},
            {"type": "file", "file": {"file_data": "data:application/pdf;base64,cGRm", "filename": "file.pdf"}},
            {"type": "text", "text": "last"},
        ]},
        {"role": "assistant", "content": "seen"},
        {"role": "user", "content": [{"type": "file", "file": {"file_id": "file-123"}}]},
    ]


@pytest.mark.parametrize("part", [AudioPart(url="https://example.com/audio.wav"), AudioPart(file_id="audio-123"),
                                  DocumentPart(url="https://example.com/file.pdf")])
def test_chat_media_does_not_fetch_unrepresentable_sources(part):
    request = Request(model="openai/test", messages=(Message.user((part,)),))
    with pytest.raises(UnsupportedFeatureError):
        request_kwargs(request, "chat")


@pytest.mark.parametrize("part", [AudioPart(data="YXVkaW8=", media_type="audio/wav"),
                                  DocumentPart(file_id="file-123"), ImagePart(url="https://example.com/image.png")])
@pytest.mark.parametrize("mixed", [False, True])
def test_text_completion_export_does_not_discard_media(part, mixed):
    parts = (TextPart(text="Describe this:"), part) if mixed else (part,)
    request = Request(model="openai/test", messages=(Message.user(parts),))
    with pytest.raises(UnsupportedFeatureError):
        request_kwargs(request, "text")
