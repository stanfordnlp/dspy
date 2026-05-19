import base64
import json
from types import SimpleNamespace

import pytest

import dspy
from dspy.clients.language_models.openai import completion_stream_to_events, responses_stream_to_events


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __iter__(self):
        for item in self.payload:
            yield f"data: {json.dumps(item)}\n".encode()
            yield b"\n"


def test_openai_completions_class_calls_chat_completions_and_maps_response():
    calls = []

    def completions(**kwargs):
        calls.append(kwargs)
        return {
            "id": "cmpl_1",
            "model": "gpt-4o-mini",
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    lm = dspy.OpenAIChatLM("openai/gpt-4o-mini", completions=completions, cache=False)

    response = lm("say hello", temperature=0.1)

    assert calls[0]["model"] == "gpt-4o-mini"
    assert calls[0]["messages"] == [{"role": "user", "content": "say hello"}]
    assert calls[0]["temperature"] == 0.1
    assert response.text == "hello"
    assert response.usage.total_tokens == 2


def test_openai_text_lm_calls_text_completions_and_maps_response():
    calls = []

    def completions(**kwargs):
        calls.append(kwargs)
        return {
            "model": "gpt-3.5-turbo-instruct",
            "choices": [{"text": "hello", "finish_reason": "stop"}],
        }

    lm = dspy.OpenAITextLM(
        "openai/gpt-3.5-turbo-instruct",
        completions=completions,
        cache=False,
    )

    response = lm("say hello")

    assert calls[0]["model"] == "gpt-3.5-turbo-instruct"
    assert calls[0]["prompt"] == "say hello\n\nBEGIN RESPONSE:"
    assert response.text == "hello"


def test_openai_responses_class_calls_responses_and_maps_response():
    calls = []

    def responses(**kwargs):
        calls.append(kwargs)
        return {
            "id": "resp_1",
            "model": "gpt-4o-mini",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "hello"}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

    lm = dspy.OpenAIResponsesLM("openai/gpt-4o-mini", responses=responses, cache=False)

    response = lm("say hello")

    assert calls[0]["model"] == "gpt-4o-mini"
    assert calls[0]["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "say hello"}]}]
    assert response.text == "hello"
    assert response.usage.total_tokens == 2


def test_openai_responses_maps_max_tokens_to_max_output_tokens_for_gpt_5_reasoning_models():
    calls = []

    def responses(**kwargs):
        calls.append(kwargs)
        return {
            "model": "gpt-5.2",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "hello"}]}],
        }

    dspy.OpenAIResponsesLM("openai/gpt-5.2", responses=responses, cache=False)("say hello", max_tokens=80)

    assert calls[0]["max_output_tokens"] == 80
    assert "max_tokens" not in calls[0]
    assert "max_completion_tokens" not in calls[0]


def test_openai_chat_maps_max_tokens_to_max_completion_tokens_for_gpt_5_reasoning_models():
    calls = []

    def completions(**kwargs):
        calls.append(kwargs)
        return {"model": "gpt-5.2", "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}

    dspy.OpenAIChatLM("openai/gpt-5.2", completions=completions, cache=False)("say hello", max_tokens=80)

    assert calls[0]["max_completion_tokens"] == 80
    assert "max_tokens" not in calls[0]


def test_openai_chat_uses_max_tokens_for_gpt_5_chat_models():
    calls = []

    def completions(**kwargs):
        calls.append(kwargs)
        return {"model": "gpt-5.2-chat-latest", "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}

    dspy.OpenAIChatLM("openai/gpt-5.2-chat-latest", completions=completions, cache=False)(
        "say hello", max_tokens=80
    )

    assert calls[0]["max_tokens"] == 80
    assert "max_completion_tokens" not in calls[0]


def test_openai_reasoning_models_reject_custom_temperature_when_effort_is_active():
    lm = dspy.OpenAIResponsesLM("openai/gpt-5.2", responses=lambda **kwargs: {}, cache=False)

    with pytest.raises(dspy.LMUnsupportedFeatureError, match="default temperature"):
        lm("say hello", temperature=0, reasoning_effort="low")


def test_openai_reasoning_models_allow_temperature_zero_when_effort_is_none():
    calls = []

    def responses(**kwargs):
        calls.append(kwargs)
        return {
            "model": "gpt-5.2",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "hello"}]}],
        }

    dspy.OpenAIResponsesLM("openai/gpt-5.2", responses=responses, cache=False)(
        "say hello", temperature=0, reasoning_effort="none"
    )

    assert calls[0]["temperature"] == 0
    assert calls[0]["reasoning"] == {"effort": "none"}


def test_openai_format_reads_local_image_paths_as_data_uris(tmp_path):
    image_path = tmp_path / "cat.jpg"
    image_path.write_bytes(b"fake jpg")
    calls = []

    def completions(**kwargs):
        calls.append(kwargs)
        return {"model": "gpt-4o-mini", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    dspy.OpenAIChatLM("openai/gpt-4o-mini", completions=completions, cache=False)(
        dspy.User("describe", dspy.lm.LMImagePart(path=str(image_path)))
    )

    image_url = calls[0]["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url == f"data:image/jpeg;base64,{base64.b64encode(b'fake jpg').decode('ascii')}"
    assert str(image_path) not in json.dumps(calls[0])


def test_openai_format_reads_local_file_paths_as_data_uris(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello file")
    calls = []

    def responses(**kwargs):
        calls.append(kwargs)
        return {
            "model": "gpt-4o-mini",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
        }

    dspy.OpenAIResponsesLM("openai/gpt-4o-mini", responses=responses, cache=False)(
        dspy.User("read", dspy.lm.LMFilePart(path=str(file_path)))
    )

    file_block = calls[0]["input"][0]["content"][1]
    assert file_block["file_data"] == f"data:text/plain;base64,{base64.b64encode(b'hello file').decode('ascii')}"
    assert file_block["filename"] == "notes.txt"
    assert str(file_path) not in json.dumps(calls[0])


def test_openai_format_reads_local_audio_paths_as_base64(tmp_path):
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"fake wav")
    calls = []

    def completions(**kwargs):
        calls.append(kwargs)
        return {"model": "gpt-4o-mini", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    dspy.OpenAIChatLM("openai/gpt-4o-mini", completions=completions, cache=False)(
        dspy.User("transcribe", dspy.lm.LMAudioPart(path=str(audio_path)))
    )

    audio = calls[0]["messages"][0]["content"][1]["input_audio"]
    assert audio == {"data": base64.b64encode(b"fake wav").decode("ascii"), "format": "wav"}
    assert str(audio_path) not in json.dumps(calls[0])


def test_openai_reasoning_config_does_not_forward_unsupported_budget_or_chat_summary_fields():
    chat_calls = []
    responses_calls = []

    def completions(**kwargs):
        chat_calls.append(kwargs)
        return {"model": "gpt-5.2", "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}

    def responses(**kwargs):
        responses_calls.append(kwargs)
        return {
            "model": "gpt-5.2",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "hello"}]}],
        }

    reasoning = dspy.lm.LMReasoningConfig(effort="none", max_tokens=123, summary="auto")
    dspy.OpenAIChatLM("openai/gpt-5.2", completions=completions, cache=False)("say hello", reasoning=reasoning)
    dspy.OpenAIResponsesLM("openai/gpt-5.2", responses=responses, cache=False)("say hello", reasoning=reasoning)

    assert chat_calls[0]["reasoning_effort"] == "none"
    assert "thinking_budget" not in chat_calls[0]
    assert "reasoning_summary" not in chat_calls[0]
    assert responses_calls[0]["reasoning"] == {"effort": "none", "summary": "auto"}
    assert "max_tokens" not in responses_calls[0]["reasoning"]


def test_openai_backends_do_not_forward_dspy_rollout_id_to_provider():
    chat_calls = []
    text_calls = []
    responses_calls = []

    def chat_completions(**kwargs):
        chat_calls.append(kwargs)
        return {"model": "gpt-4o-mini", "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}

    def text_completions(**kwargs):
        text_calls.append(kwargs)
        return {"model": "gpt-3.5-turbo-instruct", "choices": [{"text": "hello", "finish_reason": "stop"}]}

    def responses(**kwargs):
        responses_calls.append(kwargs)
        return {
            "model": "gpt-4o-mini",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "hello"}]}],
        }

    dspy.OpenAIChatLM("openai/gpt-4o-mini", completions=chat_completions, cache=False)(
        "say hello", rollout_id="rollout-1"
    )
    dspy.OpenAITextLM("openai/gpt-3.5-turbo-instruct", completions=text_completions, cache=False)(
        "say hello", rollout_id="rollout-1"
    )
    dspy.OpenAIResponsesLM("openai/gpt-4o-mini", responses=responses, cache=False)(
        "say hello", rollout_id="rollout-1"
    )

    assert "rollout_id" not in chat_calls[0]
    assert "rollout_id" not in text_calls[0]
    assert "rollout_id" not in responses_calls[0]


def test_openai_completions_class_calls_openai_compatible_endpoint_directly(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return FakeHTTPResponse(
            {
                "id": "cmpl_1",
                "model": "local-model",
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    lm = dspy.OpenAIChatLM("local-model", api_key="local", api_base="http://localhost:8000/v1", cache=False)

    response = lm("say hello")

    request, timeout = calls[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "http://localhost:8000/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer local"
    assert timeout == 60
    assert payload["model"] == "local-model"
    assert response.text == "hello"


def test_openai_responses_class_calls_openai_compatible_endpoint_directly(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return FakeHTTPResponse(
            {
                "id": "resp_1",
                "model": "local-model",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "hello"}]}],
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    lm = dspy.OpenAIResponsesLM("local-model", api_key="local", api_base="http://localhost:8000/v1", cache=False)

    response = lm("say hello")

    request, timeout = calls[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "http://localhost:8000/v1/responses"
    assert timeout == 60
    assert payload["model"] == "local-model"
    assert response.text == "hello"


@pytest.mark.asyncio
async def test_openai_async_call_and_stream_use_anyio_thread_bridge():
    def completions(**kwargs):
        if kwargs.get("stream"):
            return [
                {"choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
        return {"model": "gpt-4o-mini", "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}

    lm = dspy.OpenAIChatLM("openai/gpt-4o-mini", completions=completions, cache=False)

    response = await lm.acall("say hello")
    stream = lm.astream("say hello")
    events = [event async for event in stream]

    assert response.text == "hello"
    assert events[-1].type == "end"
    assert stream.result().text == "hello"


def test_completion_stream_to_events_builds_response():
    stream = iter(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(index=0, delta=SimpleNamespace(content="hel"), finish_reason=None)]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(index=0, delta=SimpleNamespace(content="lo"), finish_reason="stop")]
            ),
        ]
    )

    builder = dspy.lm.LMOutputBuilder()
    for event in completion_stream_to_events(stream, model="gpt-4o-mini"):
        response = builder.apply(event)

    assert response.text == "hello"
    assert response.output.finish_reason == "stop"


def test_responses_stream_to_events_builds_response_from_completed_event():
    completed_response = {
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "hello"}],
            }
        ],
    }
    stream = iter([{"type": "response.completed", "response": completed_response}])

    builder = dspy.lm.LMOutputBuilder()
    for event in responses_stream_to_events(stream, model="gpt-4o-mini"):
        response = builder.apply(event)

    assert response.text == "hello"
