import json

import dspy
import dspy.clients.execution as execution
from dspy._vendor.lm15.testing import FakeResponse, FakeTransport
from dspy.lm15 import RouterConfig


def test_ogg_audio_reaches_gemini_inline_with_its_media_type(monkeypatch):
    reply = {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "[[ ## text ## ]]\nhej\n\n[[ ## completed ## ]]"}]},
                        "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
    }
    transport = FakeTransport([FakeResponse(status=200, body=json.dumps(reply).encode())])
    monkeypatch.setattr(execution, "RouterConfig", lambda **kwargs: RouterConfig(
        **{**{k: v for k, v in kwargs.items() if k != "timeouts"}, "api_keys": {"gemini": "fake"}, "transport": transport},
    ))

    class Transcribe(dspy.Signature):
        audio: dspy.Audio = dspy.InputField()
        text: str = dspy.OutputField()

    with dspy.context(lm=dspy.LM("gemini/gemini-3.8-flash", cache=False), adapter=dspy.ChatAdapter()):
        result = dspy.Predict(Transcribe)(audio=dspy.Audio(data="T2dnUw==", audio_format="ogg"))

    assert result.text == "hej"
    parts = json.loads(transport.requests[0].body)["contents"][-1]["parts"]
    assert {"inlineData": {"mimeType": "audio/ogg", "data": "T2dnUw=="}} in parts
