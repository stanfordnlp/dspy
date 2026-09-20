import copy
import json

import pytest

import dspy
from dspy.experimental import TypeSafe
from tests.predict.test_decide import decide

sdk = pytest.importorskip("typesafe_sdk")
httpx = pytest.importorskip("httpx2")


@pytest.fixture
def transport(monkeypatch):
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append((str(request.url), body))
        answers = {}
        for name, q in body["questions"].items():
            if q["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.8}
            elif q["type"] == "score":
                answers[name] = {
                    "type": "score",
                    "score": 1.5,
                    "confidence": 0.61,
                    "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
                    "legend": {"0": "bad", "1": "fair", "2": "great"},
                }
            else:
                answers[name] = {
                    "type": "choice",
                    "choice": "2",
                    "confidence": 0.73,
                    "probabilities": {"2": 0.8, "other": 0.2},
                }
        return httpx.Response(
            200, json={"model": body["model"], "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 2}}
        )

    sync, asynchronous = sdk.TypeSafeClient, sdk.AsyncTypeSafeClient
    monkeypatch.setattr(sdk, "TypeSafeClient", lambda **kwargs: sync(**kwargs, transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(
        sdk, "AsyncTypeSafeClient", lambda **kwargs: asynchronous(**kwargs, transport=httpx.MockTransport(respond))
    )
    return calls


def test_real_sdk_request_cache_usage_and_local_parameters(transport):
    client = TypeSafe("jev-test", api_key="test-only", base_url="https://example.test")
    module = decide(True, client)
    with dspy.context(track_usage=True):
        first = module(text="x")
        module.thresholds["flag"] = 0.9
        module.weights["rating"] = [0, 2, 8]
        second = module(text="x")
    assert first.flag.value is True
    assert second.flag.value is False
    assert first.flag.confidence == pytest.approx(0.6)
    assert second.flag.confidence == pytest.approx(1 / 9)
    assert first.rating.value == pytest.approx(6.7)
    assert second.rating.value == pytest.approx(5.4)
    assert first.get_lm_usage() == {"jev-test": {"prompt_tokens": 10, "completion_tokens": 2}}
    assert second.get_lm_usage() == {}
    assert len(transport) == 1
    url, body = transport[0]
    assert url == "https://example.test/v1/systemone"
    assert body["state"] == {"text": "x"}
    assert body["questions"]["rating"]["criteria"] == ["bad", "fair", "great"]
    assert "thresholds" not in json.dumps(body)
    assert "test-only" not in json.dumps(client.history)
    assert [h["cache_hit"] for h in client.history] == [False, True]
    first.rating.probabilities[0] = 0.99
    assert module(text="x").rating.probabilities[0] == 0.1


@pytest.mark.asyncio
async def test_async_sdk_and_shared_cache(transport):
    module = decide(client=TypeSafe("jev-test", api_key="test-only"))
    asynchronous = await module.acall(text="x")
    synchronous = module(text="x")
    assert asynchronous.toDict() == synchronous.toDict()
    assert len(transport) == 1
    assert synchronous.rating == pytest.approx(6.7)


def test_cache_identity_and_controls(transport):
    client = TypeSafe("jev-test", api_key="test-only", base_url="https://a.test")
    module = decide(client=client)
    module(text="x")
    client.base_url = "https://b.test"
    module(text="x")
    client.model = "jev-other"
    module(text="x")
    module(text="changed")
    client.cache = False
    module(text="changed")
    assert len(transport) == 5
    with dspy.context(disable_history=True):
        module(text="changed")
    assert len(client.history) == 5
    with dspy.context(max_history_size=1):
        module(text="changed")
    assert len(client.history) == 1


def test_client_copy_and_environment(monkeypatch, transport):
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-environment")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://environment.test/")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-environment-key")
    client = TypeSafe()
    module = decide(client=client)
    assert module(text="x").flag is True
    assert transport[0][0] == "https://environment.test/v1/systemone"
    assert transport[0][1]["model"] == "jev-environment"
    duplicate = copy.deepcopy(client)
    duplicate.history.clear()
    assert len(client.history) == 1
    assert "api_key" not in client.dump_state()
