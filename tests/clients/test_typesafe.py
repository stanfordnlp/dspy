import copy
import io
import json

import pytest

import dspy
from dspy._vendor.lm15.transports import StdlibAsyncTransport, StdlibTransport
from dspy.experimental import TypeSafe
from tests.clients.test_lm_engine_execution import DualResponse
from tests.predict.test_decision_parameters import predict, signature


@pytest.fixture
def transport(monkeypatch):
    calls = []
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-default-key")

    def respond(request):
        body = json.loads(request.body)
        calls.append((str(request.url), body, dict(request.headers)))
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
        return DualResponse(
            200, json.dumps({
                "model": body["model"], "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 2},
            }).encode(),
        )

    monkeypatch.setattr(StdlibTransport, "stream", lambda self, request: respond(request))
    monkeypatch.setattr(StdlibAsyncTransport, "stream", lambda self, request: respond(request))
    return calls


def test_lm15_request_cache_usage_and_local_parameters(transport):
    client = TypeSafe("jev-test", api_key="test-only", base_url="https://example.test")
    module = predict(True, client)
    with dspy.context(track_usage=True):
        first = module(text="x")
        module.fields["flag"]["threshold"] = 0.9
        module.fields["rating"]["cuts"] = [0.5, 1.6]
        module.fields["label"]["weights"] = {"2": 0.1}
        second = module(text="x")
    assert first.flag.value is True
    assert second.flag.value is False
    assert first.flag.confidence == pytest.approx(0.6)
    assert second.flag.confidence == pytest.approx(1 / 9)
    assert first.rating.value == pytest.approx(1.5)
    assert second.rating.value == first.rating.value
    assert first.rating.level == 2
    assert second.rating.level == 1
    assert first.label.value == 2
    assert second.label.value == "other"
    assert second.label.probabilities == first.label.probabilities == {"2": 0.8, "other": 0.2}
    assert second.label.confidence == first.label.confidence == 0.73
    assert first.get_lm_usage() == {"jev-test": {"prompt_tokens": 10, "completion_tokens": 2}}
    assert second.get_lm_usage() == {}
    assert len(transport) == 1
    url, body, headers = transport[0]
    assert url == "https://example.test/v1/systemone"
    assert headers["Authorization"] == "Bearer test-only"
    assert body["state"] == {
        "instructions": "Assess the document.",
        "input_fields": "1. `text` (str):",
        "inputs": {"text": "x"},
    }
    assert body["questions"]["rating"]["criteria"] == ["bad", "fair", "great"]
    assert "thresholds" not in json.dumps(body)
    assert "weights" not in json.dumps(body)
    assert "cuts" not in json.dumps(body)
    assert "test-only" not in json.dumps(client.history)
    assert [h["cache_hit"] for h in client.history] == [False, True]
    first.rating.probabilities[0] = 0.99
    assert module(text="x").rating.probabilities[0] == 0.1


@pytest.mark.asyncio
async def test_async_lm15_and_shared_cache(transport):
    module = predict(client=TypeSafe("jev-test", api_key="test-only"))
    asynchronous = await module.acall(text="x")
    synchronous = module(text="x")
    assert asynchronous.toDict() == synchronous.toDict()
    module.fields["label"]["weights"] = {"2": 0.1}
    assert (await module.acall(text="x")).label == "other"
    assert len(transport) == 1
    assert synchronous.rating.value == pytest.approx(1.5)


def test_cache_identity_and_controls(transport):
    client = TypeSafe("jev-test", api_key="test-only", base_url="https://a.test")
    module = predict(client=client)
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


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.asyncio
async def test_zero_local_history_limit_preserves_global_history(transport, monkeypatch, disabled):
    from dspy.clients import base_lm

    history = []
    monkeypatch.setattr(base_lm, "GLOBAL_HISTORY", history)
    client = TypeSafe("jev-test", api_key="test-only")
    module = predict(client=client)
    with dspy.context(max_history_size=0, disable_history=disabled):
        module(text="x")
        await module.acall(text="x")
    assert client.history == module.history == []
    assert len(history) == (0 if disabled else 2)
    if not disabled:
        assert [entry["cache_hit"] for entry in history] == [False, True]


def test_client_copy_and_environment(monkeypatch, transport):
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-environment")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://environment.test/")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-environment-key")
    client = TypeSafe()
    module = predict(client=client)
    assert module(text="x").flag is True
    assert transport[0][0] == "https://environment.test/v1/systemone"
    assert transport[0][1]["model"] == "jev-environment"
    assert transport[0][2]["Authorization"] == "Bearer test-environment-key"
    duplicate = copy.deepcopy(client)
    duplicate.history.clear()
    assert len(client.history) == 1
    assert "api_key" not in client.dump_state()


@pytest.mark.asyncio
async def test_predict_demos_lm15_cache_history_and_usage(transport, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-sdk-key")
    module = dspy.Predict(signature(True))
    module.set_lm(TypeSafe("jev-test"))
    module.demos = [dspy.Example(text="labeled document", flag=False, rating=0.0, label="other")]
    with dspy.context(track_usage=True):
        first = module(text="query")
        second = await module.acall(text="query")
        await module.acall(text="query", demos=[])
    assert len(transport) == 2
    assert transport[0][1]["state"]["demos"] == [module.demos[0].toDict()]
    assert "demos" not in transport[1][1]["state"]
    assert module.demos == [dspy.Example(text="labeled document", flag=False, rating=0.0, label="other")]
    assert first.toDict() == second.toDict()
    assert first.get_lm_usage() == {"jev-test": {"prompt_tokens": 10, "completion_tokens": 2}}
    assert second.get_lm_usage() == {}
    assert len(module.history) == 3
    stream = io.StringIO()
    module.inspect_history(file=stream)
    assert "query" in stream.getvalue()
    assert "probabilities" in stream.getvalue()


@pytest.mark.asyncio
async def test_standalone_client_callbacks_copy_and_reload(transport, tmp_path):
    from dspy.utils.callback import BaseCallback

    class Calls(BaseCallback):
        def __init__(self):
            self.events = []

        def on_lm_start(self, **kwargs):
            self.events.append("start")

        def on_lm_end(self, **kwargs):
            self.events.append("end")

    callbacks = Calls()
    client = TypeSafe("jev-test", base_url="https://original.test", callbacks=[callbacks])
    assert not isinstance(client, dspy.BaseLM)
    module = dspy.Predict(signature(True))
    module.set_lm(client)
    first = module(text="x")
    assert (await module.acall(text="x")).toDict() == first.toDict()
    assert callbacks.events == ["start", "end", "start", "end"]
    duplicate = client.copy(base_url="https://copy.test")
    assert duplicate.history == []
    duplicate.callbacks.clear()
    assert client.callbacks == [callbacks]
    assert len(client.history) == len(module.history) == 2
    module(text="x", lm=duplicate)
    assert len(transport) == 2  # Different endpoints must not share cached evidence.
    path = tmp_path / "standalone.json"
    module.save(path)
    restored = dspy.Predict(signature(True))
    restored.load(path, allow_unsafe_lm_state=True)
    assert type(restored.lm) is TypeSafe
    assert restored.lm.base_url == client.base_url
    assert restored(text="x").toDict() == first.toDict()
    assert len(transport) == 2
    with pytest.raises(TypeError):
        client.copy(temperature=0.7)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_structured_questions_and_raw_evidence(transport, asynchronous):
    client = TypeSafe("jev-test")
    state = {"inputs": {"text": "query"}, "demos": [{"text": "example", "name": False}]}
    questions = {
        "name": {
            "type": "noul", "instructions": {"focus": ["impact"]},
            "criteria": {"true": {"examples": ["Outage"]}, "false": None},
        },
        "rating": {
            "type": "score", "instructions": ["Assess severity.", {"ignore": "tone"}],
            "criteria": [{"what": "bad"}, "fair", {"examples": ["Resolved"]}],
        },
        "strict": {
            "type": "choice", "instructions": None, "criteria": {"2": {"what": "Known"}, "other": None},
        },
    }
    response = await client.acall(state, questions) if asynchronous else client(state, questions)
    assert transport[0][1] == {"model": "jev-test", "state": state, "questions": questions}
    assert response == {
        "name": {"noul": 0.8},
        "rating": {"score": 1.5, "confidence": 0.61, "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}},
        "strict": {"choice": "2", "confidence": 0.73, "probabilities": {"2": 0.8, "other": 0.2}},
    }
