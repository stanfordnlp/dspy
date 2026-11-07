import json
from typing import Literal

import pytest

import dspy
from dspy._vendor.lm15.transports import StdlibAsyncTransport, StdlibTransport
from dspy.experimental import Choice, Noul, Score
from dspy.utils.dummies import DummyLM
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
        for name, question in body["questions"].items():
            if question["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.8}
            elif question["type"] == "score":
                answers[name] = {
                    "type": "score", "score": 1.5, "confidence": 0.61,
                    "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
                }
            else:
                answers[name] = {
                    "type": "choice", "choice": "2", "confidence": 0.73,
                    "probabilities": {"2": 0.8, "other": 0.2},
                }
        return DualResponse(200, json.dumps({
            "model": body["model"], "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 2},
        }).encode())

    monkeypatch.setattr(StdlibTransport, "stream", lambda self, request: respond(request))
    monkeypatch.setattr(StdlibAsyncTransport, "stream", lambda self, request: respond(request))
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["auto", "lm15"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_predict_uses_lm15_and_reinterprets_cached_evidence(transport, engine, asynchronous):
    model = "typesafe:jev-test"
    lm = dspy.LM(model, engine=engine, api_key="test-explicit-key", api_base="https://example.test", num_retries=0)
    module = predict(True, lm)
    try:
        with dspy.context(track_usage=True):
            first = await module.acall(text="x") if asynchronous else module(text="x")
            module.fields["flag"]["threshold"] = 0.9
            module.fields["rating"]["cuts"] = [0.5, 1.6]
            module.fields["label"]["weights"] = {"2": 0.1}
            second = await module.acall(text="x") if asynchronous else module(text="x")
        assert first.flag.value is True and second.flag.value is False
        assert first.flag.confidence == pytest.approx(0.6)
        assert second.flag.confidence == pytest.approx(1 / 9)
        assert first.rating.value == second.rating.value == pytest.approx(1.5)
        assert first.rating.level == 2 and second.rating.level == 1
        assert first.label.value == 2 and second.label.value == "other"
        assert second.label.probabilities == first.label.probabilities == {"2": 0.8, "other": 0.2}
        assert second.label.confidence == first.label.confidence == 0.73
        assert first.get_lm_usage()[model]["prompt_tokens"] == 10
        assert first.get_lm_usage()[model]["completion_tokens"] == 2
        assert second.get_lm_usage() == {}
        assert len(transport) == 1
        url, body, headers = transport[0]
        assert url == "https://example.test/v1/systemone"
        assert headers["Authorization"] == "Bearer test-explicit-key"
        assert body["model"] == "jev-test"
        assert body["state"] == {
            "instructions": "Assess the document.", "input_fields": "1. `text` (str):", "inputs": {"text": "x"},
        }
        assert body["questions"]["rating"]["criteria"] == ["bad", "fair", "great"]
        assert not {"threshold", "weights", "cuts"} & body["questions"]["flag"].keys()
        assert isinstance(lm.history[-1]["request"], dspy.lm15.Request)
        assert lm.history[-1]["response"].provider_data["typesafe"]["answers"]["rating"]["score"] == 1.5
        first.rating.probabilities[0] = 0.99
        third = await module.acall(text="x") if asynchronous else module(text="x")
        assert third.rating.probabilities[0] == 0.1
    finally:
        await lm.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("rich", [False, True])
async def test_empty_choice_labels_are_rejected_only_by_jev(transport, asynchronous, rich):
    annotation = Choice[("", "No match"), ("match", "Match")] if rich else Literal["", "match"]
    module = dspy.Predict(dspy.Signature({
        "text": (str, dspy.InputField()), "label": (annotation, dspy.OutputField(desc="Select a match.")),
    }))
    lm = dspy.LM("typesafe:jev-test")
    module.set_lm(lm)
    try:
        with pytest.raises(ValueError, match="Jev output 'label'.*empty choice label.*no_match"):
            await module.acall(text="x") if asynchronous else module(text="x")
        assert transport == []
    finally:
        await lm.aclose()

    evidence = {"probabilities": {"": 0.8, "match": 0.2}, "confidence": 0.6}
    module.set_lm(DummyLM([{"label": evidence if rich else ""}]))
    result = await module.acall(text="x") if asynchronous else module(text="x")
    assert (result.label.value if rich else result.label) == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_structured_questions_and_raw_evidence(transport, asynchronous):
    sig = dspy.Signature({
        "text": (str, dspy.InputField()),
        "name": (Noul, dspy.OutputField()),
        "rating": (Score["bad", "fair", "great"], dspy.OutputField()),
        "strict": (Choice[(2, ""), ("other", "")], dspy.OutputField()),
    }, "Assess the document.")
    lm = dspy.LM("typesafe:jev-test", engine="lm15")
    module = dspy.Predict(sig)
    module.set_lm(lm)
    module.demos = [dspy.Example(text="example", name=False)]
    module.fields = {
        "name": {"instructions": {"focus": ["impact"]}, "criteria": {"true": {"examples": ["Outage"]}, "false": None}},
        "rating": {"instructions": ["Assess severity.", {"ignore": "tone"}], "criteria": [{"what": "bad"}, "fair", {"examples": ["Resolved"]}]},
        "strict": {"instructions": None, "criteria": {"2": {"what": "Known"}, "other": None}},
    }
    try:
        result = await module.acall(text="query") if asynchronous else module(text="query")
        _, body, headers = transport[0]
        assert headers["Authorization"] == "Bearer test-default-key"
        assert body["state"]["inputs"] == {"text": "query"}
        assert body["state"]["demos"] == [{"text": "example", "name": False}]
        assert body["questions"] == {
            name: {"type": kind, **module.fields[name]}
            for name, kind in [("name", "noul"), ("rating", "score"), ("strict", "choice")]
        }
        assert result.name.probability == 0.8
        assert result.rating.value == pytest.approx(1.5)
        assert result.rating.confidence == 0.61
        assert result.strict.value == 2
        assert result.strict.confidence == 0.73
    finally:
        await lm.aclose()


def test_jev_predict_uses_normal_lm_persistence_and_callbacks(transport, tmp_path):
    from dspy.utils.callback import BaseCallback

    class Calls(BaseCallback):
        def __init__(self):
            self.events = []

        def on_lm_start(self, **kwargs):
            self.events.append("start")

        def on_lm_end(self, **kwargs):
            self.events.append("end")

    callbacks = Calls()
    lm = dspy.LM("typesafe:jev-test", engine="lm15", callbacks=[callbacks])
    module = predict(True, lm)
    restored = dspy.Predict(signature(True))
    try:
        first = module(text="x")
        assert module(text="x").toDict() == first.toDict()
        assert callbacks.events == ["start", "end", "start", "end"]
        path = tmp_path / "jev.json"
        module.save(path)
        restored.load(path)
        assert type(restored.lm) is dspy.LM
        assert restored.lm.engine == "lm15"
        assert restored(text="x").toDict() == first.toDict()
        assert len(transport) == 1
    finally:
        lm.close()
        if restored.lm is not None:
            restored.lm.close()
