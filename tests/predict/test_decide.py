import copy
from typing import Annotated, Literal, get_args

import pytest

import dspy
from dspy.utils.callback import BaseCallback

Rating = dspy.Score[(-2, "bad"), (3, "fair"), (10, "great")]
Label = dspy.Choice[(2, ""), ("other", "")]


class FakeClient:
    def __init__(self, probability=0.8, choice="2"):
        self.probability = probability
        self.choice = choice
        self.calls = []

    def __call__(self, state, questions):
        self.calls.append(copy.deepcopy((state, questions)))
        answers = {}
        for name, question in questions.items():
            if question["type"] == "noul":
                answers[name] = {"noul": self.probability}
            elif question["type"] == "score":
                answers[name] = {"score": 1.7, "confidence": 0.61, "probabilities": {0: 0.1, 1: 0.3, 2: 0.6}}
            else:
                answers[name] = {
                    "choice": self.choice,
                    "confidence": 0.73,
                    "probabilities": {key: 0.8 if key == self.choice else 0.2 for key in question["criteria"]},
                }
        return answers

    async def acall(self, **kwargs):
        return self(**kwargs)


def signature(rich=False):
    return dspy.Signature(
        {
            "text": (str, dspy.InputField()),
            "flag": (dspy.Noul if rich else bool, dspy.OutputField(desc="Is it relevant?")),
            "rating": (Rating if rich else Annotated[float, Rating], dspy.OutputField(desc="Rate usefulness.")),
            "label": (Label if rich else Literal[2, "other"], dspy.OutputField()),
        },
        "Assess the document.",
    )


def decide(rich=False, client=None):
    return dspy.Decide(signature(rich), client=client)


def test_native_rich_equivalence_and_request_mapping():
    client = FakeClient()
    native = decide(client=client)
    rich = decide(True, client)
    native.thresholds["flag"] = rich.thresholds["flag"] = 0.7
    a, b = native(text="example"), rich(text="example")
    assert a.flag is b.flag.value is True
    assert a.rating == b.rating.value == pytest.approx(6.7)
    assert a.label == b.label.value == 2
    assert type(a.label) is type(b.label.value) is int
    assert type(a.rating) is float
    assert b.flag.confidence == pytest.approx(1 / 7)
    assert b.flag.probability == 0.8
    assert b.rating.confidence == 0.61
    assert b.rating.probabilities == {0: 0.1, 1: 0.3, 2: 0.6}
    assert b.label.confidence == 0.73  # Not max(probabilities).
    assert bool(b.flag) is True
    assert float(b.rating) == pytest.approx(6.7)
    assert client.calls[0] == client.calls[1]
    state, questions = client.calls[0]
    assert state == {"text": "example"}
    assert questions == {
        "flag": {
            "type": "noul",
            "instructions": {
                "question": "Is it relevant?",
                "task": "Assess the document.",
                "inputs": "1. `text` (str):",
            },
        },
        "rating": {
            "type": "score",
            "criteria": ["bad", "fair", "great"],
            "instructions": {
                "question": "Rate usefulness.",
                "task": "Assess the document.",
                "inputs": "1. `text` (str):",
            },
        },
        "label": {
            "type": "choice",
            "criteria": {"2": None, "other": None},
            "instructions": {
                "question": "Decide `label`.",
                "task": "Assess the document.",
                "inputs": "1. `text` (str):",
            },
        },
    }


@pytest.mark.parametrize(
    "p,value,confidence", [(0, False, 1), (0.6, False, 0.2), (0.75, True, 0), (0.9, True, 0.2), (1, True, 1 / 3)]
)
def test_bool_boundary_and_confidence(p, value, confidence):
    module = dspy.Decide("text -> flag: dspy.Noul", client=FakeClient(p))
    module.thresholds["flag"] = 0.75
    result = module(text="x").flag
    assert result.value is value
    assert result.confidence == pytest.approx(confidence)
    assert bool(result) is value


@pytest.mark.parametrize("threshold,p,value", [(0, 0, True), (1, 0.99, False), (1, 1, True)])
def test_threshold_endpoints(threshold, p, value):
    module = dspy.Decide("text -> flag: bool", client=FakeClient(p))
    module.thresholds["flag"] = threshold
    assert module(text="x").flag is value


@pytest.mark.parametrize(
    "options,selected,expected",
    [
        (Literal[1, 2], "2", 2),
        (Literal[False, True], "False", False),
        (Literal[True, 1, "other"], "1", 1),
        (Literal[None, "other"], "None", None),
        (Literal["a", "b"], "b", "b"),
    ],
)
@pytest.mark.parametrize("rich", [False, True])
def test_literal_membership_and_exact_type(options, selected, expected, rich):
    sig = dspy.Signature(
        {
            "text": (str, dspy.InputField()),
            "label": (dspy.Choice[tuple((v, "") for v in get_args(options))] if rich else options, dspy.OutputField()),
        }
    )
    result = dspy.Decide(sig, client=FakeClient(choice=selected))(text="x").label
    value = result.value if rich else result
    assert value == expected
    assert type(value) is type(expected)


def test_score_weights_normalization_and_snapshot():
    client = FakeClient()
    module = decide(True, client)
    initial = module(text="x").rating
    module.weights["rating"] = [0, 2, 8]
    changed = module(text="x").rating
    assert initial.value == pytest.approx(6.7)  # Computed from probabilities, not raw SDK score.
    assert changed.value == pytest.approx(5.4)
    assert changed.confidence == initial.confidence == 0.61
    assert Rating.options == ((-2, "bad"), (3, "fair"), (10, "great"))
    assert client.calls[0] == client.calls[1]
    # The provider only promises approximate unit mass; normalization is explicit.
    answers = client(*client.calls[0])
    answers["rating"]["probabilities"] = {0: 0.099, 1: 0.297, 2: 0.594}
    with dspy.context(system_one=lambda **_: answers):
        result = decide(True)(text="x").rating
    assert result.value == pytest.approx(6.7)
    assert sum(result.probabilities.values()) == pytest.approx(0.99)


@pytest.mark.parametrize(
    "sig,match",
    [
        ("text -> answer", "Unsupported"),
        ("text -> answer: int", "Unsupported"),
        ("text -> answer: float", "requires a rubric"),
        ("text -> answer: dspy.Choice", "Unsupported"),
        ("text -> answer: Literal[1, '1']", "ambiguous"),
    ],
)
def test_reject_unsupported(sig, match):
    with pytest.raises(ValueError, match=match):
        dspy.Decide(sig)


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("thresholds", {"flag": -0.1}),
        ("thresholds", {"flag": float("nan")}),
        ("thresholds", {"missing": 0.3}),
        ("weights", {"rating": [0, 3]}),
        ("weights", {"rating": [-2, float("inf"), 10]}),
        ("weights", {"rating": [0, 9, 2]}),
        ("weights", {"rating": [-3, 2, 10]}),
    ],
)
def test_reject_invalid_parameters(attribute, value):
    client = FakeClient()
    module = decide(client=client)
    setattr(module, attribute, value)
    with pytest.raises(ValueError):
        module(text="x")
    assert not client.calls


@pytest.mark.parametrize(
    "field,update",
    [
        ("flag", {"noul": 1.1}),
        ("flag", {"noul": float("nan")}),
        ("rating", {"probabilities": {0: 0.1, 1: 0.9}}),
        ("rating", {"probabilities": {0: 0, 1: 0, 2: 0}}),
        ("rating", {"probabilities": {0: -0.1, 1: 0.3, 2: 0.8}}),
        ("label", {"choice": "unknown"}),
        ("label", {"confidence": 1.2}),
    ],
)
def test_reject_malformed_answers(field, update):
    client = FakeClient()
    module = decide(client=client)
    module(text="x")
    answers = client(*client.calls[0])
    answers[field].update(update)
    module.client = lambda **_: answers
    with pytest.raises(ValueError):
        module(text="x")


def test_composition_discovery_trace_callbacks_and_demos():
    class Callback(BaseCallback):
        def __init__(self):
            self.outputs = []

        def on_module_end(self, call_id, outputs, exception=None):
            self.outputs.append(outputs)

    class Pipeline(dspy.Module):
        def __init__(self):
            self.decide = decide(True)

        def forward(self, text):
            return self.decide(text=text)

    program = Pipeline()
    client = FakeClient()
    callback = Callback()
    assert program.named_predictors() == [("decide", program.decide)]
    assert program.named_parameters() == [("decide", program.decide)]
    demo = dspy.Example(text="demo", flag=dspy.Noul(value=False, confidence=0.8, probability=0.1), unused="omit")
    program.decide.demos = [demo]
    trace = []
    with dspy.context(system_one=client, trace=trace, max_trace_size=1, callbacks=[callback]):
        result = program(text="first")
        result = program(text="second")
        program.decide(text="untraced", _trace=False)
    assert trace == [(program.decide, {"text": "second"}, result)]
    assert result in callback.outputs
    assert client.calls[0][1]["flag"]["instructions"]["examples"] == [{"inputs": {"text": "demo"}, "answer": False}]
    assert demo.unused == "omit"


@pytest.mark.asyncio
async def test_async_and_client_resolution():
    global_client = FakeClient(0.1)
    explicit = FakeClient(0.9)
    module = decide(True)
    with pytest.raises(ValueError, match="Configure a System One"):
        module(text="x")
    with dspy.context(system_one=global_client):
        assert (await module.acall(text="x")).flag.value is False
        module.client = explicit
        assert (await module.acall(text="x")).flag.value is True
    assert len(global_client.calls) == len(explicit.calls) == 1


def test_defaults_and_input_errors():
    class Sig(dspy.Signature):
        text: str = dspy.InputField(default="default")
        flag: bool = dspy.OutputField()

    client = FakeClient()
    module = dspy.Decide(Sig, client=client)
    assert module().flag is True
    assert client.calls[0][0] == {"text": "default"}
    with pytest.raises(ValueError, match="Unexpected"):
        module(other=1)
    with pytest.raises(ValueError, match="Missing"):
        decide(client=client)()


def test_copy_and_json_state_preserve_config_and_rich_demos(tmp_path):
    module = decide(True)
    module.thresholds["flag"] = 0.7
    module.weights["rating"] = [-2, 1, 10]
    with dspy.context(system_one=FakeClient()):
        result = module(text="x")
        module.demos = [dspy.Example(text="x", **dict(result.items()))]
        before = module(text="next")
    duplicate = module.deepcopy()
    duplicate.weights["rating"][1] = 4
    duplicate.thresholds["flag"] = 0.3
    assert module.weights["rating"] == [-2, 1, 10]
    assert module.thresholds["flag"] == 0.7
    assert decide(True).weights["rating"] == [-2, 3, 10]
    path = tmp_path / "decide.json"
    module.save(path)
    restored = decide(True)
    restored.load(path)
    assert restored.weights == module.weights
    assert restored.thresholds == module.thresholds
    assert isinstance(restored.demos[0]["flag"], dspy.Noul)
    assert isinstance(restored.demos[0]["rating"], dspy.Score)
    assert isinstance(restored.demos[0]["label"], dspy.Choice)
    assert type(restored.demos[0]["label"].value) is int
    with dspy.context(system_one=FakeClient()):
        assert restored(text="next").toDict() == before.toDict()


def test_explicit_client_state_omits_key_and_gates_endpoint(tmp_path):
    module = decide(client=dspy.TypeSafe("jev-test", api_key="test-credential", base_url="https://example.test"))
    path = tmp_path / "decide.json"
    module.save(path)
    assert "test-credential" not in path.read_text()
    restored = decide()
    restored.load(path)
    assert restored.client.model == "jev-test"
    assert restored.client.base_url != "https://example.test"
    restored.load(path, allow_unsafe_lm_state=True)
    assert restored.client.base_url == "https://example.test"
    assert restored.client.api_key is None


def test_full_program_save_load(tmp_path):
    module = decide(True, client=dspy.TypeSafe("jev-test"))
    module.thresholds["flag"] = 0.7
    module.save(tmp_path / "program", save_program=True)
    restored = dspy.load(tmp_path / "program", allow_pickle=True)
    assert restored.weights == module.weights
    assert restored.thresholds == module.thresholds
    assert restored.client.model == "jev-test"
    restored.client = FakeClient()
    assert restored(text="x").flag.confidence == pytest.approx(1 / 7)


def test_thresholds_are_per_field_and_results_are_snapshots():
    module = dspy.Decide("text -> a: bool, b: dspy.Noul", client=FakeClient(0.6))
    assert module.thresholds == {"a": 0.5, "b": 0.5}
    module.thresholds["b"] = 0.8
    before = module(text="x")
    assert before.a is True and before.b.value is False
    module.thresholds["b"] = 0.4
    assert module(text="x").b.value is True
    assert before.b.value is False
    assert before.b.confidence == pytest.approx(0.25)


@pytest.mark.parametrize("rich", [False, True])
def test_batch_discovery_and_context(rich):
    module = decide(rich)
    with dspy.context(system_one=FakeClient()):
        results = module.batch([dspy.Example(text=str(i)).with_inputs("text") for i in range(3)], num_threads=2)
    assert len(results) == 3
    for result in results:
        assert (result.rating.value if rich else result.rating) == pytest.approx(6.7)
