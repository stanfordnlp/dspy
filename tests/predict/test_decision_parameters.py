"""Decision parameters exercised through Predict, independent of provider transport."""

import copy
import json
from typing import Annotated, Literal, get_args

import pytest

import dspy
from dspy.experimental import Choice, Noul, Score, TypeSafe

Rating = Score["bad", "fair", "great"]
Label = Choice[(2, ""), ("other", "")]


class FakeClient:
    supports_decision_requests = True

    def __init__(self, probability=0.8, choice="2", answers=None):
        self.probability = probability
        self.choice = choice
        self.answers = answers
        self.calls = []

    def __call__(self, state, questions):
        self.calls.append(copy.deepcopy((state, questions)))
        if self.answers is not None:
            return copy.deepcopy(self.answers)
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
            "flag": (Noul if rich else bool, dspy.OutputField(desc="Is it relevant?")),
            "rating": (Rating, dspy.OutputField(desc="Rate usefulness.")),
            "label": (Label if rich else Literal[2, "other"], dspy.OutputField(desc="Classify the document.")),
        },
        "Assess the document.",
    )


def predict(rich=False, client=None):
    module = dspy.Predict(signature(rich))
    module.set_lm(client)
    # Opt outputs into configurable evidence decoding without pinning numeric defaults.
    module.fields = {"flag": {}, "rating": {}, "label": {}}
    return module


@pytest.mark.parametrize("overrides", [{}, {"flag": {"threshold": 0.9}}])
def test_only_explicit_overrides_survive_inference_and_save_load(tmp_path, overrides):
    module = dspy.Predict(signature(True))
    assert module.fields == {}
    module.fields = copy.deepcopy(overrides)
    assert module.get_criteria("rating") == ["bad", "fair", "great"]
    result = module(text="x", lm=FakeClient())
    assert result.flag.value is (not bool(overrides))
    assert result.rating.level == 2
    assert result.label.value == 2
    assert module.fields == overrides
    path = tmp_path / "overrides.json"
    module.save(path)
    assert json.loads(path.read_text()).get("fields", {}) == overrides
    restored = dspy.Predict(signature(True))
    restored.load(path)
    assert restored(text="x", lm=FakeClient()).toDict() == result.toDict()
    assert restored.fields == overrides


def test_native_rich_equivalence_and_request_mapping():
    client = FakeClient()
    native, rich = predict(client=client), predict(True, client)
    native.fields["flag"]["threshold"] = rich.fields["flag"]["threshold"] = 0.7
    a, b = native(text="example"), rich(text="example")
    assert a.flag is b.flag.value is True
    assert a.rating.value == b.rating.value == pytest.approx(1.5)
    assert a.label == b.label.value == 2
    assert type(a.label) is type(b.label.value) is int
    assert b.flag.confidence == pytest.approx(1 / 7)
    assert b.flag.probability == 0.8
    assert b.rating.confidence == 0.61
    assert b.label.confidence == 0.73
    assert client.calls[0] == client.calls[1]
    state, questions = client.calls[0]
    assert state == {
        "instructions": "Assess the document.",
        "input_fields": "1. `text` (str):",
        "inputs": {"text": "example"},
        "demos": [],
    }
    assert questions == {
        "flag": {"type": "noul", "instructions": "Is it relevant?"},
        "rating": {"type": "score", "instructions": "Rate usefulness.", "criteria": ["bad", "fair", "great"]},
        "label": {"type": "choice", "instructions": "Classify the document.", "criteria": {"2": None, "other": None}},
    }


@pytest.mark.parametrize(
    "p,value,confidence", [(0, False, 1), (0.6, False, 0.2), (0.75, True, 0), (0.9, True, 0.2), (1, True, 1 / 3)]
)
def test_threshold_boundary_and_confidence(p, value, confidence):
    module = predict(True, FakeClient(p))
    module.fields["flag"]["threshold"] = 0.75
    result = module(text="x").flag
    assert result.value is value
    assert result.confidence == pytest.approx(confidence)
    assert bool(result) is value


@pytest.mark.parametrize("threshold,p,value", [(0, 0, True), (1, 0.99, False), (1, 1, True)])
def test_threshold_endpoints(threshold, p, value):
    module = predict(client=FakeClient(p))
    module.fields["flag"]["threshold"] = threshold
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
    kind = Choice[tuple((v, "") for v in get_args(options))] if rich else options
    sig = signature().with_updated_fields("label", type_=kind)
    result = dspy.Predict(sig)(text="x", lm=FakeClient(choice=selected)).label
    value = result.value if rich else result
    assert value == expected
    assert type(value) is type(expected)


@pytest.mark.parametrize("cuts,level", [([0.5, 1.5], 2), ([0.5, 1.6], 1), ([0.5, 1.4], 2)])
def test_score_cuts_normalization_and_snapshot(cuts, level):
    client = FakeClient()
    module = predict(True, client)
    before = module(text="x").rating
    module.fields["rating"]["cuts"] = cuts
    after = module(text="x").rating
    assert before.value == after.value == pytest.approx(1.5)
    assert before.level == 2 and after.level == level
    assert before.confidence == after.confidence == 0.61
    assert client.calls[0] == client.calls[1]
    answers = client(*client.calls[0])
    answers["rating"]["probabilities"] = {0: 0.099, 1: 0.297, 2: 0.594}
    result = module(text="x", lm=FakeClient(answers=answers)).rating
    assert result.value == pytest.approx(1.5)
    assert sum(result.probabilities.values()) == pytest.approx(0.99)


@pytest.mark.parametrize("rich", [False, True])
def test_choice_weights_selection_evidence_and_request(rich):
    client = FakeClient()
    module = predict(rich, client)
    before = module(text="x").label
    module.fields["label"]["weights"] = {"2": 0.1}
    after = module(text="x").label
    assert (before.value if rich else before) == 2
    assert (after.value if rich else after) == "other"
    assert client.calls[0] == client.calls[1]
    if rich:
        assert after.probabilities == before.probabilities == {"2": 0.8, "other": 0.2}
        assert after.confidence == before.confidence == 0.73
    module.fields["label"]["weights"] = {"2": 10, "other": 100}
    result = module(text="x").label
    assert (result.value if rich else result) == "other"


@pytest.mark.parametrize("weights", [{}, {"2": 1}, {"2": 0.25}, {"2": 0}])
def test_choice_weights_defaults_ties_and_zero(weights):
    module = predict(True, FakeClient())
    module.fields["label"]["weights"] = weights
    assert module(text="x").label.value == ("other" if weights.get("2") == 0 else 2)


def test_choice_weighted_tie_prefers_raw_winner():
    module = predict(True, FakeClient(choice="other"))
    module.fields["label"]["weights"] = {"other": 0.25}
    assert module(text="x").label.value == "other"
    client = FakeClient(
        answers={
            "flag": {"noul": 0.8},
            "rating": {"probabilities": {0: 0.1, 1: 0.3, 2: 0.6}, "confidence": 0.6},
            "label": {"probabilities": {"2": 1, "other": 0}, "confidence": 0.7},
        }
    )
    module.fields["label"]["weights"] = {"2": 0}
    with pytest.raises(ValueError, match="no positive probability mass"):
        module(text="x", lm=client)


@pytest.mark.parametrize(
    "field,config",
    [
        ("flag", {"threshold": -0.1}),
        ("flag", {"threshold": float("nan")}),
        ("missing", {"threshold": 0.3}),
        ("flag", None),
        ("rating", {"cuts": [0.5, 1.5], "weights": [0, 3]}),
        ("flag", {"threshold": 0.5, "unknown": "typo"}),
        *[
            ("label", {"weights": w})
            for w in [
                {"unknown": 1},
                {2: 1},
                {"2": -1},
                {"2": float("nan")},
                {"2": float("inf")},
                {"2": True},
                {"2": 0, "other": 0},
                [1, 2],
            ]
        ],
        *[
            ("rating", {"cuts": c})
            for c in [
                [0, 1.5],
                [0.5, 2],
                [1.5, 0.5],
                [0.5, 0.5],
                [0.5],
                [True, 1.5],
                [0.5, float("nan")],
            ]
        ],
    ],
)
@pytest.mark.parametrize("operation", ["call", "save", "load"])
def test_invalid_parameters_rejected(field, config, operation, tmp_path):
    module = predict(True)
    original = module.dump_state()
    client = FakeClient()
    if operation == "load":
        state = copy.deepcopy(original)
        state["fields"][field] = config
        with pytest.raises(ValueError):
            module.load_state(state)
        assert module.dump_state() == original
    else:
        module.fields[field] = config
        with pytest.raises(ValueError):
            if operation == "call":
                module(text="x", lm=client)
            else:
                module.save(tmp_path / "invalid.json")
    assert not client.calls


@pytest.mark.parametrize(
    "field,update",
    [
        ("flag", {"noul": 1.1}),
        ("flag", {"noul": float("nan")}),
        ("rating", {"probabilities": {0: 0.1, 1: 0.9}}),
        ("rating", {"probabilities": {0: 0, 1: 0, 2: 0}}),
        ("rating", {"probabilities": {0: -0.1, 1: 0.3, 2: 0.8}}),
        ("rating", {"probabilities": {0.9: 0.1, 1.9: 0.3, 2.9: 0.6}}),
        ("rating", {"probabilities": {False: 0.1, True: 0.3, 2: 0.6}}),
        ("rating", {"probabilities": {"00": 0.1, "1": 0.3, "2": 0.6}}),
        ("rating", {"probabilities": {0: 0.1, "0": 0.1, 1: 0.3, 2: 0.6}}),
        ("label", {"probabilities": {"unknown": 1}}),
        ("label", {"confidence": 1.2}),
    ],
)
def test_malformed_evidence_is_rejected(field, update):
    client = FakeClient()
    module = predict(True, client)
    module(text="x")
    answers = client(*client.calls[0])
    answers[field].update(update)
    with pytest.raises(ValueError):
        module(text="x", lm=FakeClient(answers=answers))


@pytest.mark.parametrize("rich", [False, True])
@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize(
    "field,invalid",
    [
        ("flag", ["yes", "no"]),
        ("flag", {"yes": "Relevant"}),
        ("label", ["Known", "Other"]),
        ("label", {"2": "Known"}),
        ("label", {"2": "Known", "other": None, "unknown": "Extra"}),
        ("rating", {"0": "bad", "1": "fair", "2": "great"}),
        ("rating", ["bad", "great"]),
    ],
)
def test_criteria_shape_validated_on_save_and_load(tmp_path, rich, operation, field, invalid):
    module = predict(rich)
    original = module.dump_state()
    if operation == "save":
        module.fields[field]["criteria"] = invalid
        with pytest.raises(ValueError, match="criteria"):
            module.save(tmp_path / "invalid.json")
    else:
        state = copy.deepcopy(original)
        state["fields"][field]["criteria"] = invalid
        with pytest.raises(ValueError, match="criteria"):
            module.load_state(state)
        assert module.dump_state() == original


@pytest.mark.parametrize("entry", [{"nested": {1: "integer key"}}, {"nested": float("nan")}, {"nested": {"set"}}, 42])
def test_instructions_validate_json(entry):
    module = predict(True)
    module.fields["flag"]["instructions"] = entry
    with pytest.raises(ValueError, match="instructions"):
        module.dump_state()


@pytest.mark.parametrize("setting", ["instructions", "criteria"])
def test_nested_mutations_are_revalidated(setting):
    client = FakeClient()
    module = predict(True, client)
    examples = ["Relevant"]
    module.fields["flag"][setting] = {"true": {"examples": examples}}
    module(text="x")
    examples.append("Another example")
    module(text="x")
    assert client.calls[-1][1]["flag"][setting]["true"]["examples"] == examples
    for invalid in ({"not JSON"}, float("nan")):
        examples.append(invalid)
        with pytest.raises(ValueError, match=setting):
            module(text="x")
        examples.pop()
    assert len(client.calls) == 2


def test_inflight_decoding_uses_configuration_snapshot():
    class UpdatingClient(FakeClient):
        def __call__(self, state, questions):
            module.fields["flag"]["threshold"] = 0.9
            module.fields["rating"]["cuts"][1] = 1.6
            module.fields["label"]["weights"]["2"] = 0.1
            return super().__call__(state, questions)

    module = predict(True, UpdatingClient())
    module.fields["rating"] = {"cuts": [0.5, 1.5]}
    module.fields["label"] = {"weights": {"2": 1.0, "other": 1.0}}
    first = module(text="x")
    second = module(text="x")
    assert first.flag.value is True and second.flag.value is False
    assert first.rating.level == 2 and second.rating.level == 1
    assert first.label.value == 2 and second.label.value == "other"
    assert first.rating.probabilities == second.rating.probabilities
    assert first.label.probabilities == second.label.probabilities


def test_json_state_preserves_instructions_criteria_and_numeric_settings(tmp_path):
    module = predict(True)
    module.fields = {
        "flag": {
            "threshold": 0.7,
            "instructions": {"focus": ["impact"]},
            "criteria": {"true": {"examples": ["Outage"]}, "false": None},
        },
        "rating": {
            "cuts": [0.4, 1.6],
            "instructions": ["Assess severity.", {"ignore": "tone"}],
            "criteria": [{"what": "bad", "not_for": "cosmetic"}, "fair", {"examples": ["Resolved"]}],
        },
        "label": {
            "weights": {"2": 0.25, "other": 1.5},
            "instructions": "Choose category.",
            "criteria": {"2": {"what": "Known"}, "other": None},
        },
    }
    expected = copy.deepcopy(module.fields)
    path = tmp_path / "parameters.json"
    module.save(path)
    assert json.loads(path.read_text())["fields"] == expected
    restored = predict(True)
    restored.load(path)
    client = FakeClient()
    assert module(text="x", lm=client).toDict() == restored(text="x", lm=client).toDict()
    assert client.calls[0] == client.calls[1]
    for name, config in expected.items():
        question = client.calls[0][1][name]
        assert question["instructions"] == config["instructions"]
        assert question["criteria"] == config["criteria"]
        assert not {"threshold", "cuts", "weights"} & question.keys()
    restored.fields["flag"]["instructions"]["focus"].append("urgency")
    restored.fields["rating"]["cuts"][0] = 0.2
    assert module.fields == expected


@pytest.mark.parametrize(
    "kind,expected",
    [(Noul, None), (Noul[(True, "blocked")], {"true": "blocked"}),
     (Rating, ["bad", "fair", "great"]), (Label, {"2": None, "other": None})],
)
def test_get_criteria_does_not_require_request_instructions(kind, expected):
    sig = dspy.Signature({"flag": (kind, dspy.OutputField())})
    module = dspy.Predict(sig)
    assert module.get_criteria("flag") == expected
    with pytest.raises(ValueError, match="requires an OutputField"):
        module(lm=FakeClient())


def test_set_criteria_validates_replacement_and_preserves_other_settings():
    module = predict(True)
    module.fields["flag"].update(threshold=0.8, criteria={"wrong": "invalid"})
    before = copy.deepcopy(module.fields)
    with pytest.raises(ValueError, match="criteria"):
        module.set_criteria("flag", {"also_wrong": "invalid"})
    assert module.fields == before
    replacement = {"true": {"examples": ["Outage"]}}
    module.set_criteria("flag", replacement)
    replacement["true"]["examples"].clear()
    assert module.get_criteria("flag") == {"true": {"examples": ["Outage"]}}
    assert module.fields["flag"]["threshold"] == 0.8
    assert module.fields["rating"] == before["rating"]
    assert module.fields["label"] == before["label"]


@pytest.mark.parametrize("rich", [False, True])
def test_criteria_accessors_and_whole_program_roundtrip(tmp_path, rich):
    availability = Noul[(False, "Workaround available")]
    sig = signature(rich).with_updated_fields("flag", type_=availability if rich else Annotated[bool, availability])
    module = dspy.Predict(sig)
    assert module.get_criteria("flag") == {"false": "Workaround available"}
    criteria = {"true": {"examples": ["Outage"]}, "false": None}
    module.set_criteria("flag", criteria)
    criteria["true"]["examples"].clear()
    retrieved = module.get_criteria("flag")
    assert retrieved["true"]["examples"] == ["Outage"]
    retrieved["true"]["examples"].clear()
    assert module.get_criteria("flag")["true"]["examples"] == ["Outage"]
    module.fields["flag"]["threshold"] = 0.8
    module.save(tmp_path / "program", save_program=True)
    restored = dspy.load(tmp_path / "program", allow_pickle=True)
    result = restored(text="x", lm=FakeClient()).flag
    assert (result.value if rich else result) is True
    assert restored.fields == module.fields
    module.set_criteria("flag", None)
    assert module.get_criteria("flag") is None


def test_null_instructions_and_criteria_survive_reload(tmp_path):
    module = predict(True)
    module.fields["flag"].update(instructions=None, criteria=None)
    module.save(tmp_path / "null.json")
    restored = predict(True)
    restored.load(tmp_path / "null.json")
    client = FakeClient()
    restored(text="x", lm=client)
    assert client.calls[0][1]["flag"] == {"type": "noul", "instructions": None, "criteria": None}


def test_reset_copy_and_endpoint_state(tmp_path):
    module = predict(True, TypeSafe("jev-test", api_key="test-credential", base_url="https://example.test"))
    module.fields["flag"]["threshold"] = 0.9
    module.demos = [{"text": "example", "flag": False}]
    duplicate = module.reset_copy()
    assert duplicate.lm is None and duplicate.demos == []
    duplicate.fields["flag"]["threshold"] = 0.5
    assert module.fields["flag"]["threshold"] == 0.9
    path = tmp_path / "client.json"
    module.save(path)
    assert "test-credential" not in path.read_text()
    restored = predict(True)
    restored.load(path)
    assert restored.lm.base_url != "https://example.test"
    restored.load(path, allow_unsafe_lm_state=True)
    assert restored.lm.base_url == "https://example.test" and restored.lm.api_key is None


@pytest.mark.asyncio
async def test_client_resolution_batch_and_input_context():
    module = predict(True)
    client = FakeClient(0.1)
    with dspy.context(lm=client):
        assert (await module.acall(text="x")).flag.value is False
        assert (await module.acall(text="x", lm=FakeClient(0.9))).flag.value is True
        results = module.batch([dspy.Example(text=str(i)).with_inputs("text") for i in range(3)], num_threads=2)
    assert len(results) == 3 and all(result.rating.value == pytest.approx(1.5) for result in results)
    sig = dspy.Signature(
        {"inputs": (str, dspy.InputField()), "flag": (bool, dspy.OutputField(desc="Is it actionable?"))},
        "Shared task context.",
    )
    dspy.Predict(sig)(inputs="User content", lm=client)
    state, questions = client.calls[-1]
    assert state["inputs"] == {"inputs": "User content"}
    assert state["instructions"] == "Shared task context."
    assert "Shared task context." not in json.dumps(questions)


@pytest.mark.parametrize(
    "base,override",
    [
        (Score["low", "high"], Rating),
        (Rating, Score["great", "fair", "bad"]),
        (Literal[True], Literal[1]),
        (Literal["a", "b"], Literal["a", "c"]),
        (bool, Literal[False, True]),
    ],
)
@pytest.mark.asyncio
async def test_output_override_preserves_answer_space(base, override):
    sig = signature().with_updated_fields("flag", type_=base)
    client = FakeClient()
    module = dspy.Predict(sig)
    module.set_lm(client)
    module.fields.setdefault("flag", {})
    changed = sig.with_updated_fields("flag", type_=override)
    with pytest.raises(ValueError, match="preserve the answer space"):
        module(text="x", signature=changed)
    with pytest.raises(ValueError, match="preserve the answer space"):
        await module.acall(text="x", signature=changed)
    assert not client.calls


@pytest.mark.parametrize("kind", [str, int, float, Choice])
def test_unsupported_decision_outputs_fail_before_request(kind):
    client = FakeClient()
    module = dspy.Predict(signature().with_updated_fields("flag", type_=kind))
    module.set_lm(client)
    with pytest.raises(ValueError, match="Unsupported"):
        module(text="x")
    assert not client.calls


def test_mixed_program_discovery_demos_trace_and_persistence(tmp_path):
    from dspy.utils.dummies import DummyLM

    class Pipeline(dspy.Module):
        def __init__(self):
            sig = dspy.Signature(
                {
                    "text": (str, dspy.InputField()),
                    "flag": (Annotated[bool, Noul], dspy.OutputField(desc="Is it relevant?")),
                }
            )
            self.assess = dspy.Predict(sig)
            self.explain = dspy.Predict("flag: bool -> explanation: str")

        def forward(self, text):
            return self.explain(flag=self.assess(text=text).flag)

    program = Pipeline()
    program.assess.fields["flag"] = {"threshold": 0.9}
    assert program.named_predictors() == [("assess", program.assess), ("explain", program.explain)]
    trained = dspy.LabeledFewShot(k=1).compile(
        program, trainset=[dspy.Example(text="x", flag=False, explanation="Below threshold").with_inputs("text")]
    )
    assert len(trained.assess.demos) == len(trained.explain.demos) == 1
    trained.save(tmp_path / "mixed.json")
    restored = Pipeline()
    restored.load(tmp_path / "mixed.json")
    assert restored.assess.fields == {"flag": {"threshold": 0.9}}
    restored.assess.lm = FakeClient()
    with dspy.context(lm=DummyLM([{"explanation": "Below threshold"}]), trace=[]):
        assert restored(text="x").explanation == "Below threshold"
        assert [step[0] for step in dspy.settings.trace] == [restored.assess, restored.explain]
        assert dspy.settings.trace[1][1] == {"flag": False}
