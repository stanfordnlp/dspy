import copy
import json
from typing import Annotated, Literal, get_args

import pytest

import dspy
from dspy.experimental import Choice, Decide, Noul, Score, TypeSafe
from dspy.utils.callback import BaseCallback

Rating = Score["bad", "fair", "great"]
Label = Choice[(2, ""), ("other", "")]


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
            "flag": (Noul if rich else bool, dspy.OutputField(desc="Is it relevant?")),
            "rating": (Rating if rich else Annotated[float, Rating], dspy.OutputField(desc="Rate usefulness.")),
            "label": (Label if rich else Literal[2, "other"], dspy.OutputField()),
        },
        "Assess the document.",
    )


def decide(rich=False, client=None):
    return Decide(signature(rich), client=client)


def test_native_rich_equivalence_and_request_mapping():
    client = FakeClient()
    native = decide(client=client)
    rich = decide(True, client)
    native.fields["flag"]["threshold"] = rich.fields["flag"]["threshold"] = 0.7
    a, b = native(text="example"), rich(text="example")
    assert a.flag is b.flag.value is True
    assert a.rating == b.rating.value == pytest.approx(1.5)
    assert a.label == b.label.value == 2
    assert type(a.label) is type(b.label.value) is int
    assert type(a.rating) is float
    assert b.flag.confidence == pytest.approx(1 / 7)
    assert b.flag.probability == 0.8
    assert b.rating.confidence == 0.61
    assert b.rating.probabilities == {0: 0.1, 1: 0.3, 2: 0.6}
    assert b.label.confidence == 0.73  # Not max(probabilities).
    assert bool(b.flag) is True
    assert float(b.rating) == pytest.approx(1.5)
    assert client.calls[0] == client.calls[1]
    state, questions = client.calls[0]
    assert state == {
        "instructions": "Assess the document.",
        "input_fields": "1. `text` (str):",
        "inputs": {"text": "example"},
    }
    assert questions == {
        "flag": {
            "type": "noul",
            "instructions": "Is it relevant?",
        },
        "rating": {
            "type": "score",
            "criteria": ["bad", "fair", "great"],
            "instructions": "Rate usefulness.",
        },
        "label": {
            "type": "choice",
            "criteria": {"2": None, "other": None},
            "instructions": "Decide `label`.",
        },
    }


@pytest.mark.parametrize(
    "p,value,confidence", [(0, False, 1), (0.6, False, 0.2), (0.75, True, 0), (0.9, True, 0.2), (1, True, 1 / 3)]
)
def test_bool_boundary_and_confidence(p, value, confidence):
    module = Decide("text -> flag: dspy.experimental.Noul", client=FakeClient(p))
    module.fields["flag"]["threshold"] = 0.75
    result = module(text="x").flag
    assert result.value is value
    assert result.confidence == pytest.approx(confidence)
    assert bool(result) is value


@pytest.mark.parametrize("threshold,p,value", [(0, 0, True), (1, 0.99, False), (1, 1, True)])
def test_threshold_endpoints(threshold, p, value):
    module = Decide("text -> flag: bool", client=FakeClient(p))
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
    sig = dspy.Signature(
        {
            "text": (str, dspy.InputField()),
            "label": (Choice[tuple((v, "") for v in get_args(options))] if rich else options, dspy.OutputField()),
        }
    )
    result = Decide(sig, client=FakeClient(choice=selected))(text="x").label
    value = result.value if rich else result
    assert value == expected
    assert type(value) is type(expected)


def test_score_cuts_normalization_and_snapshot():
    client = FakeClient()
    module = decide(True, client)
    initial = module(text="x").rating
    module.fields["rating"]["cuts"] = [0.5, 1.6]
    changed = module(text="x").rating
    assert initial.value == pytest.approx(1.5)  # Computed from probabilities, not raw SDK score.
    assert changed.value == initial.value
    assert initial.level == 2
    assert changed.level == 1
    assert changed.confidence == initial.confidence == 0.61
    assert Rating.options == ("bad", "fair", "great")
    assert client.calls[0] == client.calls[1]
    # The provider only promises approximate unit mass; normalization is explicit.
    answers = client(*client.calls[0])
    answers["rating"]["probabilities"] = {0: 0.099, 1: 0.297, 2: 0.594}
    with dspy.context(system_one=lambda **_: answers):
        result = decide(True)(text="x").rating
    assert result.value == pytest.approx(1.5)
    assert sum(result.probabilities.values()) == pytest.approx(0.99)


@pytest.mark.parametrize("rich", [False, True])
def test_choice_weights_selection_evidence_and_request(rich):
    client = FakeClient()
    module = decide(rich, client)
    assert module.fields["label"]["weights"] == {"2": 1.0, "other": 1.0}
    before = module(text="x").label
    module.fields["label"]["weights"] = {"2": 0.1}  # 0.8 * 0.1 < 0.2 * 1.0
    after = module(text="x").label
    assert (before.value if rich else before) == 2
    assert (after.value if rich else after) == "other"
    assert client.calls[0] == client.calls[1]
    if rich:
        assert after.probabilities == before.probabilities == {"2": 0.8, "other": 0.2}
        assert after.confidence == before.confidence == 0.73
    module.fields["label"]["weights"] = {"2": 10, "other": 100}
    scaled = module(text="x").label
    assert (scaled.value if rich else scaled) == "other"
    assert Label.options == ((2, ""), ("other", ""))


@pytest.mark.parametrize("weights", [{}, {"2": 1}, {"2": 0.25}, {"2": 0}])
def test_choice_weights_defaults_ties_and_zero(weights):
    module = decide(True, FakeClient())
    module.fields["label"]["weights"] = weights
    result = module(text="x").label
    # 0.8 * 0.25 == 0.2: the provider's selection wins the tie.
    assert result.value == ("other" if weights.get("2") == 0 else 2)


def test_choice_weighted_tie_prefers_provider_over_declaration_order():
    module = decide(True, FakeClient(choice="other"))
    module.fields["label"]["weights"] = {"other": 0.25}
    assert module(text="x").label.value == "other"  # 2 is declared first, but both weighted scores are 0.2.


@pytest.mark.parametrize(
    "weights",
    [
        {"unknown": 1},
        {2: 1},
        {"2": -1},
        {"2": float("nan")},
        {"2": float("inf")},
        {"2": True},
        {"2": 0, "other": 0},
        [1, 2],
    ],
)
def test_reject_invalid_choice_weights_before_request(weights):
    client = FakeClient()
    module = decide(client=client)
    module.fields["label"]["weights"] = weights
    with pytest.raises(ValueError, match="Choice weights"):
        module(text="x")
    assert not client.calls


def test_choice_weights_preserve_literal_types_and_json_state(tmp_path):
    module = Decide("text -> label: Literal[True, 1, None]", client=FakeClient(choice="True"))
    module.fields["label"]["weights"] = {"True": 0, "1": 2, "None": 0}
    result = module(text="x").label
    assert type(result) is int and result == 1
    module.client = None
    module.save(tmp_path / "choice.json")
    restored = Decide(module.signature)
    restored.load(tmp_path / "choice.json")
    assert restored.fields == module.fields
    restored.client = FakeClient(choice="True")
    restored.fields["label"]["weights"] = {"True": 0, "1": 0, "None": 2}
    assert restored(text="x").label is None
    assert module.fields["label"]["weights"]["1"] == 2


def test_choice_weights_reject_zero_remaining_mass_and_tie_in_declaration_order():
    module = Decide("text -> label: Literal['a', 'b', 'c']")
    answers = {"label": {"choice": "c", "probabilities": {"c": 0.5, "b": 0.25, "a": 0.25}, "confidence": 0.7}}
    module.client = lambda **_: answers
    module.fields["label"]["weights"] = {"c": 0}
    assert module(text="x").label == "a"  # Neither tied option is the provider choice.
    answers["label"]["probabilities"] = {"c": 1, "b": 0, "a": 0}
    with pytest.raises(ValueError, match="no positive probability mass"):
        module(text="x")


@pytest.mark.parametrize(
    "sig,match",
    [
        ("text -> answer", "Unsupported"),
        ("text -> answer: int", "Unsupported"),
        ("text -> answer: float", "requires a rubric"),
        ("text -> answer: dspy.experimental.Choice", "Unsupported"),
        ("text -> answer: Literal[1, '1']", "ambiguous"),
    ],
)
def test_reject_unsupported(sig, match):
    with pytest.raises(ValueError, match=match):
        Decide(sig)


@pytest.mark.parametrize(
    "field,config",
    [
        ("flag", {"threshold": -0.1}),
        ("flag", {"threshold": float("nan")}),
        ("missing", {"threshold": 0.3}),
        ("rating", {"cuts": [0.5, 1.5], "weights": [0, 3]}),
        ("flag", {"threshold": 0.5, "cuts": [0.5]}),
        ("label", {"weights": {}, "threshold": 0.5}),
        ("flag", {"threshold": 0.5, "unknown": "typo"}),
        ("rating", {}),
        ("flag", None),
    ],
)
@pytest.mark.parametrize("operation", ["call", "save", "load"])
def test_reject_invalid_parameters(field, config, operation, tmp_path):
    client = FakeClient()
    module = decide(client=client)
    initial = copy.deepcopy(module.fields)
    if operation == "load":
        module.client = None
        state = module.dump_state()
        state["fields"][field] = config
        with pytest.raises(ValueError):
            module.load_state(state)
        assert module.fields == initial
        return
    module.fields[field] = config
    with pytest.raises(ValueError):
        if operation == "call":
            module(text="x")
        else:
            module.save(tmp_path / "invalid.json")
    assert not client.calls


@pytest.mark.parametrize("fields", [None, [], {}, {"flag": {"threshold": 0.5}}])
def test_invalid_field_map_load_is_atomic(fields):
    module = decide(client=TypeSafe("jev-original"))
    original = module.dump_state()
    invalid = copy.deepcopy(original)
    invalid["fields"] = fields
    invalid["signature"]["instructions"] = "Do not apply this failed load."
    invalid["client"]["model"] = "jev-replacement"
    with pytest.raises(ValueError, match="fields"):
        module.load_state(invalid)
    assert module.dump_state() == original


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


def test_composition_discovery_trace_callbacks():
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
    assert program.named_predictors() == []
    assert program.named_parameters() == [("decide", program.decide)]
    trace = []
    with dspy.context(system_one=client, trace=trace, max_trace_size=1, callbacks=[callback]):
        result = program(text="first")
        result = program(text="second")
        program.decide(text="untraced", _trace=False)
    assert trace == [(program.decide, {"text": "second"}, result)]
    assert result in callback.outputs
    assert all("examples" not in question["instructions"] for question in client.calls[0][1].values())


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


@pytest.mark.asyncio
async def test_shared_instructions_do_not_collide_with_input_names():
    class Sig(dspy.Signature):
        """Shared task context."""

        inputs: str = dspy.InputField()
        flag: bool = dspy.OutputField(desc="Is `inputs.inputs` actionable?")

    client = FakeClient()
    module = Decide(Sig, client=client)
    module(inputs="User content")
    await module.acall(inputs="User content")
    assert client.calls[0] == client.calls[1]
    state, questions = client.calls[0]
    assert state["instructions"] == "Shared task context."
    assert state["inputs"] == {"inputs": "User content"}
    assert questions["flag"]["instructions"] == "Is `inputs.inputs` actionable?"
    assert "Shared task context." not in json.dumps(questions)


def test_defaults_and_input_errors():
    class Sig(dspy.Signature):
        text: str = dspy.InputField(default="default")
        flag: bool = dspy.OutputField()

    client = FakeClient()
    module = Decide(Sig, client=client)
    assert module().flag is True
    assert client.calls[0][0]["inputs"] == {"text": "default"}
    with pytest.raises(ValueError, match="Unexpected"):
        module(other=1)
    with pytest.raises(ValueError, match="Missing"):
        decide(client=client)()


@pytest.mark.parametrize("rich", [False, True])
def test_configured_noul_criteria_and_persistence(tmp_path, rich):
    availability = Noul[(False, "Workaround available")]
    annotation = availability if rich else Annotated[bool, availability]
    sig = signature(rich).with_updated_fields("flag", type_=annotation)
    module = Decide(sig)
    assert module.get_criteria("flag") == {"false": "Workaround available"}
    assert module.fields["flag"] == {"threshold": 0.5}
    module.fields["flag"]["threshold"] = 0.8
    duplicate = module.deepcopy()
    duplicate.set_criteria("flag", {"true": "Override"})
    assert module.get_criteria("flag") == {"false": "Workaround available"}
    module.save(tmp_path / "state.json")
    restored = Decide(sig)
    restored.load(tmp_path / "state.json")
    module.save(tmp_path / "program", save_program=True)
    whole = dspy.load(tmp_path / "program", allow_pickle=True)
    client = FakeClient(probability=0.8)
    with dspy.context(system_one=client):
        for item in (module, restored, whole):
            result = item(text="x").flag
            assert (result.value if rich else result) is True
            if rich:
                assert isinstance(result, Noul)
                assert result.options == ((False, "Workaround available"),)
                assert result.confidence == 0
            assert client.calls[-1][1]["flag"]["criteria"] == {"false": "Workaround available"}
        native_sig = sig.with_updated_fields("flag", type_=Annotated[bool, availability])
        assert module(text="x", signature=native_sig).flag is True
        with pytest.raises(ValueError, match="preserve"):
            module(text="x", signature=sig.with_updated_fields("flag", type_=Noul[(False, "Different meaning")]))


def test_criteria_accessors_validate_copy_and_save_overrides(tmp_path):
    module = decide(True)
    assert module.get_criteria("flag") is None
    assert module.get_criteria("rating") == ["bad", "fair", "great"]
    assert module.get_criteria("label") == {"2": None, "other": None}
    criteria = {
        "flag": {"true": {"examples": ["Outage"]}},
        "rating": ["Poor", {"examples": ["Partial"]}, "Excellent"],
        "label": {"2": {"examples": ["Known"]}, "other": "Unknown"},
    }
    expected = copy.deepcopy(criteria)
    for field, value in criteria.items():
        module.set_criteria(field, value)
        nested = value[1] if isinstance(value, list) else next(iter(value.values()))
        nested["examples"].append("Not stored")
        value.clear()
        retrieved = module.get_criteria(field)
        assert retrieved == expected[field]
        nested = retrieved[1] if isinstance(retrieved, list) else next(iter(retrieved.values()))
        nested["examples"].append("Not stored")
        assert module.get_criteria(field) == expected[field]
    for field, invalid in [
        ("flag", {"yes": "Wrong key"}),
        ("flag", {"true": {"bad": float("nan")}}),
        ("rating", ["Wrong count"]),
        ("label", {"2": "Missing option"}),
    ]:
        with pytest.raises(ValueError):
            module.set_criteria(field, invalid)
        assert module.get_criteria(field) == expected[field]
    for field in ("text", "missing"):
        with pytest.raises(KeyError):
            module.get_criteria(field)
        with pytest.raises(KeyError):
            module.set_criteria(field, None)
    assert Decide(module.signature).get_criteria("flag") is None
    module.save(tmp_path / "criteria.json")
    restored = decide(True)
    restored.load(tmp_path / "criteria.json")
    client = FakeClient()
    with dspy.context(system_one=client):
        restored(text="x")
    for field, value in expected.items():
        assert restored.get_criteria(field) == value
        assert client.calls[-1][1][field]["criteria"] == value
    configured = Decide(signature(True).with_updated_fields("flag", type_=Noul[(True, "Default")]))
    configured.set_criteria("flag", None)
    assert configured.get_criteria("flag") is None
    with dspy.context(system_one=client):
        configured(text="x")
    assert client.calls[-1][1]["flag"]["criteria"] is None


def test_copy_and_json_state_preserve_config(tmp_path):
    module = decide(True)
    module.fields["flag"]["threshold"] = 0.7
    module.fields["rating"]["cuts"] = [0.4, 1.6]
    module.fields["label"]["weights"]["other"] = 2
    with dspy.context(system_one=FakeClient()):
        before = module(text="next")
    duplicate = module.deepcopy()
    duplicate.fields["rating"]["cuts"][1] = 1.8
    duplicate.fields["label"]["weights"]["other"] = 9
    duplicate.fields["flag"]["threshold"] = 0.3
    assert module.fields["rating"]["cuts"] == [0.4, 1.6]
    assert module.fields["label"]["weights"] == {"2": 1, "other": 2}
    assert module.fields["flag"]["threshold"] == 0.7
    assert decide(True).fields["rating"]["cuts"] == [0.5, 1.5]
    path = tmp_path / "decide.json"
    module.save(path)
    restored = decide(True)
    restored.load(path)
    assert restored.fields == module.fields
    assert set(module.dump_state()) == {"signature", "fields", "client"}
    with dspy.context(system_one=FakeClient()):
        assert restored(text="next").toDict() == before.toDict()


def test_json_state_preserves_instructions_criteria_and_cuts(tmp_path):
    module = decide(True)
    module.signature = module.signature.with_instructions("Assess impact, not writing style.")
    instructions = {
        "flag": {"question": "Is action needed?", "focus": ["impact", "timing"]},
        "rating": ["Assess severity.", {"ignore": "tone"}],
        "label": "Choose the applicable category.",
    }
    criteria = {
        "flag": {
            "true": {"what": "Needs action", "examples": [{"text": "Production is down"}]},
            "false": None,
        },
        "rating": [
            {"what": "bad", "not_for": "cosmetic", "examples": ["Data loss"]},
            "fair",
            {"what": "great", "custom": {"tags": ["resolved", "verified"]}},
        ],
        "label": {"2": {"what": "Known category", "examples": []}, "other": None},
    }
    for name in module.fields:
        module.fields[name].update(
            instructions=copy.deepcopy(instructions[name]), criteria=copy.deepcopy(criteria[name])
        )
    module.fields["flag"]["threshold"] = 0.7
    module.fields["rating"]["cuts"] = [0.4, 1.6]
    module.fields["label"]["weights"] = {"2": 0.25, "other": 1.5}
    path = tmp_path / "structured-decide.json"
    module.save(path)

    # Inspect the file as well as the loaded object: whole-program pickle or
    # reconstructing defaults must not conceal missing optimized state.
    state = json.loads(path.read_text())
    expected = {
        "flag": {"instructions": instructions["flag"], "criteria": criteria["flag"], "threshold": 0.7},
        "rating": {"instructions": instructions["rating"], "criteria": criteria["rating"], "cuts": [0.4, 1.6]},
        "label": {
            "instructions": instructions["label"],
            "criteria": criteria["label"],
            "weights": {"2": 0.25, "other": 1.5},
        },
    }
    assert state["fields"] == expected
    assert set(state) == {"signature", "fields", "client", "metadata"}
    restored = decide(True)
    restored.load(path)
    assert restored.signature.instructions == "Assess impact, not writing style."
    assert restored.fields == expected
    client = FakeClient()
    with dspy.context(system_one=client):
        before = module(text="x")
        after = restored(text="x")
    assert before.toDict() == after.toDict()
    assert client.calls[0] == client.calls[1]
    for name in instructions:
        question = client.calls[0][1][name]
        assert question["instructions"] == instructions[name]
        assert client.calls[0][0]["instructions"] == "Assess impact, not writing style."
        assert question["criteria"] == criteria[name]
        assert not {"threshold", "cuts", "weights"} & question.keys()
    restored.fields["rating"]["criteria"][0]["examples"].append("Another example")
    restored.fields["flag"]["instructions"]["focus"].append("urgency")
    restored.fields["rating"]["cuts"][0] = 0.2
    assert module.fields == expected


def test_field_overrides_preserve_absent_vs_null_and_signature_defaults(tmp_path):
    module = decide()
    module.fields["flag"].update(instructions=None, criteria=None)
    path = tmp_path / "overrides.json"
    module.save(path)
    restored = decide()
    restored.load(path)
    assert restored.fields["flag"] == {"threshold": 0.5, "instructions": None, "criteria": None}
    assert restored.fields["rating"] == {"cuts": [0.5, 1.5]}
    override = restored.signature.with_updated_fields("rating", desc="Updated rubric question.")
    client = FakeClient()
    with dspy.context(system_one=client):
        restored(text="x", signature=override)
    questions = client.calls[0][1]
    assert questions["flag"] == {"type": "noul", "instructions": None, "criteria": None}
    assert questions["rating"]["instructions"] == "Updated rubric question."
    assert questions["rating"]["criteria"] == ["bad", "fair", "great"]


@pytest.mark.parametrize("cuts,level", [([0.5, 1.5], 2), ([0.5, 1.6], 1), ([0.5, 1.4], 2)])
def test_score_cuts_select_level_without_changing_continuous_value(cuts, level):
    module = decide(True, FakeClient())
    module.fields["rating"]["cuts"] = cuts
    result = module(text="x").rating
    # Raw index expectation is 0*.1 + 1*.3 + 2*.6 = 1.5.
    assert result.level == level
    assert result.value == pytest.approx(1.5)


@pytest.mark.parametrize("cuts", [[0, 1.5], [0.5, 2], [1.5, 0.5], [0.5, 0.5], [0.5], [True, 1.5], [0.5, float("nan")]])
def test_invalid_cuts_rejected_before_inference(cuts):
    client = FakeClient()
    module = decide(True, client)
    module.fields["rating"]["cuts"] = cuts
    with pytest.raises(ValueError, match="cuts"):
        module(text="x")
    assert client.calls == []


@pytest.mark.parametrize("entry", [{"nested": {1: "integer key"}}, {"nested": float("nan")}, {"nested": {"set"}}, 42])
def test_instruction_json_validation_is_not_silently_coercive(entry):
    module = decide()
    module.fields["flag"]["instructions"] = entry
    with pytest.raises(ValueError, match="instructions"):
        module.dump_state()


@pytest.mark.parametrize("setting", ["instructions", "criteria"])
def test_json_validator_reuse_still_checks_nested_mutations(setting):
    from unittest.mock import patch

    from pydantic import JsonValue, TypeAdapter

    client = FakeClient()
    module = decide(client=client)
    examples = ["Relevant"]
    module.fields["flag"][setting] = {"true": {"examples": examples}}
    with patch("dspy.predict.decide.TypeAdapter", wraps=TypeAdapter) as constructors:
        assert module(text="x").flag is True
        examples.append("Another example")
        assert module(text="x").flag is True
        assert client.calls[-1][1]["flag"][setting]["true"]["examples"] == examples
        for invalid in ({"not JSON"}, float("nan")):
            examples.append(invalid)
            with pytest.raises(ValueError, match=setting):
                module(text="x")
            examples.pop()
        assert len(client.calls) == 2
        assert all(call.args[0] is not JsonValue for call in constructors.call_args_list)


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
def test_criteria_persistence_rejects_incompatible_field_shape(tmp_path, rich, operation, field, invalid):
    module = decide(rich)
    criteria = {
        "flag": {"true": "Relevant", "false": None},
        "rating": ["bad", "fair", "great"],
        "label": {"2": "Known", "other": None},
        field: invalid,
    }
    if operation == "save":
        for name, entry in criteria.items():
            module.fields[name]["criteria"] = entry
        with pytest.raises(ValueError, match="criteria"):
            module.save(tmp_path / "invalid-criteria.json")
    else:
        state = module.dump_state()
        for name, entry in criteria.items():
            state["fields"][name]["criteria"] = entry
        with pytest.raises(ValueError, match="criteria"):
            module.load_state(state)
        assert all("criteria" not in config for config in module.fields.values())


@pytest.mark.asyncio
async def test_reject_undeclared_demos_input_before_inference():
    client = FakeClient()
    module = decide()
    demos = [dspy.Example(text="example", flag=False)]
    with dspy.context(system_one=client):
        for value in ([], demos):
            with pytest.raises(ValueError, match=r"Unexpected Decide inputs.*demos"):
                module(text="x", demos=value)
            with pytest.raises(ValueError, match=r"Unexpected Decide inputs.*demos"):
                await module.acall(text="x", demos=value)
        assert client.calls == []
        assert not hasattr(module, "demos")
        assert module(text="x").flag is True


def test_mixed_program_discovery_optimizer_and_persistence(tmp_path):
    from dspy.utils.dummies import DummyLM

    class Pipeline(dspy.Module):
        def __init__(self):
            self.nodes = {"decision": Decide("text -> flag: bool")}
            self.explain = dspy.Predict("flag: bool -> explanation: str")

        def forward(self, text):
            return self.explain(flag=self.nodes["decision"](text=text).flag)

    program = Pipeline()
    decision = program.nodes["decision"]
    decision.fields["flag"]["threshold"] = 0.9
    assert not isinstance(decision, dspy.Predict)
    assert decision.named_parameters() == [("self", decision)]
    assert not hasattr(decision, "demos")
    assert program.named_parameters() == [("nodes['decision']", decision), ("explain", program.explain)]
    assert program.named_predictors() == [("explain", program.explain)]
    trained = dspy.LabeledFewShot(k=1).compile(
        program, trainset=[dspy.Example(flag=False, explanation="Below threshold").with_inputs("flag")]
    )
    assert len(trained.explain.demos) == 1
    assert not hasattr(trained.nodes["decision"], "demos")
    assert trained.nodes["decision"].fields == {"flag": {"threshold": 0.9}}
    path = tmp_path / "mixed.json"
    trained.save(path)
    restored = Pipeline()
    restored.load(path)
    assert restored.nodes["decision"].fields == {"flag": {"threshold": 0.9}}
    assert len(restored.explain.demos) == 1
    trace = []
    with dspy.context(system_one=FakeClient(), lm=DummyLM([{"explanation": "Below threshold"}]), trace=trace):
        assert restored(text="x").explanation == "Below threshold"
    assert [step[0] for step in trace] == [restored.nodes["decision"], restored.explain]
    assert trace[1][1] == {"flag": False}


def test_reset_copy_preserves_configuration_without_aliasing():
    module = decide(True, client=TypeSafe("jev-test"))
    module.fields["flag"]["threshold"] = 0.9
    module.fields["rating"]["cuts"] = [0.4, 1.6]
    module.fields["label"]["weights"] = {"other": 3}
    reset = module.reset_copy()
    expected = {"flag": {"threshold": 0.9}, "rating": {"cuts": [0.4, 1.6]}, "label": {"weights": {"other": 3}}}
    assert reset.fields == expected
    assert reset.signature is module.signature
    assert reset.client.model == "jev-test"
    reset.fields["flag"]["threshold"] = 0.5
    reset.fields["rating"]["cuts"][1] = 1.8
    reset.fields["label"]["weights"]["other"] = 1
    assert module.fields == expected


def test_explicit_client_state_omits_key_and_gates_endpoint(tmp_path):
    module = decide(client=TypeSafe("jev-test", api_key="test-credential", base_url="https://example.test"))
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
    module = decide(True, client=TypeSafe("jev-test"))
    module.fields["flag"]["threshold"] = 0.7
    module.fields["label"]["weights"] = {"2": 0.1}
    module.save(tmp_path / "program", save_program=True)
    restored = dspy.load(tmp_path / "program", allow_pickle=True)
    assert restored.fields == module.fields
    assert restored.client.model == "jev-test"
    restored.client = FakeClient()
    assert restored(text="x").flag.confidence == pytest.approx(1 / 7)
    assert restored(text="x").label.value == "other"


def test_thresholds_are_per_field_and_results_are_snapshots():
    module = Decide("text -> a: bool, b: dspy.experimental.Noul", client=FakeClient(0.6))
    assert module.fields == {"a": {"threshold": 0.5}, "b": {"threshold": 0.5}}
    module.fields["b"]["threshold"] = 0.8
    before = module(text="x")
    assert before.a is True and before.b.value is False
    module.fields["b"]["threshold"] = 0.4
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
        assert (result.rating.value if rich else result.rating) == pytest.approx(1.5)


@pytest.mark.parametrize(
    "base,override",
    [
        (Score["low", "high"], Score["low", "medium", "high"]),
        (Rating, Score["great", "fair", "bad"]),
        (Literal[True], Literal[1]),
        (Literal["a", "b"], Literal["a", "c"]),
        (bool, Literal[False, True]),
    ],
)
@pytest.mark.asyncio
async def test_signature_override_rejects_changed_answer_space_before_request(base, override):
    def sig(kind):
        return dspy.Signature({"text": (str, dspy.InputField()), "answer": (kind, dspy.OutputField())})

    client = FakeClient()
    module = Decide(sig(base), client=client)
    with pytest.raises(ValueError, match="signature override"):
        module(text="x", signature=sig(override))
    with pytest.raises(ValueError, match="signature override"):
        await module.acall(text="x", signature=sig(override))
    assert not client.calls


def test_signature_override_rejects_renamed_output():
    client = FakeClient()
    module = Decide("text -> flag: bool", client=client)
    with pytest.raises(ValueError, match="signature override"):
        module(text="x", signature="text -> renamed: bool")
    assert not client.calls


def test_signature_override_preserves_parameters_and_accepts_prompt_changes():
    client = FakeClient()
    module = decide(True, client)
    module.fields["flag"]["threshold"] = 0.9
    module.fields["rating"]["cuts"] = [0.5, 1.6]
    override = module.signature.with_instructions("Assess carefully.").with_updated_fields(
        "rating", desc="New question."
    )
    result = module(text="x", signature=override)
    assert result.flag.value is False
    assert result.rating.value == pytest.approx(1.5)
    assert result.rating.level == 1
    assert client.calls[0][0]["instructions"] == "Assess carefully."
    assert client.calls[0][1]["rating"]["instructions"] == "New question."
    assert module.signature.instructions == "Assess the document."


def test_signature_override_accepts_equivalent_types_and_native_form():
    # A separately built Pydantic class need not have the same identity (e.g. after cache eviction).
    from pydantic import create_model

    equivalent = create_model("EquivalentRating", __base__=Rating)
    module = decide(True, FakeClient())
    for annotation in (equivalent, Annotated[float, equivalent]):
        override = module.signature.with_updated_fields("rating", type_=annotation)
        result = module(text="x", signature=override).rating
        assert (result if type(result) is float else result.value) == pytest.approx(1.5)
