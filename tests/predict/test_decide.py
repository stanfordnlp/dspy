import copy
from typing import Annotated, Literal, get_args

import pytest

import dspy
from dspy.experimental import Choice, Decide, Noul, Score, TypeSafe
from dspy.utils.callback import BaseCallback

Rating = Score[(-2, "bad"), (3, "fair"), (10, "great")]
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
    module = Decide("text -> flag: dspy.experimental.Noul", client=FakeClient(p))
    module.thresholds["flag"] = 0.75
    result = module(text="x").flag
    assert result.value is value
    assert result.confidence == pytest.approx(confidence)
    assert bool(result) is value


@pytest.mark.parametrize("threshold,p,value", [(0, 0, True), (1, 0.99, False), (1, 1, True)])
def test_threshold_endpoints(threshold, p, value):
    module = Decide("text -> flag: bool", client=FakeClient(p))
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
            "label": (Choice[tuple((v, "") for v in get_args(options))] if rich else options, dspy.OutputField()),
        }
    )
    result = Decide(sig, client=FakeClient(choice=selected))(text="x").label
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


@pytest.mark.parametrize("rich", [False, True])
def test_choice_weights_selection_evidence_and_request(rich):
    client = FakeClient()
    module = decide(rich, client)
    assert module.weights["label"] == {"2": 1.0, "other": 1.0}
    before = module(text="x").label
    module.weights["label"] = {"2": 0.1}  # 0.8 * 0.1 < 0.2 * 1.0
    after = module(text="x").label
    assert (before.value if rich else before) == 2
    assert (after.value if rich else after) == "other"
    assert client.calls[0] == client.calls[1]
    if rich:
        assert after.probabilities == before.probabilities == {"2": 0.8, "other": 0.2}
        assert after.confidence == before.confidence == 0.73
    module.weights["label"] = {"2": 10, "other": 100}
    scaled = module(text="x").label
    assert (scaled.value if rich else scaled) == "other"
    assert Label.options == ((2, ""), ("other", ""))


@pytest.mark.parametrize("weights", [{}, {"2": 1}, {"2": 0.25}, {"2": 0}])
def test_choice_weights_defaults_ties_and_zero(weights):
    module = decide(True, FakeClient())
    module.weights["label"] = weights
    result = module(text="x").label
    # 0.8 * 0.25 == 0.2: the provider's selection wins the tie.
    assert result.value == ("other" if weights.get("2") == 0 else 2)


def test_choice_weighted_tie_prefers_provider_over_declaration_order():
    module = decide(True, FakeClient(choice="other"))
    module.weights["label"] = {"other": 0.25}
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
    module.weights["label"] = weights
    with pytest.raises(ValueError, match="Choice weights"):
        module(text="x")
    assert not client.calls


def test_choice_weights_preserve_literal_types_and_json_state(tmp_path):
    module = Decide("text -> label: Literal[True, 1, None]", client=FakeClient(choice="True"))
    module.weights["label"] = {"True": 0, "1": 2, "None": 0}
    result = module(text="x").label
    assert type(result) is int and result == 1
    module.client = None
    module.save(tmp_path / "choice.json")
    restored = Decide(module.signature)
    restored.load(tmp_path / "choice.json")
    assert restored.weights == module.weights
    restored.client = FakeClient(choice="True")
    restored.weights["label"] = {"True": 0, "1": 0, "None": 2}
    assert restored(text="x").label is None
    assert module.weights["label"]["1"] == 2


def test_choice_weights_reject_zero_remaining_mass_and_tie_in_declaration_order():
    module = Decide("text -> label: Literal['a', 'b', 'c']")
    answers = {"label": {"choice": "c", "probabilities": {"c": 0.5, "b": 0.25, "a": 0.25}, "confidence": 0.7}}
    module.client = lambda **_: answers
    module.weights["label"] = {"c": 0}
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
    "attribute,value",
    [
        ("thresholds", {"flag": -0.1}),
        ("thresholds", {"flag": float("nan")}),
        ("thresholds", {"missing": 0.3}),
        ("weights", {"rating": [0, 3]}),
        ("weights", {"rating": [-2, float("inf"), 10]}),
        ("weights", {"rating": [-2, float("nan"), 10]}),
        ("weights", {"rating": [-2, True, 10]}),
        ("weights", {"rating": [-2, 3, 11]}),
        ("weights", {"rating": [0, 9, 2]}),
        ("weights", {"rating": [-3, 2, 10]}),
    ],
)
def test_reject_invalid_parameters(attribute, value):
    client = FakeClient()
    module = decide(client=client)
    if attribute == "weights":
        module.weights.update(value)
    else:
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


def test_defaults_and_input_errors():
    class Sig(dspy.Signature):
        text: str = dspy.InputField(default="default")
        flag: bool = dspy.OutputField()

    client = FakeClient()
    module = Decide(Sig, client=client)
    assert module().flag is True
    assert client.calls[0][0] == {"text": "default"}
    with pytest.raises(ValueError, match="Unexpected"):
        module(other=1)
    with pytest.raises(ValueError, match="Missing"):
        decide(client=client)()


def test_copy_and_json_state_preserve_config(tmp_path):
    module = decide(True)
    module.thresholds["flag"] = 0.7
    module.weights["rating"] = [-2, 1, 10]
    module.weights["label"]["other"] = 2
    with dspy.context(system_one=FakeClient()):
        before = module(text="next")
    duplicate = module.deepcopy()
    duplicate.weights["rating"][1] = 4
    duplicate.weights["label"]["other"] = 9
    duplicate.thresholds["flag"] = 0.3
    assert module.weights["rating"] == [-2, 1, 10]
    assert module.weights["label"] == {"2": 1, "other": 2}
    assert module.thresholds["flag"] == 0.7
    assert decide(True).weights["rating"] == [-2, 3, 10]
    path = tmp_path / "decide.json"
    module.save(path)
    restored = decide(True)
    restored.load(path)
    assert restored.weights == module.weights
    assert restored.thresholds == module.thresholds
    assert set(module.dump_state()) == {"signature", "thresholds", "weights", "client"}
    with dspy.context(system_one=FakeClient()):
        assert restored(text="next").toDict() == before.toDict()


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
    decision.thresholds["flag"] = 0.9
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
    assert trained.nodes["decision"].thresholds == {"flag": 0.9}
    path = tmp_path / "mixed.json"
    trained.save(path)
    restored = Pipeline()
    restored.load(path)
    assert restored.nodes["decision"].thresholds == {"flag": 0.9}
    assert len(restored.explain.demos) == 1
    trace = []
    with dspy.context(system_one=FakeClient(), lm=DummyLM([{"explanation": "Below threshold"}]), trace=trace):
        assert restored(text="x").explanation == "Below threshold"
    assert [step[0] for step in trace] == [restored.nodes["decision"], restored.explain]
    assert trace[1][1] == {"flag": False}


def test_reset_copy_preserves_configuration_without_aliasing():
    module = decide(True, client=TypeSafe("jev-test"))
    module.thresholds["flag"] = 0.9
    module.weights["rating"] = [-2, 0, 8]
    module.weights["label"] = {"other": 3}
    reset = module.reset_copy()
    assert reset.thresholds == {"flag": 0.9}
    assert reset.weights == {"rating": [-2, 0, 8], "label": {"other": 3}}
    assert reset.signature is module.signature
    assert reset.client.model == "jev-test"
    reset.thresholds["flag"] = 0.5
    reset.weights["rating"][1] = 1
    reset.weights["label"]["other"] = 1
    assert module.thresholds == {"flag": 0.9}
    assert module.weights == {"rating": [-2, 0, 8], "label": {"other": 3}}


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
    module.thresholds["flag"] = 0.7
    module.weights["label"] = {"2": 0.1}
    module.save(tmp_path / "program", save_program=True)
    restored = dspy.load(tmp_path / "program", allow_pickle=True)
    assert restored.weights == module.weights
    assert restored.thresholds == module.thresholds
    assert restored.client.model == "jev-test"
    restored.client = FakeClient()
    assert restored(text="x").flag.confidence == pytest.approx(1 / 7)
    assert restored(text="x").label.value == "other"


def test_thresholds_are_per_field_and_results_are_snapshots():
    module = Decide("text -> a: bool, b: dspy.experimental.Noul", client=FakeClient(0.6))
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


@pytest.mark.parametrize(
    "base,override",
    [
        (Score[(0, "low"), (1, "high")], Score[(0, "low"), (10, "high")]),
        (Rating, Score[(-2, "bad"), (4, "fair"), (10, "great")]),
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
    module.thresholds["flag"] = 0.9
    module.weights["rating"] = [0, 2, 8]
    override = module.signature.with_instructions("Assess carefully.").with_updated_fields(
        "rating", desc="New question."
    )
    result = module(text="x", signature=override)
    assert result.flag.value is False
    assert result.rating.value == pytest.approx(5.4)
    assert client.calls[0][1]["rating"]["instructions"]["task"] == "Assess carefully."
    assert client.calls[0][1]["rating"]["instructions"]["question"] == "New question."
    assert module.signature.instructions == "Assess the document."


def test_signature_override_accepts_equivalent_types_and_native_form():
    # A separately built Pydantic class need not have the same identity (e.g. after cache eviction).
    from pydantic import create_model

    equivalent = create_model("EquivalentRating", __base__=Rating)
    module = decide(True, FakeClient())
    for annotation in (equivalent, Annotated[float, equivalent]):
        override = module.signature.with_updated_fields("rating", type_=annotation)
        result = module(text="x", signature=override).rating
        assert (result if type(result) is float else result.value) == pytest.approx(6.7)
