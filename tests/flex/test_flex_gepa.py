from __future__ import annotations

import textwrap

import pytest

import dspy
from dspy.flex import Flex, flex
from dspy.teleprompt.flex_gepa import FlexAdapter, FlexGEPA, _format_failures
from dspy.utils.dummies import DummyLM

CANNED_PREDICTORS = textwrap.dedent(
    """
    PREDICTORS = {
        "echo": dspy.Predict("q -> a"),
    }
    """
).strip()

CANNED_FORWARD = textwrap.dedent(
    """
    def forward(self, q):
        out = self.echo(q=q)
        return dspy.Prediction(a=out.a)
    """
).strip()


def _codegen_lm():
    return DummyLM([{"predictors_src": CANNED_PREDICTORS, "forward_src": CANNED_FORWARD}])


def _make_flex(tmp_path):
    dspy.configure(lm=_codegen_lm())

    @flex(persist_to=str(tmp_path / "echo_flex.py"), flex_root=str(tmp_path))
    class Echo(dspy.Signature):
        """Echo."""

        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    return Echo()


def test_flex_adapter_build_program_returns_a_flex(tmp_path) -> None:
    student = _make_flex(tmp_path)
    adapter = FlexAdapter(student, metric_fn=lambda ex, pred, trace=None: 0.0)

    candidate = {"predictors_src": CANNED_PREDICTORS, "forward_src": CANNED_FORWARD}
    program = adapter.build_program(candidate)
    assert isinstance(program, Flex)
    assert program is not student
    assert program.forward_src is not None


def test_format_failures_handles_empty() -> None:
    assert "no failing examples" in _format_failures([])


def test_format_failures_renders_records() -> None:
    blob = _format_failures(
        [
            {"Inputs": {"q": "hi"}, "Generated Outputs": {"a": "there"}, "Feedback": "wrong"},
        ]
    )
    assert "Example 0" in blob
    assert "wrong" in blob


def test_flex_gepa_requires_a_flex_student() -> None:
    opt = FlexGEPA(
        metric=lambda ex, pred, trace=None: 1.0,
        reflection_lm=DummyLM([]),
        max_metric_calls=1,
    )

    class NotFlex(dspy.Module):
        def forward(self, **_):
            return dspy.Prediction(a="x")

    with pytest.raises(TypeError, match="dspy.Flex student"):
        opt.compile(NotFlex(), trainset=[dspy.Example(q="x", a="y").with_inputs("q")])  # type: ignore[arg-type]


def test_flex_gepa_requires_a_non_empty_trainset(tmp_path) -> None:
    student = _make_flex(tmp_path)
    opt = FlexGEPA(
        metric=lambda ex, pred, trace=None: 1.0,
        reflection_lm=DummyLM([]),
        max_metric_calls=1,
    )
    with pytest.raises(ValueError, match="trainset"):
        opt.compile(student, trainset=[])


def test_flex_gepa_exactly_one_budget_required() -> None:
    with pytest.raises(ValueError, match="Exactly one of"):
        FlexGEPA(metric=lambda *_: 0.0, reflection_lm=DummyLM([]))
    with pytest.raises(ValueError, match="Exactly one of"):
        FlexGEPA(metric=lambda *_: 0.0, reflection_lm=DummyLM([]), auto="light", max_full_evals=1)


def test_proposer_type_defaults_to_predict(tmp_path) -> None:
    student = _make_flex(tmp_path)
    adapter = FlexAdapter(student, metric_fn=lambda *_: 0.0, reflection_lm=DummyLM([]))
    assert adapter.proposer_type is dspy.Predict


def test_proposer_type_can_be_chain_of_thought(tmp_path) -> None:
    """The reflection LM can be wrapped in dspy.ChainOfThought instead of dspy.Predict."""
    student = _make_flex(tmp_path)

    revision_lm = DummyLM(
        [
            {"reasoning": "Add a verifier step.", "revised_source": 'PREDICTORS = {"echo2": dspy.Predict("q -> a")}'},
        ]
    )

    adapter = FlexAdapter(
        student,
        metric_fn=lambda *_: 0.0,
        reflection_lm=revision_lm,
        proposer_type=dspy.ChainOfThought,
    )

    candidate = {"predictors_src": CANNED_PREDICTORS, "forward_src": CANNED_FORWARD}
    reflective_dataset = {
        "predictors_src": [
            {"Inputs": {"q": "hi"}, "Generated Outputs": {"a": "x"}, "Feedback": "wrong"}
        ],
    }
    result = adapter.propose_new_texts(candidate, reflective_dataset, ["predictors_src"])
    assert "echo2" in result["predictors_src"]


def test_proposer_kwargs_are_forwarded_to_proposer_constructor(tmp_path) -> None:
    """proposer_kwargs is passed to the proposer_type's constructor."""
    student = _make_flex(tmp_path)

    constructed: list[dict] = []

    class _SpyProposer(dspy.Module):
        def __init__(self, signature, **kwargs):
            super().__init__()
            constructed.append(kwargs)
            self.inner = dspy.Predict(signature)

        def forward(self, **kw):
            return self.inner(**kw)

    revision_lm = DummyLM([{"revised_source": "PREDICTORS = {}"}])
    adapter = FlexAdapter(
        student,
        metric_fn=lambda *_: 0.0,
        reflection_lm=revision_lm,
        proposer_type=_SpyProposer,
        proposer_kwargs={"foo": 42, "bar": "x"},
    )
    candidate = {"predictors_src": CANNED_PREDICTORS, "forward_src": CANNED_FORWARD}
    reflective_dataset = {"predictors_src": [{"Inputs": {}, "Generated Outputs": {}, "Feedback": "x"}]}
    adapter.propose_new_texts(candidate, reflective_dataset, ["predictors_src"])
    assert constructed == [{"foo": 42, "bar": "x"}]
