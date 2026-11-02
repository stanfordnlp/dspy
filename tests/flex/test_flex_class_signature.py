"""Class-based signatures through dspy.Flex: the Flex's own signature, and ones defined by sandboxed code."""

import shutil
from typing import Literal

import pydantic
import pytest

import dspy
from dspy.primitives import facade
from dspy.primitives.code_interpreter import CodeExecutionError
from dspy.utils.dummies import DummyLM

INTERPRETERS = [
    pytest.param(
        dspy.PythonInterpreter,
        marks=pytest.mark.skipif(shutil.which("deno") is None, reason="Deno is not installed"),
        id="deno",
    ),
    pytest.param(dspy.LocalInterpreter, id="local"),
]


class Verdict(pydantic.BaseModel):
    label: str
    score: float


class Grade(dspy.Signature):
    """Grade the essay against the rubric."""

    essay: str = dspy.InputField(desc="the student's essay, verbatim")
    tone: Literal["formal", "casual"] = dspy.OutputField(desc="register of the essay")
    points: int = dspy.OutputField(desc="total points awarded", ge=0, le=100)
    verdict: Verdict = dspy.OutputField(desc="structured verdict")


ANSWER = {"tone": "formal", "points": 87, "verdict": '{"label": "pass", "score": 0.87}'}

# Sandboxed code with its own class-based signature: inherited fields, a typing annotation, a quoted custom type.
SANDBOX_SIGNATURE_SRC = '''
from typing import Literal


class EssayInput(dspy.Signature):
    essay: str = dspy.InputField(desc="the student's essay, verbatim")


class SubGrade(EssayInput):
    """Grade the essay against the rubric."""

    tone: Literal["formal", "casual"] = dspy.OutputField(desc="register of the essay")
    points: int = dspy.OutputField(desc="total points awarded", ge=0, le=100)
    verdict: "Verdict" = dspy.OutputField(desc="structured verdict")


class GradeModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.grade = dspy.Predict(SubGrade)

    def forward(self, essay):
        r = self.grade(essay=essay)
        return dspy.Prediction(tone=r.tone, points=r.points, verdict=r.verdict)
'''


@pytest.fixture
def built_signatures(monkeypatch):
    """Signatures of the predictors the host builds for sandboxed code."""
    signatures = []
    build = facade.FacadeInvocation._build_predictor

    def spy(self, kind, signature, kwargs):
        predictor = build(self, kind, signature, kwargs)
        signatures.append(predictor.signature)
        return predictor

    monkeypatch.setattr(facade.FacadeInvocation, "_build_predictor", spy)
    return signatures


def assert_same_fields(actual, expected):
    assert actual.instructions == expected.instructions
    assert list(actual.input_fields) == list(expected.input_fields)
    assert list(actual.output_fields) == list(expected.output_fields)
    for name, field in expected.fields.items():
        assert actual.fields[name].annotation == field.annotation
        assert actual.fields[name].metadata == field.metadata
        assert actual.fields[name].json_schema_extra == field.json_schema_extra


def assert_graded(lm, prediction):
    assert prediction.tone == "formal"
    assert prediction.points == 87
    assert prediction.verdict == Verdict(label="pass", score=0.87)
    prompt = lm.history[-1]["messages"][0]["content"]
    assert "the student's essay, verbatim" in prompt
    assert "greater than or equal to: 0" in prompt


@pytest.mark.parametrize("factory", INTERPRETERS)
def test_flex_class_signature_keeps_field_metadata(factory, built_signatures):
    flex = dspy.Flex(Grade, interpreter_factory=factory)
    lm = DummyLM([ANSWER])
    with dspy.context(lm=lm):
        prediction = flex(essay="An essay.")

    assert_graded(lm, prediction)
    assert_same_fields(built_signatures[-1], Grade)


@pytest.mark.parametrize("factory", INTERPRETERS)
def test_sandboxed_class_signature(factory, built_signatures):
    flex = dspy.Flex(Grade, interpreter_factory=factory)
    flex.load_state({"module_src": SANDBOX_SIGNATURE_SRC})
    lm = DummyLM([ANSWER])
    with dspy.context(lm=lm):
        prediction = flex(essay="An essay.")

    assert_graded(lm, prediction)
    assert_same_fields(built_signatures[-1], Grade)


@pytest.mark.parametrize("factory", INTERPRETERS)
def test_sandboxed_class_signature_requires_input_or_output_field(factory):
    flex = dspy.Flex(Grade, interpreter_factory=factory)
    flex.load_state({"module_src": SANDBOX_SIGNATURE_SRC.replace(' = dspy.InputField(desc="the student\'s essay, verbatim")', "")})

    with dspy.context(lm=DummyLM([ANSWER])):
        with pytest.raises(CodeExecutionError, match=r"\['essay'\] in SubGrade must be declared with InputField"):
            flex(essay="An essay.")


def test_flex_baseline_keeps_string_signature_without_field_metadata():
    class Answer(dspy.Signature):
        """Answer the question."""

        question: str = dspy.InputField()
        answer: int = dspy.OutputField()

    flex = dspy.Flex(Answer, interpreter_factory=dspy.LocalInterpreter)
    assert "dspy.Predict(dspy.Signature('question: str -> answer: int', 'Answer the question.'))" in flex.module_src
    assert "class Answer(dspy.Signature)" not in flex.module_src
