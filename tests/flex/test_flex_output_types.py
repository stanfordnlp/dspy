"""dspy.Flex honours its signature's declared output types.

Everything the generated code returns crosses the sandbox boundary as JSON, so a field declared
`int`, `list[str]`, or a pydantic model would otherwise come back to the caller as a bare string or
dict — unlike every other module over the same signature. Two halves are covered here:

* the *baseline source* names custom types (`text: str -> person: Person`), so the sub-predictor is
  actually asked for a Person and the host can resolve that name when it rebuilds the signature;
* the *return path* parses what comes back against the annotation, so a Flex returns the same types
  as a dspy.Predict would — and a candidate that returns the wrong shape raises a clear error
  instead of handing the metric a Prediction that breaks later.

That error is what GEPA catches: a code candidate whose output doesn't match the declared type is
dropped from trace capture and scored at the failure score, in place, like any other runtime fault.
"""

from __future__ import annotations

import enum
import json
import shutil
import textwrap
from typing import Callable

import pydantic
import pytest

import dspy
from dspy.flex import Flex
from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.teleprompt.gepa.gepa_flex_utils import make_code_key
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
from dspy.utils.dummies import DummyLM
from tests.flex import _offstack_types  # imported as a module: `Contact` is never a name here
from tests.mock_interpreter import MockInterpreter

deno_required = pytest.mark.skipif(shutil.which("deno") is None, reason="Deno is not installed")


class Person(pydantic.BaseModel):
    name: str
    age: int


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


class Extract(dspy.Signature):
    """Extract the person described by the text."""

    text: str = dspy.InputField()
    person: Person = dspy.OutputField()


class Roster(dspy.Signature):
    text: str = dspy.InputField()
    people: list[Person] = dspy.OutputField()
    color: Color = dspy.OutputField()
    count: int = dspy.OutputField()


def _sandbox() -> dspy.PythonInterpreter:
    return dspy.PythonInterpreter()


# =============================================================================
# The baseline source names the declared types (no Deno)
# =============================================================================


def test_baseline_signature_string_names_custom_types() -> None:
    # The generated code can only refer to a type by name, so the rendered signature string has to
    # carry it: without `person: Person` the sub-predictor is asked for a plain string and the LM is
    # never shown the model's schema.
    program = Flex(Extract, interpreter_factory=MockInterpreter)
    assert "text: str -> person: Person" in program.module_src


def test_baseline_signature_string_names_generic_and_enum_types() -> None:
    program = Flex(Roster, interpreter_factory=MockInterpreter)
    assert "people: list[Person]" in program.module_src
    assert "color: Color" in program.module_src
    assert "count: int" in program.module_src


def test_baseline_omits_a_type_the_signature_parser_cannot_read_back() -> None:
    # An annotation that cannot round-trip through a signature string is emitted untyped rather
    # than rendered into source that would fail to bind.
    class Unrenderable(dspy.Signature):
        text: str = dspy.InputField()
        fn: Callable[[int], int] = dspy.OutputField()

    program = Flex(Unrenderable, interpreter_factory=MockInterpreter)
    assert "'text: str -> fn'" in program.module_src  # the field is there, with no type annotation


@deno_required
def test_custom_type_resolves_when_it_is_not_on_the_calling_stack() -> None:
    """The regression `bridge._make_signature` exists for.

    dspy resolves a type named in a signature string by walking caller frames for the bare name,
    so `from myapp.models import Contact` accidentally works while `import myapp.models as models`
    does not — the same program, written two ways, with only one of them binding. `Contact` lives
    in `_offstack_types` and is reached only through the module object, so nothing on the stack
    holds the name and the explicit `custom_types` handoff is what has to resolve it.
    """
    source = textwrap.dedent(
        """
        class M(dspy.Module):
            def __init__(self):
                super().__init__()
                self.p = dspy.Predict("text: str -> contact: Contact")

            def forward(self, **inputs):
                return dspy.Prediction(contact=self.p(text=inputs["text"]).contact)
        """
    ).strip()

    with Flex(_offstack_types.ExtractContact, interpreter_factory=_sandbox) as program:
        program._bind_code(source)  # raised `Unknown name: Contact` before the handoff
        assert program.p.signature.output_fields["contact"].annotation is _offstack_types.Contact

        dspy.configure(lm=DummyLM([{"contact": json.dumps({"name": "Ada", "age": 36})}] * 2))
        out = program(text="Ada is 36")

    assert out.contact == _offstack_types.Contact(name="Ada", age=36)


@deno_required
def test_sub_signature_naming_a_custom_type_resolves_on_the_host() -> None:
    # The sandbox sends a signature as text; the host resolves `Person` from the Flex's own
    # signature, since dspy's usual caller-frame lookup runs inside dspy and cannot see it.
    source = textwrap.dedent(
        """
        class M(dspy.Module):
            def __init__(self):
                super().__init__()
                self.p = dspy.Predict("text: str -> person: Person")

            def forward(self, **inputs):
                return dspy.Prediction(person=self.p(text=inputs["text"]).person)
        """
    ).strip()
    with Flex(Extract, interpreter_factory=_sandbox) as program:
        program._bind_code(source)
        assert program.p.signature.output_fields["person"].annotation is Person


# =============================================================================
# The return path parses against the declared types (Deno)
# =============================================================================


@deno_required
def test_pydantic_output_comes_back_as_the_model_like_predict_does() -> None:
    payload = json.dumps({"name": "Ada", "age": 36})
    dspy.configure(lm=DummyLM([{"person": payload}] * 2))

    baseline = dspy.Predict(Extract)(text="Ada is 36")
    with Flex(Extract, interpreter_factory=_sandbox) as program:
        flexed = program(text="Ada is 36")

    # The point of the parse: a Flex is substitutable for a Predict over the same signature.
    assert isinstance(baseline.person, Person)
    assert isinstance(flexed.person, Person)
    assert flexed.person == baseline.person == Person(name="Ada", age=36)


@deno_required
def test_generic_enum_and_scalar_outputs_are_all_parsed() -> None:
    source = textwrap.dedent(
        """
        class M(dspy.Module):
            def __init__(self):
                super().__init__()

            def forward(self, **inputs):
                # JSON is all that crosses the boundary: dicts for models, a bare string for the
                # enum, and the count arrives as a string unless it is parsed back.
                return dspy.Prediction(
                    people=[{"name": "Ada", "age": 36}, {"name": "Bob", "age": 40}],
                    color="blue",
                    count="2",
                )
        """
    ).strip()
    with Flex(Roster, interpreter_factory=_sandbox) as program:
        program._bind_code(source)
        out = program(text="two people")

    assert out.people == [Person(name="Ada", age=36), Person(name="Bob", age=40)]
    assert out.color is Color.BLUE
    assert out.count == 2 and isinstance(out.count, int)


@deno_required
def test_payload_that_does_not_match_the_declared_type_raises() -> None:
    source = textwrap.dedent(
        """
        class M(dspy.Module):
            def __init__(self):
                super().__init__()

            def forward(self, **inputs):
                return dspy.Prediction(person="Ada, aged thirty-six")
        """
    ).strip()
    with Flex(Extract, interpreter_factory=_sandbox) as program:
        program._bind_code(source)
        with pytest.raises(CodeInterpreterError, match="person"):
            program(text="Ada is 36")


@deno_required
def test_missing_declared_output_field_raises() -> None:
    source = textwrap.dedent(
        """
        class M(dspy.Module):
            def __init__(self):
                super().__init__()

            def forward(self, **inputs):
                return dspy.Prediction(summary="Ada, 36")  # not the declared output field
        """
    ).strip()
    with Flex(Extract, interpreter_factory=_sandbox) as program:
        program._bind_code(source)
        with pytest.raises(CodeInterpreterError, match=r"missing declared output field\(s\) \['person'\]"):
            program(text="Ada is 36")


# =============================================================================
# How GEPA catches it
# =============================================================================


@deno_required
def test_gepa_scores_a_type_mismatch_as_a_failure_in_place() -> None:
    """A candidate that returns the wrong type for a declared field is a bad candidate, not a
    crashed run: the example it fails on takes the failure score in its own slot, so the search
    keeps going and simply doesn't select it."""

    def exact_match(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return 1.0 if getattr(pred, "person", None) == gold.person else 0.0

    source = textwrap.dedent(
        """
        class M(dspy.Module):
            def __init__(self):
                super().__init__()

            def forward(self, **inputs):
                if inputs["text"] == "bad":
                    return dspy.Prediction(person="not a person")
                return dspy.Prediction(person={"name": inputs["text"], "age": 1})
        """
    ).strip()

    student = Flex(Extract, interpreter_factory=_sandbox)
    adapter = DspyAdapter(student_module=student, metric_fn=exact_match, feedback_map={})
    batch = [
        dspy.Example(text="Ada", person=Person(name="Ada", age=1)).with_inputs("text"),
        dspy.Example(text="bad", person=Person(name="bad", age=1)).with_inputs("text"),
        dspy.Example(text="Bob", person=Person(name="Bob", age=1)).with_inputs("text"),
    ]

    result = adapter.evaluate(batch, {make_code_key("self"): source}, capture_traces=False)
    student.close()

    assert result.scores == [1.0, 0.0, 1.0]  # the type error is scored, not raised
    assert result.outputs[1] is None
    assert result.outputs[0].person == Person(name="Ada", age=1)  # and it is a real model, parsed
