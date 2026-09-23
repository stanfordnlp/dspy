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
import logging
import shutil
import textwrap
from typing import Callable

import pydantic
import pytest

import dspy
from dspy.predict.flex import Flex, bridge
from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.teleprompt.bootstrap_trace import FailedPrediction
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
    # An annotation the signature-string grammar cannot carry is emitted untyped rather than
    # rendered into source that would fail to bind.
    class Unrenderable(dspy.Signature):
        text: str = dspy.InputField()
        fn: Callable[[int], int] = dspy.OutputField()

    program = Flex(Unrenderable, interpreter_factory=MockInterpreter)
    assert "'text: str -> fn'" in program.module_src  # the field is there, with no type annotation


@deno_required
def test_custom_type_resolves_when_it_is_not_on_the_calling_stack() -> None:
    """The regression `bridge._resolve_signature` exists for.

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

    program = Flex(_offstack_types.ExtractContact, interpreter_factory=_sandbox)
    program._bind_code(source)
    dspy.configure(lm=DummyLM([{"contact": json.dumps({"name": "Ada", "age": 36})}] * 2))
    out = program(text="Ada is 36")  # raised `Unknown name: Contact` before the handoff
    assert out.contact == _offstack_types.Contact(name="Ada", age=36)

    # And the host-built sub-predictor resolved `Contact` to the module's class, not a lookalike:
    inv = bridge._Invocation(program._bridge, {})
    inv.construct("Predict", "text: str -> contact: Contact", "p", {})
    assert inv._predictors["p"].signature.output_fields["contact"].annotation is _offstack_types.Contact


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
    program = Flex(Extract, interpreter_factory=_sandbox)
    program._bind_code(source)
    dspy.configure(lm=DummyLM([{"person": json.dumps({"name": "Ada", "age": 36})}] * 2))
    program(text="Ada is 36")  # would raise `Unknown name: Person` if resolution failed
    inv = bridge._Invocation(program._bridge, {})
    inv.construct("Predict", "text: str -> person: Person", "p", {})
    assert inv._predictors["p"].signature.output_fields["person"].annotation is Person


# =============================================================================
# The return path parses against the declared types (Deno)
# =============================================================================


@deno_required
def test_pydantic_output_comes_back_as_the_model_like_predict_does() -> None:
    payload = json.dumps({"name": "Ada", "age": 36})
    dspy.configure(lm=DummyLM([{"person": payload}] * 2))

    baseline = dspy.Predict(Extract)(text="Ada is 36")
    program = Flex(Extract, interpreter_factory=_sandbox)
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
    program = Flex(Roster, interpreter_factory=_sandbox)
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
    program = Flex(Extract, interpreter_factory=_sandbox)
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
    program = Flex(Extract, interpreter_factory=_sandbox)
    program._bind_code(source)
    with pytest.raises(CodeInterpreterError, match=r"missing required output field\(s\) \['person'\]"):
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

    result = adapter.evaluate(batch, {"self": source}, capture_traces=False)

    assert result.scores == [1.0, 0.0, 1.0]  # the type error is scored, not raised
    assert isinstance(result.outputs[1], FailedPrediction)
    assert "not a valid" in result.outputs[1].completion_text
    assert result.outputs[0].person == Person(name="Ada", age=1)  # and it is a real model, parsed


@deno_required
def test_custom_type_inputs_are_restored_across_the_bridge(caplog) -> None:
    """A custom-type input (dspy.Image) crosses into the sandbox as its serialized string; the
    bridge hands the original object back to the host predictor, so Predict sees a real Image —
    no spurious "Type mismatch" warning, and adapters get the object rather than a tagged string.
    """
    class Caption(dspy.Signature):
        image: dspy.Image = dspy.InputField()
        a: str = dspy.OutputField()

    dspy.configure(lm=DummyLM([{"a": "a cat"}]))
    flex = Flex(Caption, interpreter_factory=lambda: dspy.PythonInterpreter())
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        out = flex(image=dspy.Image(url="https://example.com/x.png"))

    assert out.a == "a cat"
    mismatch_warnings = [r.getMessage() for r in caplog.records if "Type mismatch" in r.getMessage()]
    assert mismatch_warnings == []


# =============================================================================
# Optional / defaulted output fields: presence is required, None only where admitted
# =============================================================================


def test_optional_output_field_accepts_explicit_none() -> None:
    """Optionality applies to the value, not the key: an explicit null stays None."""
    class MaybeAnswer(dspy.Signature):
        q: str = dspy.InputField()
        a: str | None = dspy.OutputField()

    program = Flex(MaybeAnswer, interpreter_factory=MockInterpreter)
    assert program._bridge._to_prediction({"a": None}).a is None


def test_optional_output_field_may_be_omitted() -> None:
    """The Flex exit is a module boundary, not an LM response: the signature's own declarations
    apply, so an Optional field the generated code omits comes back as None."""
    class MaybeAnswer(dspy.Signature):
        q: str = dspy.InputField()
        a: str | None = dspy.OutputField()
        b: str = dspy.OutputField()

    program = Flex(MaybeAnswer, interpreter_factory=MockInterpreter)
    pred = program._bridge._to_prediction({"b": "x"})
    assert pred.a is None
    assert pred.b == "x"


def test_output_field_default_fills_a_missing_field() -> None:
    """A field declared with a default falls back to it when the generated code omits it."""
    class Defaulted(dspy.Signature):
        q: str = dspy.InputField()
        a: str = dspy.OutputField(default="fallback")

    program = Flex(Defaulted, interpreter_factory=MockInterpreter)
    assert program._bridge._to_prediction({}).a == "fallback"


def test_complex_output_field_defaults_fill_host_side_per_forward() -> None:
    """Defaults fill on the host, after the JSON crossing: any object works, regardless of the
    sandbox's boundary, and each forward gets a fresh copy (pydantic deep-copies plain defaults;
    factories are called per fill) — mutating one prediction cannot poison later forwards."""
    class Complex(dspy.Signature):
        q: str = dspy.InputField()
        person: Person = dspy.OutputField(default=Person(name="unknown", age=0))
        tags: list[str] = dspy.OutputField(default_factory=lambda: ["fresh"])

    program = Flex(Complex, interpreter_factory=MockInterpreter)
    first = program._bridge._to_prediction({})
    assert first.person == Person(name="unknown", age=0)
    first.person.name = "MUTATED"
    first.tags.append("MUTATED")

    second = program._bridge._to_prediction({})
    assert second.person == Person(name="unknown", age=0)  # fresh copy, unaffected
    assert second.tags == ["fresh"]  # factory re-invoked per fill


def test_missing_required_output_field_still_raises() -> None:
    """No default and not Optional means required: omitting it is the candidate's error."""
    class Strict(dspy.Signature):
        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    program = Flex(Strict, interpreter_factory=MockInterpreter)
    with pytest.raises(CodeInterpreterError, match=r"missing required output field\(s\) \['a'\]"):
        program._bridge._to_prediction({})


def test_none_for_a_non_optional_output_field_is_rejected() -> None:
    """A JSON null for `a: str` must not be stringified into "None" — a value the generated
    program never produced (natively the caller would see None, and an adapter can only ever
    yield text). The bridge surfaces the type mismatch instead."""
    class Strict(dspy.Signature):
        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    program = Flex(Strict, interpreter_factory=MockInterpreter)
    with pytest.raises(CodeInterpreterError, match=r"returned None for output field 'a'"):
        program._bridge._to_prediction({"a": None})


@deno_required
def test_optional_none_round_trips_through_the_sandbox() -> None:
    """End to end: a generated forward returning an explicit None crosses as JSON null and the
    caller receives a real None, not a "None" string."""
    class MaybeAnswer(dspy.Signature):
        q: str = dspy.InputField()
        a: str | None = dspy.OutputField()

    source = textwrap.dedent(
        """
        class MaybeModule(dspy.Module):
            def __init__(self):
                super().__init__()

            def forward(self, **inputs):
                return dspy.Prediction(a=None)
        """
    ).strip()

    program = Flex(MaybeAnswer, interpreter_factory=_sandbox)
    program._bind_code(source)
    assert program(q="x").a is None


# =============================================================================
# render_annotation: the signature parser's inverse (no Deno)
# =============================================================================


def test_render_annotation_round_trips_through_the_signature_parser() -> None:
    """``render_annotation`` mirrors dspy's ``_parse_type_node``: for every annotation it
    accepts, the emitted token parses back to the identical annotation under the same
    ``custom_types``. This round trip is the renderer's correctness contract."""
    from typing import Any, Literal, Optional

    from dspy.predict.flex.ctx import render_annotation
    from dspy.signatures.signature import make_signature

    custom = {"Person": Person, "Color": Color}
    cases = [
        str,
        int,
        bool,
        Person,
        Color,
        list[Person],
        dict[str, int],
        dict[str, Any],
        tuple[int, ...],
        str | None,
        Optional[float],  # noqa: UP045 — covers the typing.Union spelling, distinct origin from `float | None`
        str | int | None,
        Literal["a", "b"],
        Literal[1, 2],
        list[dict[str, list[int]]],
        Any,
    ]
    for annotation in cases:
        token = render_annotation(annotation, custom)
        probe = make_signature(f"x: {token} -> y", custom_types=custom)
        assert probe.input_fields["x"].annotation == annotation, f"{annotation!r} -> {token!r}"


def test_render_annotation_rejects_what_the_grammar_cannot_carry() -> None:
    from dspy.predict.flex.ctx import render_annotation

    with pytest.raises(ValueError):
        render_annotation(Callable[[int], int])

    class NotShared(pydantic.BaseModel):
        x: int

    # A class the parser could not resolve by name (absent from custom_types) is refused at
    # render time instead of producing a token that fails to parse back.
    with pytest.raises(ValueError):
        render_annotation(NotShared)
