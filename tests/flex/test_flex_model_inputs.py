"""Model-typed values (pydantic models, dataclasses) inside a dspy.Flex sandbox.

Everything that enters the sandbox crosses as JSON, so a `Place` input reaches the generated
`forward` as a dict — but the proposer is told the field is a `Place`, so it writes
`place_one.name` and the candidate fails with `AttributeError: 'dict' object has no attribute
'name'`. Three things are pinned down here:

* dicts that arrive from the host (Flex inputs, bridged predictor outputs) are records that
  support both `x.name` and `x["name"]`, so either access style is correct;
* when such a value is handed back to a bridged predictor whose signature declares the model,
  the host rebuilds the model before calling, so the predictor sees a `Place` and logs no
  "Type mismatch" warning;
* the proposer-facing signature spec lists the model's fields, so the code proposer knows
  what is inside the value.
"""

from __future__ import annotations

import json
import logging
import shutil
import textwrap

import pydantic
import pytest

import dspy
from dspy.predict.flex import Flex, bridge
from dspy.predict.flex.ctx import FlexContext
from dspy.utils.dummies import DummyLM

deno_required = pytest.mark.skipif(shutil.which("deno") is None, reason="Deno is not installed")


class Place(pydantic.BaseModel):
    name: str
    address: str


class Conflate(dspy.Signature):
    """Determine if two point-of-interests are the same"""

    place_one: Place = dspy.InputField(description="The first place to compare")
    place_two: Place = dspy.InputField(description="The second place to compare")
    distance: int = dspy.InputField(description="The distance between the two places in meters")
    match: bool = dspy.OutputField(description="Whether the two places are the same")


A = Place(name="Blue Bottle", address="1 Main St")
B = Place(name="Blue Bottle Coffee", address="1 Main Street")


def _flex(signature, source: str) -> Flex:
    program = Flex(signature, interpreter_factory=dspy.PythonInterpreter)
    program._bind_code(textwrap.dedent(source).strip())
    return program


# =============================================================================
# Inputs arrive as records: attribute and item access both work
# =============================================================================


@deno_required
def test_model_typed_input_supports_attribute_and_item_access() -> None:
    """The proposer writes `place_one.name`; a plain dict makes every such candidate fail."""
    program = _flex(
        Conflate,
        """
        class ConflateModule(dspy.Module):
            def __init__(self):
                super().__init__()

            def forward(self, place_one, place_two, distance):
                same = place_one.name == place_two["name"] and distance < 50
                return dspy.Prediction(match=same)
        """,
    )
    assert program(place_one=A, place_two=A, distance=10).match is True
    assert program(place_one=A, place_two=B, distance=10).match is False


@deno_required
def test_nested_model_inputs_are_hydrated() -> None:
    class Pick(dspy.Signature):
        places: list[Place] = dspy.InputField()
        first: str = dspy.OutputField()

    program = _flex(
        Pick,
        """
        class PickModule(dspy.Module):
            def __init__(self):
                super().__init__()

            def forward(self, **inputs):
                return dspy.Prediction(first=inputs["places"][0].name)
        """,
    )
    assert program(places=[A, B]).first == "Blue Bottle"


@deno_required
def test_record_key_that_shadows_a_dict_method_is_reachable_by_item() -> None:
    """Records are dicts: a key like `items` is shadowed by the dict method on attribute access,
    so item access is the way to reach it; every other dict behaviour is unchanged."""

    class Count(dspy.Signature):
        cart: dict = dspy.InputField()
        n: int = dspy.OutputField()

    program = _flex(
        Count,
        """
        class CountModule(dspy.Module):
            def __init__(self):
                super().__init__()

            def forward(self, cart):
                assert callable(cart.items)
                return dspy.Prediction(n=len(cart["items"]) + len(list(cart.keys())))
        """,
    )
    assert program(cart={"items": [1, 2, 3]}).n == 4


# =============================================================================
# Records handed back to a bridged predictor become the declared model on the host
# =============================================================================


@deno_required
def test_model_input_forwarded_to_a_bridged_predictor_is_rebuilt_on_the_host(caplog) -> None:
    """The baseline passes every Flex input straight to a `Predict` over the same signature. The
    host must see a `Place`, as it would outside Flex, not a dict plus a spurious warning."""
    dspy.configure(lm=DummyLM([{"match": "True"}]))
    program = Flex(Conflate, interpreter_factory=dspy.PythonInterpreter)

    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        out = program(place_one=A, place_two=B, distance=12)

    assert out.match is True
    assert [r.getMessage() for r in caplog.records if "Type mismatch" in r.getMessage()] == []


@deno_required
def test_model_input_forwarded_to_a_chain_of_thought_is_rebuilt_on_the_host(caplog) -> None:
    """Not every bridgeable predictor exposes `.signature` (ChainOfThought doesn't), so the bridge
    must validate against the signature it resolved when it built the predictor."""
    dspy.configure(lm=DummyLM([{"reasoning": "same name", "match": "True"}]))
    program = _flex(
        Conflate,
        """
        class ConflateModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.judge = dspy.ChainOfThought("place_one: Place, place_two: Place -> match: bool")

            def forward(self, place_one, place_two, distance):
                return dspy.Prediction(match=self.judge(place_one=place_one, place_two=place_two).match)
        """,
    )

    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        out = program(place_one=A, place_two=B, distance=12)

    assert out.match is True
    assert [r.getMessage() for r in caplog.records if "Type mismatch" in r.getMessage()] == []


def test_validate_inputs_rebuilds_models_declared_by_the_bridged_signature() -> None:
    sig = dspy.make_signature(
        "place_one: Place, distance: int, note: str -> match: bool", custom_types={"Place": Place}
    )
    inputs = {"place_one": {"name": "x", "address": "y"}, "distance": 3, "note": {"k": "v"}, "extra": {"a": 1}}

    out = bridge._validate_inputs(sig, inputs)

    assert out["place_one"] == Place(name="x", address="y")
    assert out["distance"] == 3
    assert out["note"] == {"k": "v"}  # a dict for a `str` field is left for the adapter to format
    assert out["extra"] == {"a": 1}  # not a declared field: untouched


def test_validate_inputs_leaves_a_value_the_model_rejects_untouched() -> None:
    """The bridged signature is optimizer-authored; a shape mismatch is the predictor's problem
    to surface (it warns), not a reason for the bridge to raise."""
    sig = dspy.make_signature("place: Place -> match: bool", custom_types={"Place": Place})
    out = bridge._validate_inputs(sig, {"place": {"name": "only a name"}})
    assert out["place"] == {"name": "only a name"}


# =============================================================================
# Bridged predictor outputs are records too
# =============================================================================


@deno_required
def test_bridged_predictor_model_output_supports_attribute_access() -> None:
    class Extract(dspy.Signature):
        text: str = dspy.InputField()
        place: Place = dspy.OutputField()
        label: str = dspy.OutputField()

    dspy.configure(lm=DummyLM([{"place": json.dumps({"name": "Ada", "address": "1 Main"})}]))
    program = _flex(
        Extract,
        """
        class ExtractModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.find = dspy.Predict("text: str -> place: Place")

            def forward(self, text):
                r = self.find(text=text)
                # A record is still a dict, so it can be returned as the declared model's value.
                return dspy.Prediction(place=r.place, label=r.place.name + "/" + r.place["address"])
        """,
    )
    out = program(text="Ada at 1 Main")
    assert out.label == "Ada/1 Main"
    assert out.place == Place(name="Ada", address="1 Main")


# =============================================================================
# The proposer is told what a model-typed field contains
# =============================================================================


def test_signature_spec_lists_the_fields_of_model_typed_values() -> None:
    spec = FlexContext(signature_cls=Conflate).render_signature_spec()
    assert "place_one: Place (fields: name, address)" in spec
    assert "distance: int" in spec
    assert "distance: int (fields" not in spec
