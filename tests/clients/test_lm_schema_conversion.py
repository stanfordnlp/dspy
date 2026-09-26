"""Generated Pydantic schemas receive the same preparation on both LM paths."""

import copy

import pydantic
import pytest

import dspy
from dspy.clients.execution import _canonical, prepare
from dspy.clients.legacy_requests import _strict_json_schema, chat_to_responses


@pytest.mark.parametrize("asynchronous", [False, True])
def test_generated_nested_schema_matches_legacy_preparation(asynchronous):
    class Item(pydantic.BaseModel):
        name: str

    class Answer(pydantic.BaseModel):
        items: list[Item]
        alternative: Item | str

    original = copy.deepcopy(Answer.model_json_schema())
    lm = dspy.LM("openai/gpt-4.1-mini", model_type="responses", engine="lm15", cache=False)
    call = prepare(lm, "Return an answer.", None, {"response_format": Answer}, asynchronous=asynchronous)
    canonical = _canonical(call)
    schema = canonical.config.response_format["schema"]
    legacy = chat_to_responses(call.legacy)["text"]["format"]["schema"]

    assert schema == legacy
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Item"]["additionalProperties"] is False
    assert canonical.config.response_format["strict"] is True
    assert Answer.model_json_schema() == original
    assert "additionalProperties" not in original


@pytest.mark.parametrize("additional_properties", [None, True, False])
def test_raw_schema_is_not_rewritten(additional_properties):
    schema = {"type": "object", "properties": {"word": {"type": "string"}}}
    if additional_properties is not None:
        schema["additionalProperties"] = additional_properties
    format_ = {"type": "json_schema", "json_schema": {"name": "Answer", "schema": schema, "strict": True}}
    original = copy.deepcopy(format_)
    lm = dspy.LM("openai/gpt-4.1-mini", model_type="responses", engine="lm15", cache=False)
    call = prepare(lm, "Return an answer.", None, {"response_format": format_})

    assert _canonical(call).config.response_format["schema"] == original["json_schema"]["schema"]
    assert format_ == original


# ─── a generated response_format schema is shaped for OpenAI strict mode ──


class _Inner(pydantic.BaseModel):
    note: str = "n/a"


class _Record(pydantic.BaseModel):
    name: str
    count: int = 0
    tag: str | None = None
    inner: _Inner | None = None
    items: list[_Inner] = []


def test_strict_schema_requires_every_property_and_drops_none_defaults():
    # OpenAI strict mode rejected any pydantic model with defaults or an
    # optional submodel ("required must include every key"). The shape now
    # follows the OpenAI SDK's own to_strict_json_schema rules.
    schema = _strict_json_schema(_Record.model_json_schema())
    assert schema["required"] == ["name", "count", "tag", "inner", "items"]
    assert schema["additionalProperties"] is False
    assert "default" not in schema["properties"]["tag"]  # None default dropped
    assert schema["properties"]["count"]["default"] == 0  # a value default is kept, as the OpenAI SDK keeps it
    inner = schema["$defs"]["_Inner"]
    assert inner["required"] == ["note"] and inner["additionalProperties"] is False
    assert {"$ref": "#/$defs/_Inner"} in schema["properties"]["inner"]["anyOf"]


def test_strict_schema_unravels_a_ref_with_siblings_and_a_lone_allof():
    schema = {"type": "object", "properties": {"a": {"$ref": "#/$defs/A", "description": "d"},
                                               "b": {"allOf": [{"$ref": "#/$defs/A"}]}},
              "$defs": {"A": {"type": "string"}}}
    out = _strict_json_schema(schema)
    assert out["properties"]["a"] == {"type": "string", "description": "d"}
    assert out["properties"]["b"] == {"$ref": "#/$defs/A"}  # a bare ref stays a ref
    assert out["required"] == ["a", "b"]


def test_pydantic_response_format_is_sent_strict_on_both_paths():
    lm = dspy.LM("openai/gpt-4o", engine="lm15", api_key="k", cache=False)
    request = _canonical(prepare(lm, "hi", None, {"response_format": _Record}))
    fmt = request.config.response_format
    assert fmt["type"] == "json_schema" and fmt.get("strict") is True
    assert fmt["schema"]["required"] == ["name", "count", "tag", "inner", "items"]
    # The forced LiteLLM Responses path converts the legacy body itself and
    # must mean the same contract.
    data = chat_to_responses({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}],
                              "response_format": _Record})
    fmt = data["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    assert fmt["schema"]["required"] == ["name", "count", "tag", "inner", "items"]
    assert "default" not in fmt["schema"]["properties"]["tag"]


# ─── oneOf, prefixItems, and items array schema recursion ─────────────────────


def test_strict_schema_recursively_processes_oneof_branches():
    schema = {
        "type": "object",
        "properties": {
            "result": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "const": "success"},
                            "data": {"type": "string"},
                            "extra": {"type": "string", "default": None},
                        },
                        "required": ["kind", "data"],
                    },
                    {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "const": "error"},
                            "code": {"type": "integer", "default": 0},
                        },
                        "required": ["kind"],
                    },
                ]
            }
        },
    }
    out = _strict_json_schema(schema)
    assert out["additionalProperties"] is False
    assert out["required"] == ["result"]

    branches = out["properties"]["result"]["oneOf"]
    # Branch 0: additionalProperties=False, all properties required, None default dropped
    assert branches[0]["additionalProperties"] is False
    assert branches[0]["required"] == ["kind", "data", "extra"]
    assert "default" not in branches[0]["properties"]["extra"]

    # Branch 1: additionalProperties=False, all properties required, non-None default kept
    assert branches[1]["additionalProperties"] is False
    assert branches[1]["required"] == ["kind", "code"]
    assert branches[1]["properties"]["code"]["default"] == 0


def test_strict_schema_recursively_processes_prefix_items():
    schema = {
        "type": "object",
        "properties": {
            "pair": {
                "type": "array",
                "prefixItems": [
                    {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "note": {"type": "string", "default": None},
                        },
                        "required": ["id"],
                    },
                    {"type": "string"},
                ],
            }
        },
    }
    out = _strict_json_schema(schema)
    assert out["additionalProperties"] is False
    assert out["required"] == ["pair"]

    prefix = out["properties"]["pair"]["prefixItems"]
    assert prefix[0]["additionalProperties"] is False
    assert prefix[0]["required"] == ["id", "note"]
    assert "default" not in prefix[0]["properties"]["note"]
    assert prefix[1] == {"type": "string"}


def test_strict_schema_recursively_processes_items_list():
    schema = {
        "type": "object",
        "properties": {
            "elements": {
                "type": "array",
                "items": [
                    {
                        "type": "object",
                        "properties": {
                            "val": {"type": "number"},
                            "opt": {"type": "string", "default": None},
                        },
                        "required": ["val"],
                    }
                ],
            }
        },
    }
    out = _strict_json_schema(schema)
    assert out["additionalProperties"] is False
    assert out["required"] == ["elements"]

    item = out["properties"]["elements"]["items"][0]
    assert item["additionalProperties"] is False
    assert item["required"] == ["val", "opt"]
    assert "default" not in item["properties"]["opt"]


from typing import Annotated, Literal  # noqa: E402


class _Cat(pydantic.BaseModel):
    type: Literal["cat"]
    name: str
    indoor: bool = True
    tag: str | None = None


class _Dog(pydantic.BaseModel):
    type: Literal["dog"]
    name: str
    breed: str = "mixed"


class _PetReport(pydantic.BaseModel):
    pet: Annotated[
        Annotated[_Cat, pydantic.Tag("cat")] | Annotated[_Dog, pydantic.Tag("dog")],
        pydantic.Discriminator("type"),
    ]
    pair: tuple[_Cat, int]


def test_discriminated_union_and_tuple_model_strict_schema():
    raw = _PetReport.model_json_schema()
    out = _strict_json_schema(raw)

    assert out["additionalProperties"] is False
    assert out["required"] == ["pet", "pair"]

    cat_def = out["$defs"]["_Cat"]
    assert cat_def["additionalProperties"] is False
    assert cat_def["required"] == ["type", "name", "indoor", "tag"]
    assert "default" not in cat_def["properties"]["tag"]
    assert cat_def["properties"]["indoor"]["default"] is True

    dog_def = out["$defs"]["_Dog"]
    assert dog_def["additionalProperties"] is False
    assert dog_def["required"] == ["type", "name", "breed"]
    assert dog_def["properties"]["breed"]["default"] == "mixed"

    assert len(out["properties"]["pet"]["oneOf"]) == 2
    assert len(out["properties"]["pair"]["prefixItems"]) == 2


def test_discriminated_union_response_format_is_sent_strict_on_both_paths():
    lm = dspy.LM("openai/gpt-4o", engine="lm15", api_key="k", cache=False)
    request = _canonical(prepare(lm, "describe pet", None, {"response_format": _PetReport}))
    fmt = request.config.response_format
    assert fmt["type"] == "json_schema" and fmt.get("strict") is True
    assert fmt["schema"]["required"] == ["pet", "pair"]
    assert fmt["schema"]["additionalProperties"] is False

    # Check both paths agree
    data = chat_to_responses(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "describe pet"}],
            "response_format": _PetReport,
        }
    )
    legacy_fmt = data["text"]["format"]
    assert legacy_fmt["type"] == "json_schema" and legacy_fmt["strict"] is True
    assert legacy_fmt["schema"]["required"] == ["pet", "pair"]
    assert legacy_fmt["schema"]["additionalProperties"] is False
    assert legacy_fmt["schema"] == fmt["schema"]


def test_nested_oneof_inside_prefix_items_and_vice_versa():
    # Deep nesting: an array with prefixItems where one item contains a oneOf,
    # and a oneOf branch where an item contains prefixItems.
    schema = {
        "type": "object",
        "properties": {
            "nested": {
                "type": "array",
                "prefixItems": [
                    {
                        "type": "object",
                        "properties": {
                            "choice": {
                                "oneOf": [
                                    {
                                        "type": "object",
                                        "properties": {
                                            "x": {"type": "integer"},
                                            "opt": {"type": "null", "default": None},
                                        },
                                        "required": ["x"],
                                    }
                                ]
                            }
                        },
                        "required": ["choice"],
                    }
                ],
            }
        },
    }
    out = _strict_json_schema(schema)
    assert out["additionalProperties"] is False
    outer_elem = out["properties"]["nested"]["prefixItems"][0]
    assert outer_elem["additionalProperties"] is False
    assert outer_elem["required"] == ["choice"]
    inner_branch = outer_elem["properties"]["choice"]["oneOf"][0]
    assert inner_branch["additionalProperties"] is False
    assert inner_branch["required"] == ["x", "opt"]
    assert "default" not in inner_branch["properties"]["opt"]

