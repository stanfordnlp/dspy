# ruff: noqa: UP007

from typing import Literal, Optional, Union

import pytest
from pydantic import BaseModel

from dspy.adapters.utils import _get_json_schema, get_annotation_name, parse_value


class Profile(BaseModel):
    name: str
    age: int


def test_parse_value_str_annotation():
    # Test basic string conversion
    assert parse_value(123, str) == "123"
    assert parse_value(True, str) == "True"
    assert parse_value("hello", str) == "hello"
    assert parse_value(None, str) == "None"
    assert parse_value([1, 2, 3], str) == "[1, 2, 3]"


def test_parse_value_pydantic_types():
    # Test with pydantic BaseModel - JSON string input
    json_str = '{"name": "John", "age": 30}'
    result = parse_value(json_str, Profile)
    assert isinstance(result, Profile)
    assert result.name == "John"
    assert result.age == 30

    # Test with pydantic BaseModel - dict input
    dict_input = {"name": "Jane", "age": 25}
    result = parse_value(dict_input, Profile)
    assert isinstance(result, Profile)
    assert result.name == "Jane"
    assert result.age == 25

    # Test with invalid pydantic data
    with pytest.raises(Exception):
        parse_value('{"name": "John"}', Profile)  # missing required age field


def test_parse_value_basic_types():
    # Test int
    assert parse_value("42", int) == 42
    assert parse_value(42, int) == 42

    # Test float
    assert parse_value("3.14", float) == 3.14
    assert parse_value(3.14, float) == 3.14

    # Test bool
    assert parse_value("true", bool) is True
    assert parse_value(True, bool) is True
    assert parse_value("false", bool) is False

    # Test list
    assert parse_value("[1, 2, 3]", list[int]) == [1, 2, 3]
    assert parse_value([1, 2, 3], list[int]) == [1, 2, 3]


def test_parse_value_literal():
    # Test Literal type
    assert parse_value("option1", Literal["option1", "option2"]) == "option1"
    assert parse_value("option2", Literal["option1", "option2"]) == "option2"

    # Test Literal with quotes and prefixes
    assert parse_value("'option1'", Literal["option1", "option2"]) == "option1"
    assert parse_value('"option1"', Literal["option1", "option2"]) == "option1"
    assert parse_value("Literal[option1]", Literal["option1", "option2"]) == "option1"
    assert parse_value("str[option1]", Literal["option1", "option2"]) == "option1"

    # Test invalid literal
    with pytest.raises(ValueError):
        parse_value("invalid", Literal["option1", "option2"])


def test_parse_value_literal_non_string_members():
    # Text adapters yield literal values as strings; coerce them to the member type.
    assert parse_value("2", Literal[1, 2, 3]) == 2
    assert parse_value('"2"', Literal[1, 2, 3]) == 2
    assert parse_value("True", Literal[True, False]) is True

    with pytest.raises(ValueError):
        parse_value("42", Literal[1, 2, 3])


def test_parse_value_union():
    # Test Union with None (Optional)
    assert parse_value("test", Optional[str]) == "test"
    assert parse_value("test", str | None) == "test"
    assert parse_value("5", int | None) == 5
    assert parse_value(None, Optional[str]) is None
    assert parse_value("text with [placeholder]", Optional[str]) == "text with [placeholder]"
    assert parse_value("text with [placeholder]", str | None) == "text with [placeholder]"

    # Test Union fallback to str
    assert parse_value("fallback", Union[int, str, None]) == "fallback"
    assert parse_value(5, Union[int, str, None]) == 5
    assert parse_value("fallback", int | str | None) == "fallback"
    assert parse_value(5, int | str | None) == 5
    assert parse_value("text with [placeholder]", Union[int, str, None]) == "text with [placeholder]"


def test_parse_value_json_repair():
    # Test cases where json_repair is needed
    assert parse_value('{"key": "value"}', dict) == {"key": "value"}

    # Test ast.literal_eval fallback
    assert parse_value("{'key': 'value'}", dict) == {"key": "value"}

    # Test fallback to original value when parsing fails
    malformed = "not json or literal"
    with pytest.raises(Exception):
        parse_value(malformed, dict)


@pytest.mark.parametrize(
    "pep604_annotation,typing_annotation,expected",
    [
        (str | None, Optional[str], "Union[str, NoneType]"),  # noqa: UP045
        (int | str, Union[int, str], "Union[int, str]"),
        (int | str | None, Union[int, str, None], "Union[int, str, NoneType]"),
        (list[str] | None, Optional[list[str]], "Union[list[str], NoneType]"),  # noqa: UP045
        (dict[str, int | None], dict[str, Optional[int]], "dict[str, Union[int, NoneType]]"),  # noqa: UP045
    ],
)
def test_get_annotation_name_pep604_union_matches_typing_union(pep604_annotation, typing_annotation, expected):
    assert get_annotation_name(typing_annotation) == expected
    assert get_annotation_name(pep604_annotation) == expected


def test_get_json_schema_preserves_pydantic_property_order():
    class Assessment(BaseModel):
        reasoning: str
        verdict: str
        confidence: float
        notes: str

    schema = _get_json_schema(Assessment)
    assert list(schema["properties"].keys()) == [
        "reasoning",
        "verdict",
        "confidence",
        "notes",
    ]
    assert schema["required"] == ["reasoning", "verdict", "confidence", "notes"]


def test_get_json_schema_field_named_type_not_hoisted_in_properties():
    class Event(BaseModel):
        id: str
        name: str
        type: str
        description: str

    schema = _get_json_schema(Event)
    assert list(schema["properties"].keys()) == ["id", "name", "type", "description"]
    assert schema["required"] == ["id", "name", "type", "description"]


def test_get_json_schema_nested_definitions_preserve_order():
    class Inner(BaseModel):
        z: int
        a: str
        type: str

    class Outer(BaseModel):
        b: str
        inner: Inner
        a: int

    schema = _get_json_schema(Outer)
    assert list(schema["properties"].keys()) == ["b", "inner", "a"]
    assert schema["required"] == ["b", "inner", "a"]
    assert list(schema["$defs"]["Inner"]["properties"].keys()) == ["z", "a", "type"]
    assert schema["$defs"]["Inner"]["required"] == ["z", "a", "type"]

    # Verify 'type' keyword is placed at index 0 in schema descriptors
    assert list(schema.keys())[0] == "type"
    assert list(schema["$defs"]["Inner"].keys())[0] == "type"
    assert list(schema["properties"]["b"].keys())[0] == "type"
    assert list(schema["properties"]["a"].keys())[0] == "type"
    assert list(schema["$defs"]["Inner"]["properties"]["type"].keys())[0] == "type"

