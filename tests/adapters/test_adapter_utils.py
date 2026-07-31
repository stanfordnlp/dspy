# ruff: noqa: UP007

from typing import Literal, Optional, Union

import pydantic
import pytest
from pydantic import BaseModel

import dspy
from dspy.adapters.utils import parse_value


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


def test_parse_value_enforces_str_constraints_from_field_info():
    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField(max_length=3)

    field = Sig.output_fields["answer"]

    assert parse_value("ok", field.annotation, field) == "ok"
    with pytest.raises(pydantic.ValidationError, match="at most 3 characters"):
        parse_value("toolong", field.annotation, field)


def test_parse_value_enforces_numeric_constraints_from_field_info():
    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        score: int = dspy.OutputField(ge=1, le=5)

    field = Sig.output_fields["score"]

    assert parse_value("3", field.annotation, field) == 3
    with pytest.raises(pydantic.ValidationError, match="less than or equal to 5"):
        parse_value("99", field.annotation, field)
    with pytest.raises(pydantic.ValidationError, match="greater than or equal to 1"):
        parse_value("0", field.annotation, field)


def test_parse_value_enforces_constraints_on_optional_annotation():
    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        answer: str | None = dspy.OutputField(max_length=3)

    field = Sig.output_fields["answer"]

    assert parse_value("ok", field.annotation, field) == "ok"
    with pytest.raises(pydantic.ValidationError, match="at most 3 characters"):
        parse_value("toolong", field.annotation, field)


def test_parse_value_enforces_constraints_on_container_annotation():
    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        tags: list[str] = dspy.OutputField(max_length=2)

    field = Sig.output_fields["tags"]

    assert parse_value('["a", "b"]', field.annotation, field) == ["a", "b"]
    with pytest.raises(pydantic.ValidationError, match="at most 2 items"):
        parse_value('["a", "b", "c"]', field.annotation, field)


def test_parse_value_unconstrained_field_is_unchanged():
    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        count: int = dspy.OutputField()

    answer = Sig.output_fields["answer"]
    count = Sig.output_fields["count"]

    assert parse_value("anything at all", answer.annotation, answer) == "anything at all"
    # A digit-only string must stay a string rather than being reinterpreted as an int.
    assert parse_value("1234", answer.annotation, answer) == "1234"
    assert parse_value("7", count.annotation, count) == 7


def test_parse_value_without_field_info_keeps_previous_behavior():
    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField(max_length=3)

    field = Sig.output_fields["answer"]

    assert parse_value("toolong", field.annotation) == "toolong"
    assert parse_value(123, str) == "123"
