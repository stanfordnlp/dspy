# ruff: noqa: UP007

from typing import Literal, Optional, Union

import pytest
from pydantic import BaseModel

from dspy.adapters.utils import _extract_first_json_object, parse_value


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


@pytest.mark.parametrize(
    "text,expected",
    [
        # JSON at the start of text
        ('{"name": "John", "age": 30} and some trailing text', '{"name": "John", "age": 30}'),
        # JSON in the middle of text
        ('Here is your result: {"status": "success", "data": [1, 2, 3]} done', '{"status": "success", "data": [1, 2, 3]}'),
        # JSON at the end of text
        ('The answer is {"result": 42}', '{"result": 42}'),
        # Nested JSON objects
        ('Response: {"outer": {"inner": {"deep": "value"}}, "count": 5}', '{"outer": {"inner": {"deep": "value"}}, "count": 5}'),
        # JSON with braces inside string values
        ('{"message": "Use {placeholders} like {this}", "valid": true}', '{"message": "Use {placeholders} like {this}", "valid": true}'),
        # JSON with escaped quotes in strings
        ('{"quote": "She said \\"hello\\" to me"}', '{"quote": "She said \\"hello\\" to me"}'),
        # No JSON present
        ("This is just plain text without any JSON", None),
        # Empty JSON object
        ("Here is an empty object: {}", "{}"),
        # Unbalanced braces (no valid JSON)
        ("This has { an opening but no closing", None),
        # Multiple JSON objects - should extract only the first
        ('{"first": 1} and then {"second": 2}', '{"first": 1}'),
        # JSON with newlines
        ("""Here is the result:
    {
        "name": "Alice",
        "scores": [95, 87, 92]
    }
    End of message""", """{
        "name": "Alice",
        "scores": [95, 87, 92]
    }"""),
    ],
)
def test_extract_first_json_object(text, expected):
    result = _extract_first_json_object(text)
    assert result == expected
