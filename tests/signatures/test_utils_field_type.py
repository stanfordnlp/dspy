import pydantic
import pytest

from dspy.signatures.utils import get_dspy_field_type
from dspy import InputField, OutputField


def test_get_dspy_field_type_input():
    field = InputField()
    assert get_dspy_field_type(field) == "input"


def test_get_dspy_field_type_output():
    field = OutputField()
    assert get_dspy_field_type(field) == "output"


def test_get_dspy_field_type_missing_flag_raises():
    field = pydantic.Field(json_schema_extra={})
    with pytest.raises(ValueError, match="does not have a __dspy_field_type"):
        get_dspy_field_type(field)
