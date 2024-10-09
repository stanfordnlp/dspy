from typing import Any, Optional, Union
from dspy.adapters.chat_adapter import _format_field_value
import pytest

import pydantic
import dspy
from dspy.functional import TypedPredictor
from dspy.signatures.signature import signature_to_template


def get_field_and_parser(signature: dspy.Signature) -> tuple[Any, Any]:
    module = TypedPredictor(signature)
    signature = module._prepare_signature()
    assert "answer" in signature.fields, "'answer' not in signature.fields"
    field = signature.fields.get("answer")
    parser = field.json_schema_extra.get("parser")
    return field, parser


class Mysubmodel(pydantic.BaseModel):
    sub_floating: float


class MyModel(pydantic.BaseModel):
    floating: float
    string: str
    boolean: bool
    integer: int
    optional: Optional[str]
    sequence_of_strings: list[str]
    union: Union[str, float]
    submodel: Mysubmodel
    optional_submodel: Optional[Mysubmodel]
    optional_existing_submodule: Optional[Mysubmodel]


def build_model_instance() -> MyModel:
    return MyModel(
        floating=3.14,
        string="foobar",
        boolean=True,
        integer=42,
        optional=None,
        sequence_of_strings=["foo", "bar"],
        union=3.14,
        submodel=Mysubmodel(sub_floating=42.42),
        optional_submodel=None,
        optional_existing_submodule=Mysubmodel(sub_floating=42.42),
    )


@pytest.mark.parametrize(
    "test_type,serialized, expected", [(str, "foo", "foo"), (int, "42", 42), (float, "42.42", 42.42)]
)
def test_basic_types(test_type: type, serialized: str, expected: Any):
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: test_type = dspy.OutputField()

    _, parser = get_field_and_parser(MySignature)
    assert parser is test_type, "Parser is not correct for 'answer'"
    assert parser(serialized) == expected, f"{test_type}({serialized})!= {expected}"


def test_boolean():
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: bool = dspy.OutputField()

    _, parser = get_field_and_parser(MySignature)
    assert parser("true"), f"Parsing 'true' failed"
    assert not parser("false"), f"Parsing 'false' failed"


@pytest.mark.parametrize(
    "test_type,serialized, expected",
    [(list[str], '["foo", "bar"]', ["foo", "bar"]), (tuple[int, float], "[42, 3.14]", (42, 3.14))],
)
def test_sequences(test_type: type, serialized: str, expected: Any):
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: test_type = dspy.OutputField()

    _, parser = get_field_and_parser(MySignature)

    assert parser(serialized) == expected, f"Parsing {expected} failed"


@pytest.mark.parametrize(
    "test_type,serialized, expected",
    [
        (Optional[str], '"foobar"', "foobar"),
        (Optional[str], "null", None),
        (Union[str, float], "3.14", 3.14),
        (Union[str, bool], "true", True),
    ],
)
def test_unions(test_type: type, serialized: str, expected: Any):
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: test_type = dspy.OutputField()

    _, parser = get_field_and_parser(MySignature)

    assert parser(serialized) == expected, f"Parsing {expected} failed"


def test_pydantic():
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: MyModel = dspy.OutputField()

    _, parser = get_field_and_parser(MySignature)

    instance = build_model_instance()
    parsed_instance = parser(instance.model_dump_json())

    formatted_instance = _format_field_value(instance.model_dump_json())
    assert formatted_instance == instance.model_dump_json(), f"{formatted_instance} != {instance.model_dump_json()}"

    assert parsed_instance == instance, f"{instance} != {parsed_instance}"


def test_optional_pydantic():
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: Optional[MyModel] = dspy.OutputField()

    _, parser = get_field_and_parser(MySignature)

    instance = build_model_instance()
    parsed_instance = parser(instance.model_dump_json())
    assert parsed_instance == instance, f"{instance} != {parsed_instance}"

    formatted_instance = _format_field_value(instance.model_dump_json())
    assert formatted_instance == instance.model_dump_json(), f"{formatted_instance} != {instance.model_dump_json()}"

    # Check null case
    parsed_instance = parser("null")
    assert parsed_instance == None, "Optional[MyModel] should be None"


def test_nested_pydantic():
    class NestedModel(pydantic.BaseModel):
        model: MyModel

    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: NestedModel = dspy.OutputField()

    _, parser = get_field_and_parser(MySignature)

    instance = NestedModel(model=build_model_instance())
    parsed_instance = parser(instance.model_dump_json())

    formatted_instance = _format_field_value(instance.model_dump_json())
    assert formatted_instance == instance.model_dump_json(), f"{formatted_instance} != {instance.model_dump_json()}"

    assert parsed_instance == instance, f"{instance} != {parsed_instance}"


def test_dataclass():
    from dataclasses import dataclass, asdict
    import ujson

    @dataclass(frozen=True)
    class MyDataclass:
        string: str
        number: int
        floating: float
        boolean: bool

    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: MyDataclass = dspy.OutputField()

    _, parser = get_field_and_parser(MySignature)

    instance = MyDataclass("foobar", 42, 3.14, True)
    parsed_instance = parser('{"string": "foobar", "number": 42, "floating": 3.14, "boolean": true}')
    assert parsed_instance == instance, f"{instance} != {parsed_instance}"

    formatted_instance = _format_field_value(ujson.dumps(asdict(instance)))
    assert formatted_instance == ujson.dumps(asdict(instance)), f"{formatted_instance} != {ujson.dumps(asdict(instance))}"
