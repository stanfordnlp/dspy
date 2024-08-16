from typing import Generic, TypeVar, Any, Optional, Union
import pytest

import pydantic
import dspy
from dspy.evaluate import Evaluate
from dspy.functional import TypedPredictor
from dspy.utils import DummyLM

def get_field_and_parser(signature:dspy.Signature) -> tuple[Any, Any]:
    module = TypedPredictor(signature)
    signature = module._prepare_signature()
    assert "answer" in signature.fields, "'answer' not in signature.fields"
    field = signature.fields.get('answer')
    parser = field.json_schema_extra.get("parser")
    return field, parser
    

@pytest.mark.parametrize("test_type,serialized, expected", [(str, "foo", "foo"), (int, "42", 42), (float, "42.42", 42.42)])
def test_basic_types(test_type:type, serialized:str, expected:Any):
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
    
    
@pytest.mark.parametrize("test_type,serialized, expected", [(list[str], '{"value" : ["foo", "bar"]}', ['foo', 'bar']), (tuple[int, float], '{"value":[42, 3.14]}', (42, 3.14))])
def test_sequences(test_type:type, serialized:str, expected:Any):
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: test_type = dspy.OutputField()
        
    
    _, parser = get_field_and_parser(MySignature)
    
    assert parser(serialized) == expected, f"Parsing {expected} failed"
    
@pytest.mark.parametrize("test_type,serialized, expected", [
    (Optional[str], '{"value": "foobar"}', "foobar"),
    (Optional[str], '{"value": null}', None),
    (Union[str, float], '{"value": 3.14}', 3.14),
    (Union[str, bool], '{"value": true}', True)
    ])
def test_unions(test_type:type, serialized:str, expected:Any):
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: test_type = dspy.OutputField()
        
    _, parser = get_field_and_parser(MySignature)
    
    assert parser(serialized) == expected, f"Parsing {expected} failed"
    
    
def test_pydantic():
    
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
        
        
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: MyModel = dspy.OutputField()
        
    _, parser = get_field_and_parser(MySignature)
    
    instance = MyModel(
        floating=3.14,
        string="foobar",
        boolean=True,
        integer=42,
        optional=None,
        sequence_of_strings=["foo", "bar"],
        union=3.14,
        submodel=Mysubmodel(sub_floating=42.42),
        optional_submodel=None
    )
    
    parser(instance.model_dump_json()) == instance, f"{instance}!= {parser(instance)}"
    
        