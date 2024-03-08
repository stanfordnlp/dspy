import textwrap
import pytest
import pydantic
from dspy import Signature, infer_prefix, InputField, OutputField
from typing import List

import dspy
from dspy.utils.dummies import DummyLM


def test_field_types_and_custom_attributes():
    class TestSignature(Signature):
        """Instructions"""

        input1: str = InputField()
        input2: int = InputField()
        output1: List[str] = OutputField()
        output2 = OutputField()

    assert TestSignature.instructions == "Instructions"
    assert TestSignature.input_fields["input1"].annotation == str
    assert TestSignature.input_fields["input2"].annotation == int
    assert TestSignature.output_fields["output1"].annotation == List[str]
    assert TestSignature.output_fields["output2"].annotation == str


def test_no_input_output():
    with pytest.raises(TypeError):

        class TestSignature(Signature):
            input1: str


def test_no_input_output2():
    with pytest.raises(TypeError):

        class TestSignature(Signature):
            input1: str = pydantic.Field()


def test_all_fields_have_prefix():
    class TestSignature(Signature):
        input = InputField(prefix="Modified:")
        output = OutputField()

    assert TestSignature.input_fields["input"].json_schema_extra["prefix"] == "Modified:"
    assert TestSignature.output_fields["output"].json_schema_extra["prefix"] == "Output:"


def test_signature_parsing():
    signature = Signature("input1, input2 -> output")
    assert "input1" in signature.input_fields
    assert "input2" in signature.input_fields
    assert "output" in signature.output_fields


def test_with_signature():
    signature1 = Signature("input1, input2 -> output")
    signature2 = signature1.with_instructions("This is a test")
    assert signature2.instructions == "This is a test"
    assert signature1 is not signature2, "The type should be immutable"


def test_with_updated_field():
    signature1 = Signature("input1, input2 -> output")
    signature2 = signature1.with_updated_fields("input1", prefix="Modified:")
    assert signature2.input_fields["input1"].json_schema_extra["prefix"] == "Modified:"
    assert signature1.input_fields["input1"].json_schema_extra["prefix"] == "Input 1:"
    assert signature1 is not signature2, "The type should be immutable"
    for key in signature1.fields.keys():
        if key != "input1":
            assert signature1.fields[key].json_schema_extra == signature2.fields[key].json_schema_extra
    assert signature1.instructions == signature2.instructions


def test_empty_signature():
    with pytest.raises(ValueError):
        Signature("")


def test_instructions_signature():
    with pytest.raises(ValueError):
        Signature("")


def test_signature_instructions():
    sig1 = Signature("input1 -> output1", instructions="This is a test")
    assert sig1.instructions == "This is a test"
    sig2 = Signature("input1 -> output1", "This is a test")
    assert sig2.instructions == "This is a test"


def test_signature_instructions_none():
    sig1 = Signature("a, b -> c")
    assert sig1.instructions == f"Given the fields `a`, `b`, produce the fields `c`."


def test_signature_from_dict():
    signature = Signature({"input1": InputField(), "input2": InputField(), "output": OutputField()})
    for k in ["input1", "input2", "output"]:
        assert k in signature.fields
        assert signature.fields[k].annotation == str


def test_signature_from_dict():
    signature = Signature({"input1": InputField(), "input2": InputField(), "output": OutputField()})
    assert "input1" in signature.input_fields
    assert "input2" in signature.input_fields
    assert "output" in signature.output_fields


def test_signature_equality():
    sig1 = Signature("input1 -> output1")
    sig2 = Signature("input1 -> output1")
    assert sig1.equals(sig2)


def test_signature_inequality():
    sig1 = Signature("input1 -> output1")
    sig2 = Signature("input2 -> output2")
    assert not sig1.equals(sig2)


def test_equality_format():
    class TestSignature(Signature):
        input = InputField(format=lambda x: x)
        output = OutputField()

    assert TestSignature.equals(TestSignature)


def test_signature_reverse():
    sig = Signature("input1 -> output1")
    assert sig.signature == "input1 -> output1"


def test_insert_field_at_various_positions():
    class InitialSignature(Signature):
        input1: str = InputField()
        output1: int = OutputField()

    S1 = InitialSignature.prepend("new_input_start", InputField(), str)
    S2 = InitialSignature.append("new_input_end", InputField(), str)
    assert "new_input_start" == list(S1.input_fields.keys())[0]
    assert "new_input_end" == list(S2.input_fields.keys())[-1]

    S3 = InitialSignature.prepend("new_output_start", OutputField(), str)
    S4 = InitialSignature.append("new_output_end", OutputField(), str)
    assert "new_output_start" == list(S3.output_fields.keys())[0]
    assert "new_output_end" == list(S4.output_fields.keys())[-1]


def test_infer_prefix():
    assert infer_prefix("someAttributeName42IsCool") == "Some Attribute Name 42 Is Cool"
    assert infer_prefix("version2Update") == "Version 2 Update"
    assert infer_prefix("modelT45Enhanced") == "Model T 45 Enhanced"
    assert infer_prefix("someAttributeName") == "Some Attribute Name"
    assert infer_prefix("some_attribute_name") == "Some Attribute Name"
    assert infer_prefix("URLAddress") == "URL Address"
    assert infer_prefix("isHTTPSecure") == "Is HTTP Secure"
    assert infer_prefix("isHTTPSSecure123") == "Is HTTPS Secure 123"


def test_insantiating():
    sig = Signature("input -> output")
    assert issubclass(sig, Signature)
    assert sig.__name__ == "StringSignature"
    value = sig(input="test", output="test")
    assert isinstance(value, sig)


def test_insantiating2():
    class SubSignature(Signature):
        input = InputField()
        output = OutputField()

    assert issubclass(SubSignature, Signature)
    assert SubSignature.__name__ == "SubSignature"
    value = SubSignature(input="test", output="test")
    assert isinstance(value, SubSignature)


def test_multiline_instructions():
    class MySignature(Signature):
        """First line
        Second line"""

        output = OutputField()

    predictor = dspy.Predict(MySignature)

    lm = DummyLM(["short answer"])
    dspy.settings.configure(lm=lm)
    assert predictor().output == "short answer"

    assert lm.get_convo(-1) == textwrap.dedent("""\
        First line
                Second line
        
        ---
        
        Follow the following format.
        
        Output: ${output}
        
        ---
        
        Output: short answer""")
