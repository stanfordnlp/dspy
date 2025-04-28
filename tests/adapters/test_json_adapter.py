from unittest import mock

import pydantic
import pytest
from pydantic import create_model

import dspy

import json


def test_json_adapter_passes_structured_output_when_supported_by_model():
    class OutputField3(pydantic.BaseModel):
        subfield1: int = pydantic.Field(description="Int subfield 1", ge=0, le=10)
        subfield2: float = pydantic.Field(description="Float subfield 2")

    class TestSignature(dspy.Signature):
        input1: str = dspy.InputField()
        output1: str = dspy.OutputField()  # Description intentionally left blank
        output2: bool = dspy.OutputField(desc="Boolean output field")
        output3: OutputField3 = dspy.OutputField(desc="Nested output field")
        output4_unannotated = dspy.OutputField(desc="Unannotated output field")

    program = dspy.Predict(TestSignature)

    # Configure DSPy to use an OpenAI LM that supports structured outputs
    dspy.configure(lm=dspy.LM(model="openai/gpt4o"), adapter=dspy.JSONAdapter())
    with mock.patch("litellm.completion") as mock_completion:
        program(input1="Test input")

    def clean_schema_extra(field_name, field_info):
        attrs = dict(field_info.__repr_args__())
        if "json_schema_extra" in attrs:
            attrs["json_schema_extra"] = {
                k: v
                for k, v in attrs["json_schema_extra"].items()
                if k != "__dspy_field_type" and not (k == "desc" and v == f"${{{field_name}}}")
            }
        return attrs

    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args
    response_format = call_kwargs.get("response_format")
    assert response_format is not None
    assert issubclass(response_format, pydantic.BaseModel)
    assert response_format.model_fields.keys() == {"output1", "output2", "output3", "output4_unannotated"}

    # Configure DSPy to use a model from a fake provider that doesn't support structured outputs
    dspy.configure(lm=dspy.LM(model="fakeprovider/fakemodel"), adapter=dspy.JSONAdapter())
    with mock.patch("litellm.completion") as mock_completion:
        program(input1="Test input")

    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args
    assert response_format not in call_kwargs


def test_json_adapter_falls_back_when_structured_outputs_fails():
    class TestSignature(dspy.Signature):
        input1: str = dspy.InputField()
        output1: str = dspy.OutputField(desc="String output field")

    dspy.configure(lm=dspy.LM(model="openai/gpt4o"), adapter=dspy.JSONAdapter())
    program = dspy.Predict(TestSignature)
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.side_effect = [Exception("Bad structured outputs!"), mock_completion.return_value]
        program(input1="Test input")
        assert mock_completion.call_count == 2
        _, first_call_kwargs = mock_completion.call_args_list[0]
        assert issubclass(first_call_kwargs.get("response_format"), pydantic.BaseModel)
        _, second_call_kwargs = mock_completion.call_args_list[1]
        assert second_call_kwargs.get("response_format") == {"type": "json_object"}


def test_json_adapter_with_structured_outputs_does_not_mutate_original_signature():
    class OutputField3(pydantic.BaseModel):
        subfield1: int = pydantic.Field(description="Int subfield 1")
        subfield2: float = pydantic.Field(description="Float subfield 2")

    class TestSignature(dspy.Signature):
        input1: str = dspy.InputField()
        output1: str = dspy.OutputField()  # Description intentionally left blank
        output2: bool = dspy.OutputField(desc="Boolean output field")
        output3: OutputField3 = dspy.OutputField(desc="Nested output field")
        output4_unannotated = dspy.OutputField(desc="Unannotated output field")

    dspy.configure(lm=dspy.LM(model="openai/gpt4o"), adapter=dspy.JSONAdapter())
    program = dspy.Predict(TestSignature)
    with mock.patch("litellm.completion"):
        program(input1="Test input")

    assert program.signature.output_fields == TestSignature.output_fields


def test_json_adapter_sync_call():
    signature = dspy.make_signature("question->answer")
    adapter = dspy.ChatAdapter()
    lm = dspy.utils.DummyLM([{"answer": "Paris"}])
    result = adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})
    assert result == [{"answer": "Paris"}]


@pytest.mark.asyncio
async def test_json_adapter_async_call():
    signature = dspy.make_signature("question->answer")
    adapter = dspy.ChatAdapter()
    lm = dspy.utils.DummyLM([{"answer": "Paris"}])
    result = await adapter.acall(lm, {}, signature, [], {"question": "What is the capital of France?"})
    assert result == [{"answer": "Paris"}]


def test_json_adapter_passes_different_input_and_output_fields():
    class InputField1SubField3(pydantic.BaseModel):
        subsubfield1: int = pydantic.Field(description="Int subsubfield 1")
        subsubfield2: str = pydantic.Field(description="String subsubfield 2")
    class InputField1(pydantic.BaseModel):
        subfield1: int = pydantic.Field(description="Int subfield 1", ge=0, le=10)
        subfield2: str = pydantic.Field(description="Str subfield 2")
        subfield3: InputField1SubField3 = pydantic.Field(description="Nested field with InputField1SubField3")

    class OutputField1SubField2(pydantic.BaseModel):
        subsubfield1: bool = pydantic.Field(description="Boolean subsubfield 1")
        subsubfield2: float = pydantic.Field(description="Float subsubfield 2")
    class OutputField2(pydantic.BaseModel):
        subfield1: str = pydantic.Field(description="Str subfield 1")
        subfield2: OutputField1SubField2 = pydantic.Field(description="Nested field with OutputField1SubField2")
    
    class TestSignature(dspy.Signature):
        input1: InputField1 = dspy.InputField(desc="Nested input field")
        input2: float = dspy.InputField(desc="Float input field")
        output1: str = dspy.OutputField(desc="String output field")
        output2: OutputField2 = dspy.OutputField(desc="Nested output field")
        
    dspy.configure(lm=dspy.LM(model="openai/gpt4o"), adapter=dspy.JSONAdapter())
    program = dspy.Predict(TestSignature)

    with mock.patch("litellm.completion") as mock_completion:
        program(input1={"subfield1": 5, "subfield2": "Test input1", "subfield3": {"subsubfield1": 100, "subsubfield2": "Test input2"}}, input2=0.1)

    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args

    request_data = call_kwargs.get("messages")[1]

    assert request_data is not None

    input_message_content = request_data.get('content')
    assert input_message_content is not None

    input1_json_start = input_message_content.find('[[ ## input1 ## ]]')
    input1_json_end = input_message_content.find('[[ ## input2 ## ]]')

    assert input1_json_start != -1, "'[[ ## input1 ## ]]' not found in content"
    assert input1_json_end != -1, "'[[ ## input2 ## ]]' not found in content"

    input1_json_str = input_message_content[input1_json_start + len('[[ ## input1 ## ]]'):input1_json_end].strip()
    input1_data = json.loads(input1_json_str)

    assert 'subfield1' in input1_data
    assert 'subfield2' in input1_data
    assert 'subfield3' in input1_data
    assert 'subsubfield1' in input1_data['subfield3']
    assert 'subsubfield2' in input1_data['subfield3']

    response_format = call_kwargs.get("response_format")
    assert response_format is not None

    assert "output1" in response_format.model_fields
    
    # Check that the output field "output2" is also in the response format
    assert "output2" in response_format.model_fields

    output2_field = response_format.model_fields["output2"]
    assert output2_field.annotation.__name__ == "OutputField2"
    assert "subfield1" in output2_field.annotation.__annotations__
    assert "subfield2" in output2_field.annotation.__annotations__

    subfield2 = output2_field.annotation.__annotations__["subfield2"]
    assert "subsubfield1" in subfield2.__annotations__
    assert "subsubfield2" in subfield2.__annotations__
    


def test_json_adapter_sends_signature_info_correctly():
    class TestSignature(dspy.Signature):
        input1: str = dspy.InputField(desc="String input field")
        input2: float = dspy.InputField(desc="Float input field")
        output1: str = dspy.OutputField(desc="string output field")

    dspy.configure(lm=dspy.LM(model="openai/gpt4o"), adapter=dspy.JSONAdapter())
    program = dspy.Predict(TestSignature)

    with mock.patch("litellm.completion") as mock_completion:
        program(input1="Test input", input2=0.1)

    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args

    assert len(call_kwargs["messages"]) == 2
    assert call_kwargs["messages"][0]["role"] == "system"
    content = call_kwargs["messages"][0]["content"]
    assert content is not None
    assert "1. `input1` (str)" in content
    assert "2. `input2` (float)" in content
    assert "1. `output1` (str)" in content

    assert call_kwargs["messages"][1]["role"] == "user"
    content = call_kwargs["messages"][1]["content"]
    assert content is not None
    assert "[[ ## input1 ## ]]\nTest input" in content
    assert "[[ ## input2 ## ]]\n0.1" in content
    assert "`output1`" in content


def test_json_adapter_parse_invalid_field_keys():
    signature = dspy.make_signature("input1->output1")
    adapter = dspy.JSONAdapter()
    invalid_completion = '{"output": "Test output"}'
    with pytest.raises(ValueError, match=r"Expected"):
        adapter.parse(signature, invalid_completion)