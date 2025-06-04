from unittest import mock

import pydantic
import pytest
from litellm.utils import Choices, Message, ModelResponse

import dspy


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


def test_json_adapter_on_pydantic_model():
    from litellm.utils import Choices, Message, ModelResponse

    class User(pydantic.BaseModel):
        id: int
        name: str
        email: str

    class Answer(pydantic.BaseModel):
        analysis: str
        result: str

    class TestSignature(dspy.Signature):
        user: User = dspy.InputField(desc="The user who asks the question")
        question: str = dspy.InputField(desc="Question the user asks")
        answer: Answer = dspy.OutputField(desc="Answer to this question")

    program = dspy.Predict(TestSignature)

    dspy.configure(lm=dspy.LM(model="openai/gpt4o", cache=False), adapter=dspy.JSONAdapter())

    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    message=Message(
                        content="{'answer': {'analysis': 'Paris is the captial of France', 'result': 'Paris'}}"
                    )
                )
            ],
            model="openai/gpt4o",
        )
        result = program(
            user={"id": 5, "name": "name_test", "email": "email_test"}, question="What is the capital of France?"
        )

        # Check that litellm.completion was called exactly once
        mock_completion.assert_called_once()

        _, call_kwargs = mock_completion.call_args
        # Assert that there are exactly 2 messages (system + user)
        assert len(call_kwargs["messages"]) == 2

        assert call_kwargs["messages"][0]["role"] == "system"
        content = call_kwargs["messages"][0]["content"]
        assert content is not None

        # Assert that system prompt includes correct input field descriptions
        expected_input_fields = (
            "1. `user` (User): The user who asks the question\n2. `question` (str): Question the user asks\n"
        )
        assert expected_input_fields in content

        # Assert that system prompt includes correct output field description
        expected_output_fields = "1. `answer` (Answer): Answer to this question\n"
        assert expected_output_fields in content

        # Assert that system prompt includes input formatting structure
        expected_input_structure = "[[ ## user ## ]]\n{user}\n\n[[ ## question ## ]]\n{question}\n\n"
        assert expected_input_structure in content

        # Assert that system prompt includes output formatting structure
        expected_output_structure = (
            "Outputs will be a JSON object with the following fields.\n\n{\n  "
            '"answer": "{answer}        # note: the value you produce must adhere to the JSON schema: '
            '{\\"type\\": \\"object\\", \\"properties\\": {\\"analysis\\": {\\"type\\": \\"string\\", \\"title\\": '
            '\\"Analysis\\"}, \\"result\\": {\\"type\\": \\"string\\", \\"title\\": \\"Result\\"}}, \\"required\\": '
            '[\\"analysis\\", \\"result\\"], \\"title\\": \\"Answer\\"}"\n}'
        )
        assert expected_output_structure in content

        assert call_kwargs["messages"][1]["role"] == "user"
        user_message_content = call_kwargs["messages"][1]["content"]
        assert user_message_content is not None

        # Assert that the user input data is formatted correctly
        expected_input_data = (
            '[[ ## user ## ]]\n{"id": 5, "name": "name_test", "email": "email_test"}\n\n[[ ## question ## ]]\n'
            "What is the capital of France?\n\n"
        )
        assert expected_input_data in user_message_content

        # Assert that the adapter output has expected fields and values
        assert result.answer.analysis == "Paris is the captial of France"
        assert result.answer.result == "Paris"


def test_json_adapter_parse_raise_error_on_mismatch_fields():
    signature = dspy.make_signature("question->answer")
    adapter = dspy.JSONAdapter()
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(message=Message(content="{'answer1': 'Paris'}")),
            ],
            model="openai/gpt4o",
        )
        lm = dspy.LM(model="openai/gpt-4o-mini")
        with pytest.raises(dspy.utils.exceptions.AdapterParseError) as e:
            adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})

    assert e.value.adapter_name == "JSONAdapter"
    assert e.value.signature == signature
    assert e.value.lm_response == "{'answer1': 'Paris'}"
    assert e.value.parsed_result == {}

    assert str(e.value) == (
        "Adapter JSONAdapter failed to parse the LM response. \n\n"
        "LM Response: {'answer1': 'Paris'} \n\n"
        "Expected to find output fields in the LM response: [answer] \n\n"
        "Actual output fields parsed from the LM response: [] \n\n"
    )


def test_json_adapter_formats_image():
    # Test basic image formatting
    image = dspy.Image(url="https://example.com/image.jpg")

    class MySignature(dspy.Signature):
        image: dspy.Image = dspy.InputField()
        text: str = dspy.OutputField()

    adapter = dspy.JSONAdapter()
    messages = adapter.format(MySignature, [], {"image": image})

    assert len(messages) == 2
    user_message_content = messages[1]["content"]
    assert user_message_content is not None

    # The message should have 3 chunks of types: text, image_url, text
    assert len(user_message_content) == 3
    assert user_message_content[0]["type"] == "text"
    assert user_message_content[2]["type"] == "text"

    # Assert that the image is formatted correctly
    expected_image_content = {"type": "input_image", "image_url": "https://example.com/image.jpg"}
    assert expected_image_content in user_message_content


def test_json_adapter_formats_image_with_few_shot_examples():
    class MySignature(dspy.Signature):
        image: dspy.Image = dspy.InputField()
        text: str = dspy.OutputField()

    adapter = dspy.JSONAdapter()

    demos = [
        dspy.Example(
            image=dspy.Image(url="https://example.com/image1.jpg"),
            text="This is a test image",
        ),
        dspy.Example(
            image=dspy.Image(url="https://example.com/image2.jpg"),
            text="This is another test image",
        ),
    ]
    messages = adapter.format(MySignature, demos, {"image": dspy.Image(url="https://example.com/image3.jpg")})

    # 1 system message, 2 few shot examples (1 user and assistant message for each example), 1 user message
    assert len(messages) == 6

    assert {"type": "input_image", "image_url": "https://example.com/image1.jpg"} in messages[1]["content"]
    assert {"type": "input_image", "image_url": "https://example.com/image2.jpg"} in messages[3]["content"]
    assert {"type": "input_image", "image_url": "https://example.com/image3.jpg"} in messages[5]["content"]



def test_json_adapter_formats_image_with_nested_images():
    class ImageWrapper(pydantic.BaseModel):
        images: list[dspy.Image]
        tag: list[str]

    class MySignature(dspy.Signature):
        image: ImageWrapper = dspy.InputField()
        text: str = dspy.OutputField()

    image1 = dspy.Image(url="https://example.com/image1.jpg")
    image2 = dspy.Image(url="https://example.com/image2.jpg")
    image3 = dspy.Image(url="https://example.com/image3.jpg")

    image_wrapper = ImageWrapper(images=[image1, image2, image3], tag=["test", "example"])

    adapter = dspy.JSONAdapter()
    messages = adapter.format(MySignature, [], {"image": image_wrapper})

    expected_image1_content = {"type": "input_image", "image_url": "https://example.com/image1.jpg"}
    expected_image2_content = {"type": "input_image", "image_url": "https://example.com/image2.jpg"}
    expected_image3_content = {"type": "input_image", "image_url": "https://example.com/image3.jpg"}

    assert expected_image1_content in messages[1]["content"]
    assert expected_image2_content in messages[1]["content"]
    assert expected_image3_content in messages[1]["content"]


def test_json_adapter_formats_image_with_few_shot_examples_with_nested_images():
    class ImageWrapper(pydantic.BaseModel):
        images: list[dspy.Image]
        tag: list[str]

    class MySignature(dspy.Signature):
        image: ImageWrapper = dspy.InputField()
        text: str = dspy.OutputField()

    image1 = dspy.Image(url="https://example.com/image1.jpg")
    image2 = dspy.Image(url="https://example.com/image2.jpg")
    image3 = dspy.Image(url="https://example.com/image3.jpg")

    image_wrapper = ImageWrapper(images=[image1, image2, image3], tag=["test", "example"])
    demos = [
        dspy.Example(
            image=image_wrapper,
            text="This is a test image",
        ),
    ]

    image_wrapper_2 = ImageWrapper(images=[dspy.Image(url="https://example.com/image4.jpg")], tag=["test", "example"])
    adapter = dspy.JSONAdapter()
    messages = adapter.format(MySignature, demos, {"image": image_wrapper_2})

    assert len(messages) == 4

    # Image information in the few-shot example's user message
    expected_image1_content = {"type": "input_image", "image_url": "https://example.com/image1.jpg"}
    expected_image2_content = {"type": "input_image", "image_url": "https://example.com/image2.jpg"}
    expected_image3_content = {"type": "input_image", "image_url": "https://example.com/image3.jpg"}
    assert expected_image1_content in messages[1]["content"]
    assert expected_image2_content in messages[1]["content"]
    assert expected_image3_content in messages[1]["content"]

    # The query image is formatted in the last user message
    assert {"type": "input_image", "image_url": "https://example.com/image4.jpg"} in messages[-1]["content"]
