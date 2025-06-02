from typing import Literal
from unittest import mock

import pydantic
import pytest

import dspy


@pytest.mark.parametrize(
    "input_literal, output_literal, input_value, expected_input_str, expected_output_str",
    [
        # Scenario 1: double quotes escaped within strings
        (
            Literal["one", "two", 'three"'],
            Literal["four", "five", 'six"'],
            "two",
            "Literal['one', 'two', 'three\"']",
            "Literal['four', 'five', 'six\"']",
        ),
        # Scenario 2: Single quotes inside strings
        (
            Literal["she's here", "okay", "test"],
            Literal["done", "maybe'soon", "later"],
            "she's here",
            "Literal[\"she's here\", 'okay', 'test']",
            "Literal['done', \"maybe'soon\", 'later']",
        ),
        # Scenario 3: Strings containing both single and double quotes
        (
            Literal["both\"and'", "another"],
            Literal["yet\"another'", "plain"],
            "another",
            "Literal['both\"and\\'', 'another']",
            "Literal['yet\"another\\'', 'plain']",
        ),
        # Scenario 4: No quotes at all (check the default)
        (
            Literal["foo", "bar"],
            Literal["baz", "qux"],
            "foo",
            "Literal['foo', 'bar']",
            "Literal['baz', 'qux']",
        ),
        # Scenario 5: Mixed types
        (
            Literal[1, "bar"],
            Literal[True, 3, "foo"],
            "bar",
            "Literal[1, 'bar']",
            "Literal[True, 3, 'foo']",
        ),
    ],
)
def test_chat_adapter_quotes_literals_as_expected(
    input_literal, output_literal, input_value, expected_input_str, expected_output_str
):
    """
    This test verifies that when we declare Literal fields with various mixes
    of single/double quotes, the generated content string includes those
    Literals exactly as we want them to appear (like IPython does).
    """

    class TestSignature(dspy.Signature):
        input_text: input_literal = dspy.InputField()
        output_text: output_literal = dspy.OutputField()

    program = dspy.Predict(TestSignature)

    dspy.configure(lm=dspy.LM(model="openai/gpt4o"), adapter=dspy.ChatAdapter())

    with mock.patch("litellm.completion") as mock_completion:
        program(input_text=input_value)

    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args
    content = call_kwargs["messages"][0]["content"]

    assert expected_input_str in content
    assert expected_output_str in content


def test_chat_adapter_sync_call():
    signature = dspy.make_signature("question->answer")
    adapter = dspy.ChatAdapter()
    lm = dspy.utils.DummyLM([{"answer": "Paris"}])
    result = adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})
    assert result == [{"answer": "Paris"}]


@pytest.mark.asyncio
async def test_chat_adapter_async_call():
    signature = dspy.make_signature("question->answer")
    adapter = dspy.ChatAdapter()
    lm = dspy.utils.DummyLM([{"answer": "Paris"}])
    result = await adapter.acall(lm, {}, signature, [], {"question": "What is the capital of France?"})
    assert result == [{"answer": "Paris"}]


def test_chat_adapter_with_pydantic_models():
    """
    This test verifies that ChatAdapter can handle different input and output field types, both basic and nested.
    """

    class DogClass(pydantic.BaseModel):
        dog_breeds: list[str] = pydantic.Field(description="List of the breeds of dogs")
        num_dogs: int = pydantic.Field(description="Number of dogs the owner has", ge=0, le=10)

    class PetOwner(pydantic.BaseModel):
        name: str = pydantic.Field(description="Name of the owner")
        num_pets: int = pydantic.Field(description="Amount of pets the owner has", ge=0, le=100)
        dogs: DogClass = pydantic.Field(description="Nested Pydantic class with dog specific information ")

    class Answer(pydantic.BaseModel):
        result: str
        analysis: str

    class TestSignature(dspy.Signature):
        owner: PetOwner = dspy.InputField()
        question: str = dspy.InputField()
        output: Answer = dspy.OutputField()

    dspy.configure(lm=dspy.LM(model="openai/gpt4o"), adapter=dspy.ChatAdapter())
    program = dspy.Predict(TestSignature)

    with mock.patch("litellm.completion") as mock_completion:
        program(
            owner=PetOwner(name="John", num_pets=5, dogs=DogClass(dog_breeds=["labrador", "chihuahua"], num_dogs=2)),
            question="How many non-dog pets does John have?",
        )

    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args

    system_content = call_kwargs["messages"][0]["content"]
    user_content = call_kwargs["messages"][1]["content"]
    assert "1. `owner` (PetOwner)" in system_content
    assert "2. `question` (str)" in system_content
    assert "1. `output` (Answer)" in system_content

    assert "name" in user_content
    assert "num_pets" in user_content
    assert "dogs" in user_content
    assert "dog_breeds" in user_content
    assert "num_dogs" in user_content
    assert "How many non-dog pets does John have?" in user_content


def test_chat_adapter_signature_information():
    """
    This test ensures that the signature information sent to the LM follows an expected format.
    """

    class TestSignature(dspy.Signature):
        input1: str = dspy.InputField(desc="String Input")
        input2: int = dspy.InputField(desc="Integer Input")
        output: str = dspy.OutputField(desc="String Output")

    dspy.configure(lm=dspy.LM(model="openai/gpt4o"), adapter=dspy.ChatAdapter())
    program = dspy.Predict(TestSignature)

    with mock.patch("litellm.completion") as mock_completion:
        program(input1="Test", input2=11)

    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args

    assert len(call_kwargs["messages"]) == 2
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1]["role"] == "user"

    system_content = call_kwargs["messages"][0]["content"]
    user_content = call_kwargs["messages"][1]["content"]

    assert "1. `input1` (str)" in system_content
    assert "2. `input2` (int)" in system_content
    assert "1. `output` (str)" in system_content
    assert "[[ ## input1 ## ]]\n{input1}" in system_content
    assert "[[ ## input2 ## ]]\n{input2}" in system_content
    assert "[[ ## output ## ]]\n{output}" in system_content
    assert "[[ ## completed ## ]]" in system_content

    assert "[[ ## input1 ## ]]" in user_content
    assert "[[ ## input2 ## ]]" in user_content
    assert "[[ ## output ## ]]" in user_content
    assert "[[ ## completed ## ]]" in user_content


def test_chat_adapter_exception_raised_on_failure():
    """
    This test ensures that on an error, ChatAdapter raises an explicit exception.
    """
    signature = dspy.make_signature("question->answer")
    adapter = dspy.ChatAdapter()
    invalid_completion = "{'output':'mismatched value'}"
    with pytest.raises(dspy.utils.exceptions.AdapterParseError, match="Adapter ChatAdapter failed to parse*"):
        adapter.parse(signature, invalid_completion)


def test_chat_adapter_formats_image():
    # Test basic image formatting
    image = dspy.Image(url="https://example.com/image.jpg")

    class MySignature(dspy.Signature):
        image: dspy.Image = dspy.InputField()
        text: str = dspy.OutputField()

    adapter = dspy.ChatAdapter()
    messages = adapter.format(MySignature, [], {"image": image})

    assert len(messages) == 2
    user_message_content = messages[1]["content"]
    assert user_message_content is not None

    # The message should have 3 chunks of types: text, image_url, text
    assert len(user_message_content) == 3
    assert user_message_content[0]["type"] == "text"
    assert user_message_content[2]["type"] == "text"

    # Assert that the image is formatted correctly
    expected_image_content = {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    assert expected_image_content in user_message_content


def test_chat_adapter_formats_image_with_few_shot_examples():
    class MySignature(dspy.Signature):
        image: dspy.Image = dspy.InputField()
        text: str = dspy.OutputField()

    adapter = dspy.ChatAdapter()

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

    assert {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}} in messages[1]["content"]
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}} in messages[3]["content"]
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image3.jpg"}} in messages[5]["content"]


def test_chat_adapter_formats_image_with_nested_images():
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

    adapter = dspy.ChatAdapter()
    messages = adapter.format(MySignature, [], {"image": image_wrapper})

    expected_image1_content = {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}}
    expected_image2_content = {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}
    expected_image3_content = {"type": "image_url", "image_url": {"url": "https://example.com/image3.jpg"}}

    assert expected_image1_content in messages[1]["content"]
    assert expected_image2_content in messages[1]["content"]
    assert expected_image3_content in messages[1]["content"]


def test_chat_adapter_formats_image_with_few_shot_examples_with_nested_images():
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
    adapter = dspy.ChatAdapter()
    messages = adapter.format(MySignature, demos, {"image": image_wrapper_2})

    assert len(messages) == 4

    # Image information in the few-shot example's user message
    expected_image1_content = {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}}
    expected_image2_content = {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}
    expected_image3_content = {"type": "image_url", "image_url": {"url": "https://example.com/image3.jpg"}}
    assert expected_image1_content in messages[1]["content"]
    assert expected_image2_content in messages[1]["content"]
    assert expected_image3_content in messages[1]["content"]

    # The query image is formatted in the last user message
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image4.jpg"}} in messages[-1]["content"]


def test_chat_adapter_with_tool():
    class MySignature(dspy.Signature):
        """Answer question with the help of the tools"""

        question: str = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        answer: str = dspy.OutputField()
        tool_calls: dspy.ToolCalls = dspy.OutputField()

    def get_weather(city: str) -> str:
        """Get the weather for a city"""
        return f"The weather in {city} is sunny"

    def get_population(country: str, year: int) -> str:
        """Get the population for a country"""
        return f"The population of {country} in {year} is 1000000"

    tools = [dspy.Tool(get_weather), dspy.Tool(get_population)]

    adapter = dspy.ChatAdapter()
    messages = adapter.format(MySignature, [], {"question": "What is the weather in Tokyo?", "tools": tools})

    assert len(messages) == 2

    # The output field type description should be included in the system message even if the output field is nested
    assert dspy.ToolCalls.description() in messages[0]["content"]

    # The user message should include the question and the tools
    assert "What is the weather in Tokyo?" in messages[1]["content"]
    assert "get_weather" in messages[1]["content"]
    assert "get_population" in messages[1]["content"]

    # Tool arguments format should be included in the user message
    assert "{'city': {'type': 'string'}}" in messages[1]["content"]
    assert "{'country': {'type': 'string'}, 'year': {'type': 'integer'}}" in messages[1]["content"]
