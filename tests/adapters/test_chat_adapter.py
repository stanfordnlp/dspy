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
    print(system_content)
    print("\n\n\n\n\n")
    print(user_content)
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
    signature = dspy.make_signature("question->answer")
    adapter = dspy.ChatAdapter()
    invalid_completion = "{'output':'mismatched value'}"
    with pytest.raises(ValueError) as error:
        adapter.parse(signature, invalid_completion)
