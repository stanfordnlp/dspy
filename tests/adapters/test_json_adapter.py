from unittest import mock

import pydantic
import pytest
from litellm.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse

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
    dspy.configure(lm=dspy.LM(model="openai/gpt-4o"), adapter=dspy.JSONAdapter())
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


def test_json_adapter_not_using_structured_outputs_when_not_supported_by_model():
    class TestSignature(dspy.Signature):
        input1: str = dspy.InputField()
        output1: str = dspy.OutputField()
        output2: bool = dspy.OutputField()

    program = dspy.Predict(TestSignature)

    # Configure DSPy to use a model from a fake provider that doesn't support structured outputs
    dspy.configure(lm=dspy.LM(model="fakeprovider/fakemodel", cache=False), adapter=dspy.JSONAdapter())
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content=("{'output1': 'Test output', 'output2': True}")))],
            model="openai/gpt-4o",
        )

        program(input1="Test input")

    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args
    assert "response_format" not in call_kwargs


def test_json_adapter_falls_back_when_structured_outputs_fails():
    class TestSignature(dspy.Signature):
        input1: str = dspy.InputField()
        output1: str = dspy.OutputField(desc="String output field")

    dspy.configure(lm=dspy.LM(model="openai/gpt4o", cache=False), adapter=dspy.JSONAdapter())
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
                        content="{'answer': {'analysis': 'Paris is the capital of France', 'result': 'Paris'}}"
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
        assert result.answer.analysis == "Paris is the capital of France"
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
    expected_image_content = {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
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

    assert {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}} in messages[1]["content"]
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}} in messages[3]["content"]
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image3.jpg"}} in messages[5]["content"]


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

    expected_image1_content = {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}}
    expected_image2_content = {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}
    expected_image3_content = {"type": "image_url", "image_url": {"url": "https://example.com/image3.jpg"}}

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
    expected_image1_content = {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}}
    expected_image2_content = {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}
    expected_image3_content = {"type": "image_url", "image_url": {"url": "https://example.com/image3.jpg"}}
    assert expected_image1_content in messages[1]["content"]
    assert expected_image2_content in messages[1]["content"]
    assert expected_image3_content in messages[1]["content"]

    # The query image is formatted in the last user message
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image4.jpg"}} in messages[-1]["content"]


def test_json_adapter_with_tool():
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

    adapter = dspy.JSONAdapter()
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

    with mock.patch("litellm.completion") as mock_completion:
        lm = dspy.LM(model="openai/gpt-4o-mini")
        adapter(lm, {}, MySignature, [], {"question": "What is the weather in Tokyo?", "tools": tools})

    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args

    # Assert tool calls are included in the `tools` arg
    assert len(call_kwargs["tools"]) > 0
    assert call_kwargs["tools"][0] == {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                    },
                },
                "required": ["city"],
            },
        },
    }
    assert call_kwargs["tools"][1] == {
        "type": "function",
        "function": {
            "name": "get_population",
            "description": "Get the population for a country",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                    },
                    "year": {
                        "type": "integer",
                    },
                },
                "required": ["country", "year"],
            },
        },
    }


def test_json_adapter_with_code():
    # Test with code as input field
    class CodeAnalysis(dspy.Signature):
        """Analyze the time complexity of the code"""

        code: dspy.Code = dspy.InputField()
        result: str = dspy.OutputField()

    adapter = dspy.JSONAdapter()
    messages = adapter.format(CodeAnalysis, [], {"code": "print('Hello, world!')"})

    assert len(messages) == 2

    # The output field type description should be included in the system message even if the output field is nested
    assert dspy.Code.description() in messages[0]["content"]

    # The user message should include the question and the tools
    assert "print('Hello, world!')" in messages[1]["content"]

    # Test with code as output field
    class CodeGeneration(dspy.Signature):
        """Generate code to answer the question"""

        question: str = dspy.InputField()
        code: dspy.Code = dspy.OutputField()

    adapter = dspy.JSONAdapter()
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'code': 'print(\"Hello, world!\")'}"))],
            model="openai/gpt-4o-mini",
        )
        result = adapter(
            dspy.LM(model="openai/gpt-4o-mini", cache=False),
            {},
            CodeGeneration,
            [],
            {"question": "Write a python program to print 'Hello, world!'"},
        )
        assert result[0]["code"].code == 'print("Hello, world!")'


def test_json_adapter_formats_conversation_history():
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        answer: str = dspy.OutputField()

    history = dspy.History(
        messages=[
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is the capital of Germany?", "answer": "Berlin"},
        ]
    )

    adapter = dspy.JSONAdapter()
    messages = adapter.format(MySignature, [], {"question": "What is the capital of France?", "history": history})

    assert len(messages) == 6
    assert messages[1]["content"] == "[[ ## question ## ]]\nWhat is the capital of France?"
    assert messages[2]["content"] == '{\n  "answer": "Paris"\n}'
    assert messages[3]["content"] == "[[ ## question ## ]]\nWhat is the capital of Germany?"
    assert messages[4]["content"] == '{\n  "answer": "Berlin"\n}'


@pytest.mark.asyncio
async def test_json_adapter_on_pydantic_model_async():
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

    with mock.patch("litellm.acompletion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    message=Message(
                        content="{'answer': {'analysis': 'Paris is the capital of France', 'result': 'Paris'}}"
                    )
                )
            ],
            model="openai/gpt4o",
        )

        with dspy.context(lm=dspy.LM(model="openai/gpt4o", cache=False), adapter=dspy.JSONAdapter()):
            result = await program.acall(
                user={"id": 5, "name": "name_test", "email": "email_test"}, question="What is the capital of France?"
            )

        # Check that litellm.acompletion was called exactly once
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
        assert result.answer.analysis == "Paris is the capital of France"
        assert result.answer.result == "Paris"


def test_json_adapter_fallback_to_json_mode_on_structured_output_failure():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField(desc="String output field")

    dspy.configure(lm=dspy.LM(model="openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter())
    program = dspy.Predict(TestSignature)

    with mock.patch("litellm.completion") as mock_completion:
        # First call raises error to simulate structured output failure, second call returns a valid response
        mock_completion.side_effect = [
            RuntimeError("Structured output failed!"),
            ModelResponse(choices=[Choices(message=Message(content="{'answer': 'Test output'}"))]),
        ]

        result = program(question="Dummy question!")
        # The parse should succeed on the second call
        assert mock_completion.call_count == 2
        assert result.answer == "Test output"

        # The first call should have tried structured output
        _, first_call_kwargs = mock_completion.call_args_list[0]
        assert issubclass(first_call_kwargs.get("response_format"), pydantic.BaseModel)

        # The second call should have used JSON mode
        _, second_call_kwargs = mock_completion.call_args_list[1]
        assert second_call_kwargs.get("response_format") == {"type": "json_object"}


@pytest.mark.asyncio
async def test_json_adapter_fallback_to_json_mode_on_structured_output_failure_async():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField(desc="String output field")

    program = dspy.Predict(TestSignature)

    with mock.patch("litellm.acompletion") as mock_acompletion:
        # First call raises error to simulate structured output failure, second call returns a valid response
        mock_acompletion.side_effect = [
            RuntimeError("Structured output failed!"),
            ModelResponse(choices=[Choices(message=Message(content="{'answer': 'Test output'}"))]),
        ]

        with dspy.context(lm=dspy.LM(model="openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter()):
            result = await program.acall(question="Dummy question!")
        # The parse should succeed on the second call
        assert mock_acompletion.call_count == 2
        assert result.answer == "Test output"

        # The first call should have tried structured output
        _, first_call_kwargs = mock_acompletion.call_args_list[0]
        assert issubclass(first_call_kwargs.get("response_format"), pydantic.BaseModel)

        # The second call should have used JSON mode
        _, second_call_kwargs = mock_acompletion.call_args_list[1]
        assert second_call_kwargs.get("response_format") == {"type": "json_object"}


def test_error_message_on_json_adapter_failure():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField(desc="String output field")

    program = dspy.Predict(TestSignature)

    dspy.configure(lm=dspy.LM(model="openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter())

    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.side_effect = RuntimeError("RuntimeError!")

        with pytest.raises(RuntimeError) as error:
            program(question="Dummy question!")

        assert "RuntimeError!" in str(error.value)

        mock_completion.side_effect = ValueError("ValueError!")
        with pytest.raises(ValueError) as error:
            program(question="Dummy question!")

        assert "ValueError!" in str(error.value)


@pytest.mark.asyncio
async def test_error_message_on_json_adapter_failure_async():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField(desc="String output field")

    program = dspy.Predict(TestSignature)

    with mock.patch("litellm.acompletion") as mock_acompletion:
        with dspy.context(lm=dspy.LM(model="openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter()):
            mock_acompletion.side_effect = RuntimeError("RuntimeError!")
            with pytest.raises(RuntimeError) as error:
                await program.acall(question="Dummy question!")

            assert "RuntimeError!" in str(error.value)

            mock_acompletion.side_effect = ValueError("ValueError!")
            with pytest.raises(ValueError) as error:
                await program.acall(question="Dummy question!")

            assert "ValueError!" in str(error.value)


def test_json_adapter_toolcalls_native_function_calling():
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        answer: str = dspy.OutputField()
        tool_calls: dspy.ToolCalls = dspy.OutputField()

    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny"

    tools = [dspy.Tool(get_weather)]

    adapter = dspy.JSONAdapter(use_native_function_calling=True)

    # Case 1: Tool calls are present in the response, while content is None.
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                function=Function(arguments='{"city":"Paris"}', name="get_weather"),
                                id="call_pQm8ajtSMxgA0nrzK2ivFmxG",
                                type="function",
                            )
                        ],
                    ),
                ),
            ],
            model="openai/gpt-4o-mini",
        )
        result = adapter(
            dspy.LM(model="openai/gpt-4o-mini", cache=False),
            {},
            MySignature,
            [],
            {"question": "What is the weather in Paris?", "tools": tools},
        )

        assert result[0]["tool_calls"] == dspy.ToolCalls(
            tool_calls=[dspy.ToolCalls.ToolCall(name="get_weather", args={"city": "Paris"})]
        )
        # `answer` is not present, so we set it to None
        assert result[0]["answer"] is None

    # Case 2: Tool calls are not present in the response, while content is present.
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'answer': 'Paris'}"))],
            model="openai/gpt-4o-mini",
        )
        result = adapter(
            dspy.LM(model="openai/gpt-4o-mini", cache=False),
            {},
            MySignature,
            [],
            {"question": "What is the weather in Paris?", "tools": tools},
        )
        assert result[0]["answer"] == "Paris"
        assert result[0]["tool_calls"] is None


def test_json_adapter_toolcalls_no_native_function_calling():
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        answer: str = dspy.OutputField()
        tool_calls: dspy.ToolCalls = dspy.OutputField()

    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny"

    tools = [dspy.Tool(get_weather)]

    # Patch _get_structured_outputs_response_format to track calls
    with mock.patch("dspy.adapters.json_adapter._get_structured_outputs_response_format") as mock_structured:
        # Patch litellm.completion to return a dummy response
        with mock.patch("litellm.completion") as mock_completion:
            mock_completion.return_value = ModelResponse(
                choices=[Choices(message=Message(content="{'answer': 'sunny', 'tool_calls': {'tool_calls': []}}"))],
                model="openai/gpt-4o-mini",
            )
            adapter = dspy.JSONAdapter(use_native_function_calling=False)
            lm = dspy.LM(model="openai/gpt-4o-mini", cache=False)
            adapter(lm, {}, MySignature, [], {"question": "What is the weather in Tokyo?", "tools": tools})

        # _get_structured_outputs_response_format is not called because without using native function calling,
        # JSONAdapter falls back to json mode for stable quality.
        mock_structured.assert_not_called()
        mock_completion.assert_called_once()
        _, call_kwargs = mock_completion.call_args
        assert call_kwargs["response_format"] == {"type": "json_object"}
