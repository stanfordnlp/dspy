import re
from typing import Literal
from unittest import mock

import pydantic
import pytest
from litellm.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse

import dspy
from dspy.experimental import Citations, Document
from tests.adapters.conftest import format_messages_and_lm_kwargs


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

    dspy.configure(lm=dspy.LM(model="openai/gpt-4o"), adapter=dspy.ChatAdapter())

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


def test_chat_adapter_format_exact_messages_for_simple_signature():
    class QA(dspy.Signature):
        """Answer the question."""

        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(dspy.ChatAdapter(), QA, [], {"question": "What is the capital of France?"})

    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs

    assert messages == [
        {
            "role": "system",
            "content": """Your input fields are:
1. `question` (str):
Your output fields are:
1. `answer` (str):
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}

[[ ## answer ## ]]
{answer}

[[ ## completed ## ]]
In adhering to this structure, your objective is:\x20
        Answer the question.""",
        },
        {
            "role": "user",
            "content": """[[ ## question ## ]]
What is the capital of France?

Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.""",
        },
    ]


def test_chat_adapter_format_exact_messages_with_demo_and_typed_outputs():
    class MultiAnswer(dspy.Signature):
        """Answer the question with multiple answers and scores"""

        question: str = dspy.InputField()
        answers: list[str] = dspy.OutputField()
        scores: list[float] = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(dspy.ChatAdapter(),
        MultiAnswer,
        demos=[{"question": "Q1", "answers": ["A1", "A2"], "scores": [0.1, 0.9]}],
        inputs={"question": "Q2"},
    )

    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs

    assert messages == [
        {
            "role": "system",
            "content": """Your input fields are:
1. `question` (str):
Your output fields are:
1. `answers` (list[str]):\x20
2. `scores` (list[float]):
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}

[[ ## answers ## ]]
{answers}        # note: the value you produce must adhere to the JSON schema: {"type": "array", "items": {"type": "string"}}

[[ ## scores ## ]]
{scores}        # note: the value you produce must adhere to the JSON schema: {"type": "array", "items": {"type": "number"}}

[[ ## completed ## ]]
In adhering to this structure, your objective is:\x20
        Answer the question with multiple answers and scores""",
        },
        {"role": "user", "content": """[[ ## question ## ]]
Q1"""},
        {
            "role": "assistant",
            "content": """[[ ## answers ## ]]
["A1", "A2"]

[[ ## scores ## ]]
[0.1, 0.9]

[[ ## completed ## ]]
""",
        },
        {
            "role": "user",
            "content": """[[ ## question ## ]]
Q2

Respond with the corresponding output fields, starting with the field `[[ ## answers ## ]]` (must be formatted as a valid Python list[str]), then `[[ ## scores ## ]]` (must be formatted as a valid Python list[float]), and then ending with the marker for `[[ ## completed ## ]]`.""",
        },
    ]


def test_chat_adapter_format_exact_messages_with_nested_pydantic_models():
    class Address(pydantic.BaseModel):
        city: str
        country: str

    class Person(pydantic.BaseModel):
        name: str
        address: Address
        tags: list[str]

    class Summary(pydantic.BaseModel):
        headline: str
        score: float

    class PydanticSignature(dspy.Signature):
        person: Person = dspy.InputField()
        summary: Summary = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(dspy.ChatAdapter(),
        PydanticSignature,
        [],
        {"person": Person(name="Ada", address=Address(city="London", country="UK"), tags=["math", "code"])},
    )

    expected_messages = [{"role": "system",
      "content": 'Your input fields are:\n'
                 '1. `person` (Person):\n'
                 'Your output fields are:\n'
                 '1. `summary` (Summary):\n'
                 'All interactions will be structured in the following way, with the appropriate '
                 'values filled in.\n'
                 '\n'
                 '[[ ## person ## ]]\n'
                 '{person}\n'
                 '\n'
                 '[[ ## summary ## ]]\n'
                 '{summary}        # note: the value you produce must adhere to the JSON schema: '
                 '{"type": "object", "properties": {"headline": {"type": "string", "title": '
                 '"Headline"}, "score": {"type": "number", "title": "Score"}}, "required": '
                 '["headline", "score"], "title": "Summary"}\n'
                 '\n'
                 '[[ ## completed ## ]]\n'
                 'In adhering to this structure, your objective is: \n'
                 '        Given the fields `person`, produce the fields `summary`.'},
     {"role": "user",
      "content": '[[ ## person ## ]]\n'
                 '{"name": "Ada", "address": {"city": "London", "country": "UK"}, "tags": ["math", '
                 '"code"]}\n'
                 '\n'
                 'Respond with the corresponding output fields, starting with the field `[[ ## summary '
                 '## ]]` (must be formatted as a valid Python Summary), and then ending with the '
                 'marker for `[[ ## completed ## ]]`.'}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_with_incomplete_demo():
    class IncompleteDemoSignature(dspy.Signature):
        question: str = dspy.InputField()
        context: str = dspy.InputField()
        answer: str = dspy.OutputField()
        confidence: float = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(dspy.ChatAdapter(),
        IncompleteDemoSignature,
        [{"question": "Q1", "answer": "A1"}],
        {"question": "Q2", "context": "C2"},
    )

    expected_messages = [{"role": "system",
      "content": "Your input fields are:\n"
                 "1. `question` (str): \n"
                 "2. `context` (str):\n"
                 "Your output fields are:\n"
                 "1. `answer` (str): \n"
                 "2. `confidence` (float):\n"
                 "All interactions will be structured in the following way, with the appropriate "
                 "values filled in.\n"
                 "\n"
                 "[[ ## question ## ]]\n"
                 "{question}\n"
                 "\n"
                 "[[ ## context ## ]]\n"
                 "{context}\n"
                 "\n"
                 "[[ ## answer ## ]]\n"
                 "{answer}\n"
                 "\n"
                 "[[ ## confidence ## ]]\n"
                 "{confidence}        # note: the value you produce must be a single float value\n"
                 "\n"
                 "[[ ## completed ## ]]\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Given the fields `question`, `context`, produce the fields `answer`, "
                 "`confidence`."},
     {"role": "user",
      "content": "This is an example of the task, though some input or output fields are not "
                 "supplied.\n"
                 "\n"
                 "[[ ## question ## ]]\n"
                 "Q1"},
     {"role": "assistant",
      "content": "[[ ## answer ## ]]\n"
                 "A1\n"
                 "\n"
                 "[[ ## confidence ## ]]\n"
                 "Not supplied for this particular example.\n"
                 "\n"
                 "[[ ## completed ## ]]\n"},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Q2\n"
                 "\n"
                 "[[ ## context ## ]]\n"
                 "C2\n"
                 "\n"
                 "Respond with the corresponding output fields, starting with the field `[[ ## answer "
                 "## ]]`, then `[[ ## confidence ## ]]` (must be formatted as a valid Python float), "
                 "and then ending with the marker for `[[ ## completed ## ]]`."}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_with_history():
    class HistorySignature(dspy.Signature):
        history: dspy.History = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    history = dspy.History(
        messages=[
            {"question": "What is 1+1?", "answer": "2"},
            {"question": "What is 2+2?", "answer": "4"},
        ]
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(dspy.ChatAdapter(),
        HistorySignature,
        [],
        {"history": history, "question": "What is 3+3?"},
    )

    expected_messages = [{"role": "system",
      "content": "Your input fields are:\n"
                 "1. `history` (History): \n"
                 "2. `question` (str):\n"
                 "Your output fields are:\n"
                 "1. `answer` (str):\n"
                 "All interactions will be structured in the following way, with the appropriate "
                 "values filled in.\n"
                 "\n"
                 "[[ ## history ## ]]\n"
                 "{history}\n"
                 "\n"
                 "[[ ## question ## ]]\n"
                 "{question}\n"
                 "\n"
                 "[[ ## answer ## ]]\n"
                 "{answer}\n"
                 "\n"
                 "[[ ## completed ## ]]\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Given the fields `history`, `question`, produce the fields `answer`."},
     {"role": "user", "content": "[[ ## question ## ]]\nWhat is 1+1?"},
     {"role": "assistant", "content": "[[ ## answer ## ]]\n2\n\n[[ ## completed ## ]]\n"},
     {"role": "user", "content": "[[ ## question ## ]]\nWhat is 2+2?"},
     {"role": "assistant", "content": "[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]\n"},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "What is 3+3?\n"
                 "\n"
                 "Respond with the corresponding output fields, starting with the field `[[ ## answer "
                 "## ]]`, and then ending with the marker for `[[ ## completed ## ]]`."}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_with_list_value_for_string_input():
    class ListAsStringSignature(dspy.Signature):
        context: str = dspy.InputField()
        answer: str = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(dspy.ChatAdapter(), ListAsStringSignature, [], {"context": ["alpha", "beta"]})

    expected_messages = [{"role": "system",
      "content": "Your input fields are:\n"
                 "1. `context` (str):\n"
                 "Your output fields are:\n"
                 "1. `answer` (str):\n"
                 "All interactions will be structured in the following way, with the appropriate "
                 "values filled in.\n"
                 "\n"
                 "[[ ## context ## ]]\n"
                 "{context}\n"
                 "\n"
                 "[[ ## answer ## ]]\n"
                 "{answer}\n"
                 "\n"
                 "[[ ## completed ## ]]\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Given the fields `context`, produce the fields `answer`."},
     {"role": "user",
      "content": "[[ ## context ## ]]\n"
                 "[1] «alpha»\n"
                 "[2] «beta»\n"
                 "\n"
                 "Respond with the corresponding output fields, starting with the field `[[ ## answer "
                 "## ]]`, and then ending with the marker for `[[ ## completed ## ]]`."}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_with_literal_output():
    class LiteralSignature(dspy.Signature):
        question: str = dspy.InputField()
        verdict: Literal["yes", "no"] = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(dspy.ChatAdapter(), LiteralSignature, [], {"question": "Is the sky blue?"})

    expected_messages = [{"role": "system",
      "content": "Your input fields are:\n"
                 "1. `question` (str):\n"
                 "Your output fields are:\n"
                 "1. `verdict` (Literal['yes', 'no']):\n"
                 "All interactions will be structured in the following way, with the appropriate "
                 "values filled in.\n"
                 "\n"
                 "[[ ## question ## ]]\n"
                 "{question}\n"
                 "\n"
                 "[[ ## verdict ## ]]\n"
                 "{verdict}        # note: the value you produce must exactly match (no extra "
                 "characters) one of: yes; no\n"
                 "\n"
                 "[[ ## completed ## ]]\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Given the fields `question`, produce the fields `verdict`."},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Is the sky blue?\n"
                 "\n"
                 "Respond with the corresponding output fields, starting with the field `[[ ## verdict "
                 "## ]]` (must be formatted as a valid Python Literal['yes', 'no']), and then ending "
                 "with the marker for `[[ ## completed ## ]]`."}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_with_multimodal_custom_type_inputs():
    class CustomTypeSignature(dspy.Signature):
        image: dspy.Image = dspy.InputField()
        audio: dspy.Audio = dspy.InputField()
        file: dspy.File = dspy.InputField()
        document: Document = dspy.InputField()
        answer: str = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(dspy.ChatAdapter(),
        CustomTypeSignature,
        [],
        {
            "image": dspy.Image("https://example.com/cat.png"),
            "audio": dspy.Audio(data="QUJD", audio_format="wav"),
            "file": dspy.File.from_file_id("file-123", filename="notes.txt"),
            "document": Document(data="Alpha beta", title="Doc"),
        },
    )

    expected_messages = [{"role": "system",
      "content": "Your input fields are:\n"
                 "1. `image` (Image): \n"
                 "2. `audio` (Audio): \n"
                 "3. `file` (File): \n"
                 "4. `document` (Document): \n"
                 "    Type description of Document: A document containing text content that can be "
                 "referenced and cited. Include the full text content and optionally a title for "
                 "proper referencing.\n"
                 "Your output fields are:\n"
                 "1. `answer` (str):\n"
                 "All interactions will be structured in the following way, with the appropriate "
                 "values filled in.\n"
                 "\n"
                 "[[ ## image ## ]]\n"
                 "{image}\n"
                 "\n"
                 "[[ ## audio ## ]]\n"
                 "{audio}\n"
                 "\n"
                 "[[ ## file ## ]]\n"
                 "{file}\n"
                 "\n"
                 "[[ ## document ## ]]\n"
                 "{document}\n"
                 "\n"
                 "[[ ## answer ## ]]\n"
                 "{answer}\n"
                 "\n"
                 "[[ ## completed ## ]]\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Given the fields `image`, `audio`, `file`, `document`, produce the fields "
                 "`answer`."},
     {"role": "user",
      "content": [{"type": "text", "text": "[[ ## image ## ]]\n"},
                  {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                  {"type": "text", "text": "\n\n[[ ## audio ## ]]\n"},
                  {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}},
                  {"type": "text", "text": "\n\n[[ ## file ## ]]\n"},
                  {"type": "file", "file": {"file_id": "file-123", "filename": "notes.txt"}},
                  {"type": "text", "text": "\n\n[[ ## document ## ]]\n"},
                  {"type": "document",
                   "source": {"type": "text", "media_type": "text/plain", "data": "Alpha beta"},
                   "citations": {"enabled": True},
                   "title": "Doc"},
                  {"type": "text",
                   "text": "\n"
                           "\n"
                           "Respond with the corresponding output fields, starting with the field `[[ "
                           "## answer ## ]]`, and then ending with the marker for `[[ ## completed ## "
                           "]]`."}]}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_with_history_demo_pydantic_tools_and_image():
    def search(query: str, k: int = 3) -> str:
        """Search for documents."""
        return query

    class Location(pydantic.BaseModel):
        city: str
        country: str

    class Profile(pydantic.BaseModel):
        name: str
        location: Location
        interests: list[str]

    class AnswerCard(pydantic.BaseModel):
        answer: str
        sources: list[str]

    class RichRenderingSignature(dspy.Signature):
        """Answer using all supplied context."""

        history: dspy.History = dspy.InputField()
        image: dspy.Image = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        profile: Profile = dspy.InputField()
        question: str = dspy.InputField()
        answer: AnswerCard = dspy.OutputField()

    tool = dspy.Tool(search)
    demo_profile = Profile(
        name="Ada",
        location=Location(city="London", country="UK"),
        interests=["math", "machines"],
    )
    current_profile = Profile(
        name="Grace",
        location=Location(city="Arlington", country="USA"),
        interests=["compilers", "navy"],
    )
    history = dspy.History(
        messages=[
            {
                "profile": demo_profile,
                "question": "Who is Ada?",
                "answer": AnswerCard(answer="Ada is a mathematician.", sources=["memory"]),
            }
        ]
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(dspy.ChatAdapter(),
        RichRenderingSignature,
        demos=[
            {
                "image": dspy.Image("https://example.com/demo.png"),
                "tools": [tool],
                "profile": demo_profile,
                "question": "What should we mention?",
                "answer": AnswerCard(answer="Mention analytical engines.", sources=["demo"]),
            }
        ],
        inputs={
            "history": history,
            "image": dspy.Image("https://example.com/current.png"),
            "tools": [tool],
            "profile": current_profile,
            "question": "What should the answer include?",
        },
    )

    expected_messages = [{"role": "system",
      "content": 'Your input fields are:\n'
                 '1. `history` (History): \n'
                 '2. `image` (Image): \n'
                 '3. `tools` (list[Tool]): \n'
                 '4. `profile` (Profile): \n'
                 '5. `question` (str):\n'
                 'Your output fields are:\n'
                 '1. `answer` (AnswerCard):\n'
                 'All interactions will be structured in the following way, with the appropriate '
                 'values filled in.\n'
                 '\n'
                 '[[ ## history ## ]]\n'
                 '{history}\n'
                 '\n'
                 '[[ ## image ## ]]\n'
                 '{image}\n'
                 '\n'
                 '[[ ## tools ## ]]\n'
                 '{tools}\n'
                 '\n'
                 '[[ ## profile ## ]]\n'
                 '{profile}\n'
                 '\n'
                 '[[ ## question ## ]]\n'
                 '{question}\n'
                 '\n'
                 '[[ ## answer ## ]]\n'
                 '{answer}        # note: the value you produce must adhere to the JSON schema: '
                 '{"type": "object", "properties": {"answer": {"type": "string", "title": "Answer"}, '
                 '"sources": {"type": "array", "items": {"type": "string"}, "title": "Sources"}}, '
                 '"required": ["answer", "sources"], "title": "AnswerCard"}\n'
                 '\n'
                 '[[ ## completed ## ]]\n'
                 'In adhering to this structure, your objective is: \n'
                 '        Answer using all supplied context.'},
     {"role": "user",
      "content": [{"type": "text",
                   "text": "This is an example of the task, though some input or output fields are not "
                           "supplied.\n"
                           "\n"
                           "[[ ## image ## ]]\n"},
                  {"type": "image_url", "image_url": {"url": "https://example.com/demo.png"}},
                  {"type": "text",
                   "text": '\n'
                           '\n'
                           '[[ ## tools ## ]]\n'
                           '["search, whose description is <desc>Search for documents.</desc>. It '
                           "takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', "
                           '\'default\': 3}}."]\n'
                           '\n'
                           '[[ ## profile ## ]]\n'
                           '{"name": "Ada", "location": {"city": "London", "country": "UK"}, '
                           '"interests": ["math", "machines"]}\n'
                           '\n'
                           '[[ ## question ## ]]\n'
                           'What should we mention?'}]},
     {"role": "assistant",
      "content": '[[ ## answer ## ]]\n'
                 '{"answer": "Mention analytical engines.", "sources": ["demo"]}\n'
                 '\n'
                 '[[ ## completed ## ]]\n'},
     {"role": "user",
      "content": '[[ ## profile ## ]]\n'
                 '{"name": "Ada", "location": {"city": "London", "country": "UK"}, "interests": '
                 '["math", "machines"]}\n'
                 '\n'
                 '[[ ## question ## ]]\n'
                 'Who is Ada?'},
     {"role": "assistant",
      "content": '[[ ## answer ## ]]\n'
                 '{"answer": "Ada is a mathematician.", "sources": ["memory"]}\n'
                 '\n'
                 '[[ ## completed ## ]]\n'},
     {"role": "user",
      "content": [{"type": "text", "text": "[[ ## image ## ]]\n"},
                  {"type": "image_url", "image_url": {"url": "https://example.com/current.png"}},
                  {"type": "text",
                   "text": '\n'
                           '\n'
                           '[[ ## tools ## ]]\n'
                           '["search, whose description is <desc>Search for documents.</desc>. It '
                           "takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', "
                           '\'default\': 3}}."]\n'
                           '\n'
                           '[[ ## profile ## ]]\n'
                           '{"name": "Grace", "location": {"city": "Arlington", "country": "USA"}, '
                           '"interests": ["compilers", "navy"]}\n'
                           '\n'
                           '[[ ## question ## ]]\n'
                           'What should the answer include?\n'
                           '\n'
                           'Respond with the corresponding output fields, starting with the field `[[ '
                           '## answer ## ]]` (must be formatted as a valid Python AnswerCard), and '
                           'then ending with the marker for `[[ ## completed ## ]]`.'}]}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs

def test_chat_adapter_format_exact_messages_with_base_custom_type_input():
    class Event(dspy.Type):
        label: str

        def format(self):
            return [{"type": "event", "event": {"label": self.label}}]

        @classmethod
        def description(cls):
            return "An event block."

    class EventSignature(dspy.Signature):
        event: Event = dspy.InputField()
        answer: str = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(dspy.ChatAdapter(), EventSignature, [], {"event": Event(label="launch")})

    expected_messages = [{"role": "system",
      "content": "Your input fields are:\n"
                 "1. `event` (Event): \n"
                 "    Type description of Event: An event block.\n"
                 "Your output fields are:\n"
                 "1. `answer` (str):\n"
                 "All interactions will be structured in the following way, with the appropriate "
                 "values filled in.\n"
                 "\n"
                 "[[ ## event ## ]]\n"
                 "{event}\n"
                 "\n"
                 "[[ ## answer ## ]]\n"
                 "{answer}\n"
                 "\n"
                 "[[ ## completed ## ]]\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Given the fields `event`, produce the fields `answer`."},
     {"role": "user",
      "content": [{"type": "text", "text": "[[ ## event ## ]]\n"},
                  {"type": "event", "event": {"label": "launch"}},
                  {"type": "text",
                   "text": "\n"
                           "\n"
                           "Respond with the corresponding output fields, starting with the field `[[ "
                           "## answer ## ]]`, and then ending with the marker for `[[ ## completed ## "
                           "]]`."}]}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs

def test_chat_adapter_format_exact_messages_with_citations_output_demo():
    class CitationSignature(dspy.Signature):
        question: str = dspy.InputField()
        citations: Citations = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(dspy.ChatAdapter(),
        CitationSignature,
        [
            {
                "question": "Q1",
                "citations": Citations.from_dict_list(
                    [
                        {
                            "cited_text": "alpha",
                            "document_index": 0,
                            "start_char_index": 0,
                            "end_char_index": 5,
                        }
                    ]
                ),
            }
        ],
        {"question": "Q2"},
    )

    expected_messages = [{"role": "system",
      "content": 'Your input fields are:\n'
                 '1. `question` (str):\n'
                 'Your output fields are:\n'
                 '1. `citations` (Citations): \n'
                 '    Type description of Citations: Citations with quoted text and source references. '
                 'Include the exact text being cited and information about its source.\n'
                 'All interactions will be structured in the following way, with the appropriate '
                 'values filled in.\n'
                 '\n'
                 '[[ ## question ## ]]\n'
                 '{question}\n'
                 '\n'
                 '[[ ## citations ## ]]\n'
                 '{citations}        # note: the value you produce must adhere to the JSON schema: '
                 '{"type": "object", "$defs": {"Citation": {"type": "object", "description": '
                 '"Individual citation with character location information.", "properties": {"type": '
                 '{"type": "string", "default": "char_location", "title": "Type"}, "cited_text": '
                 '{"type": "string", "title": "Cited Text"}, "document_index": {"type": "integer", '
                 '"title": "Document Index"}, "document_title": {"anyOf": [{"type": "string"}, '
                 '{"type": "null"}], "default": null, "title": "Document Title"}, "end_char_index": '
                 '{"type": "integer", "title": "End Char Index"}, "start_char_index": {"type": '
                 '"integer", "title": "Start Char Index"}, "supported_text": {"anyOf": [{"type": '
                 '"string"}, {"type": "null"}], "default": null, "title": "Supported Text"}}, '
                 '"required": ["cited_text", "document_index", "start_char_index", "end_char_index"], '
                 '"title": "Citation"}}, "description": "Experimental: This class may change or be '
                 'removed in a future release without warning (introduced in v3.0.4).\\n\\nCitations '
                 'extracted from an LM response with source references.\\n\\n    This type represents '
                 'citations returned by language models that support\\n    citation extraction, '
                 "particularly Anthropic's Citations API through LiteLLM.\\n    Citations include the "
                 'quoted text and source information.\\n\\n    Examples:\\n        ```python\\n        '
                 'import os\\n        import dspy\\n        from dspy.signatures import '
                 'Signature\\n        from dspy.experimental import Citations, Document\\n        '
                 'os.environ[\\"ANTHROPIC_API_KEY\\"] = \\"YOUR_ANTHROPIC_API_KEY\\"\\n\\n        '
                 "class AnswerWithSources(Signature):\\n            '''Answer questions using provided "
                 "documents with citations.'''\\n            documents: list[Document] = "
                 'dspy.InputField()\\n            question: str = dspy.InputField()\\n            '
                 'answer: str = dspy.OutputField()\\n            citations: Citations = '
                 'dspy.OutputField()\\n\\n        # Create documents to provide as sources\\n        '
                 'docs = [\\n            Document(\\n                data=\\"The Earth orbits the Sun '
                 'in an elliptical path.\\",\\n                title=\\"Basic Astronomy '
                 'Facts\\"\\n            ),\\n            Document(\\n                data=\\"Water '
                 'boils at 100°C at standard atmospheric pressure.\\",\\n                '
                 'title=\\"Physics Fundamentals\\",\\n                metadata={\\"author\\": \\"Dr. '
                 'Smith\\", \\"year\\": 2023}\\n            )\\n        ]\\n\\n        # Use with a '
                 'model that supports citations like Claude\\n        lm = '
                 'dspy.LM(\\"anthropic/claude-opus-4-1-20250805\\")\\n        predictor = '
                 'dspy.Predict(AnswerWithSources)\\n        result = predictor(documents=docs, '
                 'question=\\"What temperature does water boil?\\", lm=lm)\\n\\n        for citation '
                 'in result.citations.citations:\\n            print(citation.format())\\n        '
                 '```\\n    ", "properties": {"citations": {"type": "array", "items": {"$ref": '
                 '"#/$defs/Citation"}, "title": "Citations"}}, "required": ["citations"], "title": '
                 '"Citations"}\n'
                 '\n'
                 '[[ ## completed ## ]]\n'
                 'In adhering to this structure, your objective is: \n'
                 '        Given the fields `question`, produce the fields `citations`.'},
     {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
     {"role": "assistant",
      "content": '[[ ## citations ## ]]\n'
                 '<<CUSTOM-TYPE-START-IDENTIFIER>>[{"type": "char_location", "cited_text": "alpha", '
                 '"document_index": 0, "start_char_index": 0, "end_char_index": '
                 '5}]<<CUSTOM-TYPE-END-IDENTIFIER>>\n'
                 '\n'
                 '[[ ## completed ## ]]\n'},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Q2\n"
                 "\n"
                 "Respond with the corresponding output fields, starting with the field `[[ ## "
                 "citations ## ]]` (must be formatted as a valid Python Citations), and then ending "
                 "with the marker for `[[ ## completed ## ]]`."}]
    def normalize_citations_schema_description(content):
        return re.sub(
            r'"description": ".*?", "properties":',
            '"description": "<CITATIONS_SCHEMA_DESCRIPTION>", "properties":',
            content,
        )

    messages[0]["content"] = normalize_citations_schema_description(messages[0]["content"])
    expected_messages[0]["content"] = normalize_citations_schema_description(expected_messages[0]["content"])

    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs

def test_chat_adapter_format_exact_messages_and_lm_kwargs_with_native_citations():
    class AnthropicLM(dspy.utils.DummyLM):
        def __init__(self):
            super().__init__([{}])
            self.model = "anthropic/claude-3-5-sonnet"

    class CitationSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        citations: Citations = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(
        dspy.ChatAdapter(),
        CitationSignature,
        [],
        {"question": "Q?"},
        lm=AnthropicLM(),
    )

    expected_messages = [{"role": "system",
      "content": "Your input fields are:\n"
                 "1. `question` (str):\n"
                 "Your output fields are:\n"
                 "1. `answer` (str):\n"
                 "All interactions will be structured in the following way, with the appropriate "
                 "values filled in.\n"
                 "\n"
                 "[[ ## question ## ]]\n"
                 "{question}\n"
                 "\n"
                 "[[ ## answer ## ]]\n"
                 "{answer}\n"
                 "\n"
                 "[[ ## completed ## ]]\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Given the fields `question`, produce the fields `answer`, `citations`."},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Q?\n"
                 "\n"
                 "Respond with the corresponding output fields, starting with the field `[[ ## answer "
                 "## ]]`, and then ending with the marker for `[[ ## completed ## ]]`."}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs

def test_chat_adapter_format_exact_messages_preserves_passthrough_lm_kwargs():
    class PassthroughSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(
        dspy.ChatAdapter(),
        PassthroughSignature,
        [],
        {"question": "Q?"},
        lm_kwargs={"temperature": 0.7, "max_tokens": 42, "stream": True, "cache": False},
    )

    expected_messages = [{"role": "system",
      "content": "Your input fields are:\n"
                 "1. `question` (str):\n"
                 "Your output fields are:\n"
                 "1. `answer` (str):\n"
                 "All interactions will be structured in the following way, with the appropriate "
                 "values filled in.\n"
                 "\n"
                 "[[ ## question ## ]]\n"
                 "{question}\n"
                 "\n"
                 "[[ ## answer ## ]]\n"
                 "{answer}\n"
                 "\n"
                 "[[ ## completed ## ]]\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Given the fields `question`, produce the fields `answer`."},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Q?\n"
                 "\n"
                 "Respond with the corresponding output fields, starting with the field `[[ ## answer "
                 "## ]]`, and then ending with the marker for `[[ ## completed ## ]]`."}]
    assert messages == expected_messages
    expected_lm_kwargs = {"temperature": 0.7, "max_tokens": 42, "stream": True, "cache": False}
    assert lm_kwargs == expected_lm_kwargs

def test_chat_adapter_format_exact_messages_and_lm_kwargs_with_native_reasoning():
    class ReasoningLM(dspy.utils.DummyLM):
        @property
        def supports_reasoning(self):
            return True

    class NativeReasoningSignature(dspy.Signature):
        question: str = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        answer: str = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(
        dspy.ChatAdapter(),
        NativeReasoningSignature,
        [],
        {"question": "Q?"},
        lm=ReasoningLM([{}]),
    )

    expected_messages = [{"role": "system",
      "content": "Your input fields are:\n"
                 "1. `question` (str):\n"
                 "Your output fields are:\n"
                 "1. `answer` (str):\n"
                 "All interactions will be structured in the following way, with the appropriate "
                 "values filled in.\n"
                 "\n"
                 "[[ ## question ## ]]\n"
                 "{question}\n"
                 "\n"
                 "[[ ## answer ## ]]\n"
                 "{answer}\n"
                 "\n"
                 "[[ ## completed ## ]]\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Given the fields `question`, produce the fields `reasoning`, `answer`."},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Q?\n"
                 "\n"
                 "Respond with the corresponding output fields, starting with the field `[[ ## answer "
                 "## ]]`, and then ending with the marker for `[[ ## completed ## ]]`."}]
    assert messages == expected_messages
    expected_lm_kwargs = {"reasoning_effort": "low"}
    assert lm_kwargs == expected_lm_kwargs

def test_chat_adapter_format_exact_messages_with_reasoning_and_code_outputs():
    python_code = dspy.Code["python"]

    class CodeSignature(dspy.Signature):
        question: str = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        code: python_code = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(dspy.ChatAdapter(),
        CodeSignature,
        [{"question": "Q1", "reasoning": dspy.Reasoning(content="Think"), "code": python_code(code="print('hi')")}],
        {"question": "Q2"},
    )

    expected_messages = [{"role": "system",
      "content": "Your input fields are:\n"
                 "1. `question` (str):\n"
                 "Your output fields are:\n"
                 "1. `reasoning` (str): \n"
                 "2. `code` (Code_python): \n"
                 "    Type description of Code_python: Code represented in a string, specified in the "
                 "`code` field. If this is an output field, the code field should follow the markdown "
                 "code block format, e.g. \n"
                 "```python\n"
                 "{code}\n"
                 "```\n"
                 "Programming language: python\n"
                 "All interactions will be structured in the following way, with the appropriate "
                 "values filled in.\n"
                 "\n"
                 "[[ ## question ## ]]\n"
                 "{question}\n"
                 "\n"
                 "[[ ## reasoning ## ]]\n"
                 "{reasoning}\n"
                 "\n"
                 "[[ ## code ## ]]\n"
                 "{code}\n"
                 "\n"
                 "[[ ## completed ## ]]\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Given the fields `question`, produce the fields `reasoning`, `code`."},
     {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
     {"role": "assistant",
      "content": "[[ ## reasoning ## ]]\n"
                 "Think\n"
                 "\n"
                 "[[ ## code ## ]]\n"
                 "print('hi')\n"
                 "\n"
                 "[[ ## completed ## ]]\n"},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Q2\n"
                 "\n"
                 "Respond with the corresponding output fields, starting with the field `[[ ## "
                 "reasoning ## ]]` (must be formatted as a valid Python str), then `[[ ## code ## ]]` "
                 "(must be formatted as a valid Python Code_python), and then ending with the marker "
                 "for `[[ ## completed ## ]]`."}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_and_lm_kwargs_with_native_tool_calling():
    class FunctionCallingLM(dspy.utils.DummyLM):
        @property
        def supports_function_calling(self):
            return True

    def search(query: str, k: int = 3) -> str:
        """Search for documents."""
        return query

    class NativeToolSignature(dspy.Signature):
        question: str = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        tool_calls: dspy.ToolCalls = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(
        dspy.ChatAdapter(use_native_function_calling=True),
        NativeToolSignature,
        [],
        {"question": "Q?", "tools": [dspy.Tool(search)]},
        lm=FunctionCallingLM([{}]),
    )

    expected_messages = [{"role": "system",
      "content": "Your input fields are:\n"
                 "1. `question` (str):\n"
                 "Your output fields are:\n"
                 "\n"
                 "All interactions will be structured in the following way, with the appropriate "
                 "values filled in.\n"
                 "\n"
                 "[[ ## question ## ]]\n"
                 "{question}\n"
                 "\n"
                 "\n"
                 "\n"
                 "[[ ## completed ## ]]\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Given the fields `question`, `tools`, produce the fields `tool_calls`."},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Q?\n"
                 "\n"
                 "Respond with the corresponding output fields, starting with the field , and then "
                 "ending with the marker for `[[ ## completed ## ]]`."}]
    assert messages == expected_messages
    expected_lm_kwargs = {"tools": [{"type": "function",
                "function": {"name": "search",
                             "description": "Search for documents.",
                             "parameters": {"type": "object",
                                            "properties": {"query": {"type": "string"},
                                                           "k": {"type": "integer", "default": 3}},
                                            "required": ["query", "k"]}}}]}
    assert lm_kwargs == expected_lm_kwargs

def test_chat_adapter_format_exact_messages_with_tool_input():
    def search(query: str, k: int = 3) -> str:
        """Search for documents."""
        return query

    class ToolSignature(dspy.Signature):
        question: str = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        answer: str = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(dspy.ChatAdapter(),
        ToolSignature,
        [],
        {"question": "Q?", "tools": [dspy.Tool(search)]},
    )

    expected_messages = [{"role": "system",
      "content": "Your input fields are:\n"
                 "1. `question` (str): \n"
                 "2. `tools` (list[Tool]):\n"
                 "Your output fields are:\n"
                 "1. `answer` (str):\n"
                 "All interactions will be structured in the following way, with the appropriate "
                 "values filled in.\n"
                 "\n"
                 "[[ ## question ## ]]\n"
                 "{question}\n"
                 "\n"
                 "[[ ## tools ## ]]\n"
                 "{tools}\n"
                 "\n"
                 "[[ ## answer ## ]]\n"
                 "{answer}\n"
                 "\n"
                 "[[ ## completed ## ]]\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Given the fields `question`, `tools`, produce the fields `answer`."},
     {"role": "user",
      "content": '[[ ## question ## ]]\n'
                 'Q?\n'
                 '\n'
                 '[[ ## tools ## ]]\n'
                 '["search, whose description is <desc>Search for documents.</desc>. It takes '
                 "arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', 'default': "
                 '3}}."]\n'
                 '\n'
                 'Respond with the corresponding output fields, starting with the field `[[ ## answer '
                 '## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.'}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_kitchen_sink():
    def search(query: str, k: int = 3) -> str:
        """Search for documents."""
        return query

    class Event(dspy.Type):
        label: str

        def format(self):
            return [{"type": "event", "event": {"label": self.label}}]

        @classmethod
        def description(cls):
            return "An event block."

    class Location(pydantic.BaseModel):
        city: str
        country: str

    class Profile(pydantic.BaseModel):
        name: str
        location: Location
        interests: list[str]

    class AnswerCard(pydantic.BaseModel):
        answer: str
        sources: list[str]

    class KitchenSinkSignature(dspy.Signature):
        """Answer carefully using every available signal."""

        history: dspy.History = dspy.InputField()
        image: dspy.Image = dspy.InputField()
        audio: dspy.Audio = dspy.InputField()
        file: dspy.File = dspy.InputField()
        document: Document = dspy.InputField()
        event: Event = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        profile: Profile = dspy.InputField()
        context: str = dspy.InputField()
        question: str = dspy.InputField()
        answer: AnswerCard = dspy.OutputField()
        verdict: Literal["yes", "no"] = dspy.OutputField()
        confidence: float = dspy.OutputField()

    tool = dspy.Tool(search)
    demo_profile = Profile(
        name="Ada",
        location=Location(city="London", country="UK"),
        interests=["math", "machines"],
    )
    current_profile = Profile(
        name="Grace",
        location=Location(city="Arlington", country="USA"),
        interests=["compilers", "navy"],
    )
    history = dspy.History(
        messages=[
            {
                "profile": demo_profile,
                "context": ["old note", "older note"],
                "question": "Who is Ada?",
                "answer": AnswerCard(answer="Ada is a mathematician.", sources=["memory"]),
                "verdict": "yes",
                "confidence": 0.8,
            }
        ]
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(dspy.ChatAdapter(),
        KitchenSinkSignature,
        demos=[
            {
                "image": dspy.Image("https://example.com/demo.png"),
                "audio": dspy.Audio(data="REVNTw==", audio_format="wav"),
                "file": dspy.File.from_file_id("file-demo", filename="demo.txt"),
                "document": Document(data="Demo document", title="Demo Doc"),
                "event": Event(label="demo-event"),
                "tools": [tool],
                "profile": demo_profile,
                "context": ["demo context one", "demo context two"],
                "question": "What should we mention?",
                "answer": AnswerCard(answer="Mention analytical engines.", sources=["demo"]),
                "verdict": "yes",
                "confidence": 0.9,
            },
            {
                "question": "Incomplete example question",
                "answer": AnswerCard(answer="Partial answer.", sources=["partial"]),
            },
        ],
        inputs={
            "history": history,
            "image": dspy.Image("https://example.com/current.png"),
            "audio": dspy.Audio(data="Q1VSUkVOVA==", audio_format="wav"),
            "file": dspy.File.from_file_id("file-current", filename="current.txt"),
            "document": Document(data="Current document", title="Current Doc"),
            "event": Event(label="current-event"),
            "tools": [tool],
            "profile": current_profile,
            "context": ["current context one", "current context two"],
            "question": "What should the answer include?",
        },
    )

    expected_messages = [{"role": "system",
      "content": 'Your input fields are:\n'
                 '1. `history` (History): \n'
                 '2. `image` (Image): \n'
                 '3. `audio` (Audio): \n'
                 '4. `file` (File): \n'
                 '5. `document` (Document): \n'
                 '    Type description of Document: A document containing text content that can be '
                 'referenced and cited. Include the full text content and optionally a title for '
                 'proper referencing.\n'
                 '6. `event` (Event): \n'
                 '    Type description of Event: An event block.\n'
                 '7. `tools` (list[Tool]): \n'
                 '8. `profile` (Profile): \n'
                 '9. `context` (str): \n'
                 '10. `question` (str):\n'
                 'Your output fields are:\n'
                 '1. `answer` (AnswerCard): \n'
                 "2. `verdict` (Literal['yes', 'no']): \n"
                 '3. `confidence` (float):\n'
                 'All interactions will be structured in the following way, with the appropriate '
                 'values filled in.\n'
                 '\n'
                 '[[ ## history ## ]]\n'
                 '{history}\n'
                 '\n'
                 '[[ ## image ## ]]\n'
                 '{image}\n'
                 '\n'
                 '[[ ## audio ## ]]\n'
                 '{audio}\n'
                 '\n'
                 '[[ ## file ## ]]\n'
                 '{file}\n'
                 '\n'
                 '[[ ## document ## ]]\n'
                 '{document}\n'
                 '\n'
                 '[[ ## event ## ]]\n'
                 '{event}\n'
                 '\n'
                 '[[ ## tools ## ]]\n'
                 '{tools}\n'
                 '\n'
                 '[[ ## profile ## ]]\n'
                 '{profile}\n'
                 '\n'
                 '[[ ## context ## ]]\n'
                 '{context}\n'
                 '\n'
                 '[[ ## question ## ]]\n'
                 '{question}\n'
                 '\n'
                 '[[ ## answer ## ]]\n'
                 '{answer}        # note: the value you produce must adhere to the JSON schema: '
                 '{"type": "object", "properties": {"answer": {"type": "string", "title": "Answer"}, '
                 '"sources": {"type": "array", "items": {"type": "string"}, "title": "Sources"}}, '
                 '"required": ["answer", "sources"], "title": "AnswerCard"}\n'
                 '\n'
                 '[[ ## verdict ## ]]\n'
                 '{verdict}        # note: the value you produce must exactly match (no extra '
                 'characters) one of: yes; no\n'
                 '\n'
                 '[[ ## confidence ## ]]\n'
                 '{confidence}        # note: the value you produce must be a single float value\n'
                 '\n'
                 '[[ ## completed ## ]]\n'
                 'In adhering to this structure, your objective is: \n'
                 '        Answer carefully using every available signal.'},
     {"role": "user",
      "content": [{"type": "text",
                   "text": "This is an example of the task, though some input or output fields are not "
                           "supplied.\n"
                           "\n"
                           "[[ ## image ## ]]\n"},
                  {"type": "image_url", "image_url": {"url": "https://example.com/demo.png"}},
                  {"type": "text", "text": "\n\n[[ ## audio ## ]]\n"},
                  {"type": "input_audio", "input_audio": {"data": "REVNTw==", "format": "wav"}},
                  {"type": "text", "text": "\n\n[[ ## file ## ]]\n"},
                  {"type": "file", "file": {"file_id": "file-demo", "filename": "demo.txt"}},
                  {"type": "text", "text": "\n\n[[ ## document ## ]]\n"},
                  {"type": "document",
                   "source": {"type": "text", "media_type": "text/plain", "data": "Demo document"},
                   "citations": {"enabled": True},
                   "title": "Demo Doc"},
                  {"type": "text", "text": "\n\n[[ ## event ## ]]\n"},
                  {"type": "event", "event": {"label": "demo-event"}},
                  {"type": "text",
                   "text": '\n'
                           '\n'
                           '[[ ## tools ## ]]\n'
                           '["search, whose description is <desc>Search for documents.</desc>. It '
                           "takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', "
                           '\'default\': 3}}."]\n'
                           '\n'
                           '[[ ## profile ## ]]\n'
                           '{"name": "Ada", "location": {"city": "London", "country": "UK"}, '
                           '"interests": ["math", "machines"]}\n'
                           '\n'
                           '[[ ## context ## ]]\n'
                           '[1] «demo context one»\n'
                           '[2] «demo context two»\n'
                           '\n'
                           '[[ ## question ## ]]\n'
                           'What should we mention?'}]},
     {"role": "assistant",
      "content": '[[ ## answer ## ]]\n'
                 '{"answer": "Mention analytical engines.", "sources": ["demo"]}\n'
                 '\n'
                 '[[ ## verdict ## ]]\n'
                 'yes\n'
                 '\n'
                 '[[ ## confidence ## ]]\n'
                 '0.9\n'
                 '\n'
                 '[[ ## completed ## ]]\n'},
     {"role": "user",
      "content": "This is an example of the task, though some input or output fields are not "
                 "supplied.\n"
                 "\n"
                 "[[ ## question ## ]]\n"
                 "Incomplete example question"},
     {"role": "assistant",
      "content": '[[ ## answer ## ]]\n'
                 '{"answer": "Partial answer.", "sources": ["partial"]}\n'
                 '\n'
                 '[[ ## verdict ## ]]\n'
                 'Not supplied for this particular example. \n'
                 '\n'
                 '[[ ## confidence ## ]]\n'
                 'Not supplied for this particular example.\n'
                 '\n'
                 '[[ ## completed ## ]]\n'},
     {"role": "user",
      "content": '[[ ## profile ## ]]\n'
                 '{"name": "Ada", "location": {"city": "London", "country": "UK"}, "interests": '
                 '["math", "machines"]}\n'
                 '\n'
                 '[[ ## context ## ]]\n'
                 '[1] «old note»\n'
                 '[2] «older note»\n'
                 '\n'
                 '[[ ## question ## ]]\n'
                 'Who is Ada?'},
     {"role": "assistant",
      "content": '[[ ## answer ## ]]\n'
                 '{"answer": "Ada is a mathematician.", "sources": ["memory"]}\n'
                 '\n'
                 '[[ ## verdict ## ]]\n'
                 'yes\n'
                 '\n'
                 '[[ ## confidence ## ]]\n'
                 '0.8\n'
                 '\n'
                 '[[ ## completed ## ]]\n'},
     {"role": "user",
      "content": [{"type": "text", "text": "[[ ## image ## ]]\n"},
                  {"type": "image_url", "image_url": {"url": "https://example.com/current.png"}},
                  {"type": "text", "text": "\n\n[[ ## audio ## ]]\n"},
                  {"type": "input_audio", "input_audio": {"data": "Q1VSUkVOVA==", "format": "wav"}},
                  {"type": "text", "text": "\n\n[[ ## file ## ]]\n"},
                  {"type": "file", "file": {"file_id": "file-current", "filename": "current.txt"}},
                  {"type": "text", "text": "\n\n[[ ## document ## ]]\n"},
                  {"type": "document",
                   "source": {"type": "text", "media_type": "text/plain", "data": "Current document"},
                   "citations": {"enabled": True},
                   "title": "Current Doc"},
                  {"type": "text", "text": "\n\n[[ ## event ## ]]\n"},
                  {"type": "event", "event": {"label": "current-event"}},
                  {"type": "text",
                   "text": '\n'
                           '\n'
                           '[[ ## tools ## ]]\n'
                           '["search, whose description is <desc>Search for documents.</desc>. It '
                           "takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', "
                           '\'default\': 3}}."]\n'
                           '\n'
                           '[[ ## profile ## ]]\n'
                           '{"name": "Grace", "location": {"city": "Arlington", "country": "USA"}, '
                           '"interests": ["compilers", "navy"]}\n'
                           '\n'
                           '[[ ## context ## ]]\n'
                           '[1] «current context one»\n'
                           '[2] «current context two»\n'
                           '\n'
                           '[[ ## question ## ]]\n'
                           'What should the answer include?\n'
                           '\n'
                           'Respond with the corresponding output fields, starting with the field `[[ '
                           '## answer ## ]]` (must be formatted as a valid Python AnswerCard), then '
                           "`[[ ## verdict ## ]]` (must be formatted as a valid Python Literal['yes', "
                           "'no']), then `[[ ## confidence ## ]]` (must be formatted as a valid Python "
                           'float), and then ending with the marker for `[[ ## completed ## ]]`.'}]}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs

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

    dspy.configure(lm=dspy.LM(model="openai/gpt-4o"), adapter=dspy.ChatAdapter())
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

    dspy.configure(lm=dspy.LM(model="openai/gpt-4o"), adapter=dspy.ChatAdapter())
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
    with pytest.raises(dspy.utils.exceptions.AdapterParseError, match=r"Adapter ChatAdapter failed to parse.*"):
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

    assert "[[ ## completed ## ]]\n" in messages[2]["content"]
    assert "[[ ## completed ## ]]\n" in messages[4]["content"]

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


def test_chat_adapter_with_code():
    # Test with code as input field
    class CodeAnalysis(dspy.Signature):
        """Analyze the time complexity of the code"""

        code: dspy.Code = dspy.InputField()
        result: str = dspy.OutputField()

    adapter = dspy.ChatAdapter()
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

    adapter = dspy.ChatAdapter()
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content='[[ ## code ## ]]\nprint("Hello, world!")'))],
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


def test_code_output_field_omits_json_schema_in_prompt():
    """Regression test for #9251: dspy.Code should avoid duplicating large JSON schema text."""
    class CodeGeneration(dspy.Signature):
        """Generate code to answer the question"""
        question: str = dspy.InputField()
        code: dspy.Code = dspy.OutputField()

    adapter = dspy.ChatAdapter()
    messages = adapter.format(CodeGeneration, [], {"question": "Hello"})
    system_content = messages[0]["content"]

    assert dspy.Code.description() in system_content
    assert "JSON schema" not in system_content
    assert '"properties"' not in system_content
    assert "Code type in DSPy" not in system_content


def test_citations_output_field_keeps_json_schema_in_prompt():
    """Non-Code custom types should keep schema guidance for structured output reliability."""

    class CitationGeneration(dspy.Signature):
        question: str = dspy.InputField()
        citations: Citations = dspy.OutputField()

    adapter = dspy.ChatAdapter()
    messages = adapter.format(CitationGeneration, [], {"question": "Hello"})
    system_content = messages[0]["content"]

    assert "must adhere to the JSON schema" in system_content
    assert "Type description of Citations" in system_content


def test_chat_adapter_formats_conversation_history():
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

    adapter = dspy.ChatAdapter()
    messages = adapter.format(MySignature, [], {"question": "What is the capital of France?", "history": history})

    assert len(messages) == 6
    assert messages[1]["content"] == "[[ ## question ## ]]\nWhat is the capital of France?"
    assert messages[2]["content"] == "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]\n"
    assert messages[3]["content"] == "[[ ## question ## ]]\nWhat is the capital of Germany?"
    assert messages[4]["content"] == "[[ ## answer ## ]]\nBerlin\n\n[[ ## completed ## ]]\n"


def test_chat_adapter_fallback_to_json_adapter_on_exception():
    signature = dspy.make_signature("question->answer")
    adapter = dspy.ChatAdapter()

    with mock.patch("litellm.completion") as mock_completion:
        # Mock returning a response compatible with JSONAdapter but not ChatAdapter
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'answer': 'Paris'}"))],
            model="openai/gpt-4o-mini",
        )

        lm = dspy.LM("openai/gpt-4o-mini", cache=False)

        with mock.patch("dspy.adapters.json_adapter.JSONAdapter.__call__") as mock_json_adapter_call:
            adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})
            mock_json_adapter_call.assert_called_once()

        # The parse should succeed
        result = adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})
        assert result == [{"answer": "Paris"}]


def test_chat_adapter_respects_use_json_adapter_fallback_flag():
    signature = dspy.make_signature("question->answer")
    adapter = dspy.ChatAdapter(use_json_adapter_fallback=False)

    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="nonsense"))],
            model="openai/gpt-4o-mini",
        )

        lm = dspy.LM("openai/gpt-4o-mini", cache=False)

        with mock.patch("dspy.adapters.json_adapter.JSONAdapter.__call__") as mock_json_adapter_call:
            with pytest.raises(dspy.utils.exceptions.AdapterParseError):
                adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})
        mock_json_adapter_call.assert_not_called()


@pytest.mark.asyncio
async def test_chat_adapter_fallback_to_json_adapter_on_exception_async():
    signature = dspy.make_signature("question->answer")
    adapter = dspy.ChatAdapter()

    with mock.patch("litellm.acompletion") as mock_completion:
        # Mock returning a response compatible with JSONAdapter but not ChatAdapter
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'answer': 'Paris'}"))],
            model="openai/gpt-4o-mini",
        )

        lm = dspy.LM("openai/gpt-4o-mini", cache=False)

        with mock.patch("dspy.adapters.json_adapter.JSONAdapter.acall") as mock_json_adapter_acall:
            await adapter.acall(lm, {}, signature, [], {"question": "What is the capital of France?"})
            mock_json_adapter_acall.assert_called_once()

        # The parse should succeed
        result = await adapter.acall(lm, {}, signature, [], {"question": "What is the capital of France?"})
        assert result == [{"answer": "Paris"}]


def test_chat_adapter_toolcalls_native_function_calling():
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


def test_chat_adapter_toolcalls_vague_match():
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        tool_calls: dspy.ToolCalls = dspy.OutputField()

    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny"

    tools = [dspy.Tool(get_weather)]

    adapter = dspy.ChatAdapter()

    with mock.patch("litellm.completion") as mock_completion:
        # Case 1: tool_calls field is a list of dicts
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    message=Message(
                        content="[[ ## tool_calls ## ]]\n[{'name': 'get_weather', 'args': {'city': 'Paris'}]"
                    )
                )
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

    with mock.patch("litellm.completion") as mock_completion:
        # Case 2: tool_calls field is a single dict with "name" and "args" keys
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    message=Message(
                        content="[[ ## tool_calls ## ]]\n{'name': 'get_weather', 'args': {'city': 'Paris'}}"
                    )
                )
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


def test_chat_adapter_native_reasoning():
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        answer: str = dspy.OutputField()

    adapter = dspy.ChatAdapter()

    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    message=Message(
                        content="[[ ## answer ## ]]\nParis\n[[ ## completion ## ]]",
                        reasoning_content="Step-by-step thinking about the capital of France",
                    ),
                )
            ],
            model="anthropic/claude-3-7-sonnet-20250219",
        )
        modified_signature = adapter._call_preprocess(
            dspy.LM(model="anthropic/claude-3-7-sonnet-20250219", reasoning_effort="low", cache=False),
            {},
            MySignature,
            {"question": "What is the capital of France?"},
        )
        assert "reasoning" not in modified_signature.output_fields

        result = adapter(
            dspy.LM(model="anthropic/claude-3-7-sonnet-20250219", reasoning_effort="low", cache=False),
            {},
            MySignature,
            [],
            {"question": "What is the capital of France?"},
        )
        assert result[0]["reasoning"] == dspy.Reasoning(content="Step-by-step thinking about the capital of France")


def test_chat_adapter_parses_float_with_underscores():
    """
    This test verifies that ChatAdapter can parse float numbers with underscores.
    After json-repair version 0.54.1, floats like "123_456.789" are treated as normal float numbers.
    """

    class Score(pydantic.BaseModel):
        score: float

    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        score: Score = dspy.OutputField()

    adapter = dspy.ChatAdapter()

    # Simulate a response with a float number containing underscores
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(message=Message(content="[[ ## score ## ]]\n{'score': 123_456.789}\n[[ ## completed ## ]]"))
            ],
            model="openai/gpt-4o-mini",
        )

        lm = dspy.LM("openai/gpt-4o-mini", cache=False)
        result = adapter(lm, {}, MySignature, [], {"question": "What is the score?"})

        # The underscore-separated float should be parsed as a normal float
        assert result[0]["score"].score == 123456.789


def test_format_system_message():
    class MySignature(dspy.Signature):
        """Answer the question with multiple answers and scores"""

        question: str = dspy.InputField()
        answers: list[str] = dspy.OutputField()
        scores: list[float] = dspy.OutputField()

    adapter = dspy.ChatAdapter()
    system_message = adapter.format_system_message(MySignature)
    expected_system_message = """Your input fields are:
1. `question` (str):
Your output fields are:
1. `answers` (list[str]):\x20
2. `scores` (list[float]):
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}

[[ ## answers ## ]]
{answers}        # note: the value you produce must adhere to the JSON schema: {"type": "array", "items": {"type": "string"}}

[[ ## scores ## ]]
{scores}        # note: the value you produce must adhere to the JSON schema: {"type": "array", "items": {"type": "number"}}

[[ ## completed ## ]]
In adhering to this structure, your objective is:\x20
        Answer the question with multiple answers and scores"""
    assert system_message == expected_system_message


def test_null_content_raises_adapter_parse_error():
    """When the LM returns content=None with no tool calls (e.g. content filter),
    the adapter should raise AdapterParseError instead of silently returning None fields."""
    from dspy.utils.exceptions import AdapterParseError

    lm = dspy.LM("openai/gpt-4o-mini", cache=False)
    response = ModelResponse(
        choices=[Choices(message=Message(content=None))],
        model="openai/gpt-4o-mini",
    )

    with dspy.context(lm=lm):
        with mock.patch("litellm.completion", return_value=response):
            cot = dspy.ChainOfThought("question -> answer")
            with pytest.raises(AdapterParseError):
                cot(question="test")


def test_empty_string_content_raises_adapter_parse_error():
    """Same as above but with empty string content."""
    from dspy.utils.exceptions import AdapterParseError

    lm = dspy.LM("openai/gpt-4o-mini", cache=False)
    response = ModelResponse(
        choices=[Choices(message=Message(content=""))],
        model="openai/gpt-4o-mini",
    )

    with dspy.context(lm=lm):
        with mock.patch("litellm.completion", return_value=response):
            cot = dspy.ChainOfThought("question -> answer")
            with pytest.raises(AdapterParseError):
                cot(question="test")


def test_tool_call_with_null_content_does_not_raise():
    """Tool-call-only responses legitimately have content=None.
    _call_postprocess must NOT raise when tool_calls are present."""
    adapter = dspy.ChatAdapter(use_native_function_calling=True)
    sig_cls = dspy.Signature("question, tools: list[dspy.Tool] -> answer, tool_calls: dspy.ToolCalls")

    outputs = [{"text": None, "tool_calls": [
        {"function": {"name": "search", "arguments": '{"query": "test"}'}, "id": "call_1", "type": "function"}
    ]}]

    result = adapter._call_postprocess(sig_cls, sig_cls, outputs, None, {})
    assert result is not None
    assert len(result) == 1
