import sys
import time
from unittest import mock

import pydantic
import pytest
from litellm import Choices, Message, ModelResponse

import dspy
from dspy.adapters.chat_adapter import FieldInfoWithName
from dspy.adapters.xml_adapter import XMLAdapter
from dspy.utils.exceptions import AdapterParseError
from tests.adapters.conftest import format_messages_and_lm_kwargs


def test_xml_adapter_format_and_parse_basic():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    adapter = XMLAdapter()
    # Format output fields as XML
    fields_with_values = {FieldInfoWithName(name="answer", info=TestSignature.output_fields["answer"]): "Paris"}
    xml = adapter.format_field_with_value(fields_with_values)
    assert xml.strip() == "<answer>\nParis\n</answer>"

    # Parse XML output
    completion = "<answer>Paris</answer>"
    parsed = adapter.parse(TestSignature, completion)
    assert parsed == {"answer": "Paris"}


def test_xml_adapter_parse_multiple_fields():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        explanation: str = dspy.OutputField()

    adapter = XMLAdapter()
    completion = """
<answer>Paris</answer>
<explanation>The capital of France is Paris.</explanation>
"""
    parsed = adapter.parse(TestSignature, completion)
    assert parsed == {"answer": "Paris", "explanation": "The capital of France is Paris."}


def test_xml_adapter_parse_raises_on_missing_field():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        explanation: str = dspy.OutputField()

    adapter = XMLAdapter()
    completion = "<answer>Paris</answer>"
    with pytest.raises(dspy.utils.exceptions.AdapterParseError) as e:
        adapter.parse(TestSignature, completion)
    assert e.value.adapter_name == "XMLAdapter"
    assert e.value.signature == TestSignature
    assert e.value.lm_response == "<answer>Paris</answer>"
    assert "explanation" in str(e.value)


def test_xml_adapter_parse_casts_types():
    class TestSignature(dspy.Signature):
        number: int = dspy.OutputField()
        flag: bool = dspy.OutputField()

    adapter = XMLAdapter()
    completion = """
<number>42</number>
<flag>true</flag>
"""
    parsed = adapter.parse(TestSignature, completion)
    assert parsed == {"number": 42, "flag": True}


def test_xml_adapter_parse_raises_on_type_error():
    class TestSignature(dspy.Signature):
        number: int = dspy.OutputField()

    adapter = XMLAdapter()
    completion = "<number>not_a_number</number>"
    with pytest.raises(dspy.utils.exceptions.AdapterParseError) as e:
        adapter.parse(TestSignature, completion)
    assert "Failed to parse field" in str(e.value)


def test_xml_adapter_format_and_parse_nested_model():
    class InnerModel(pydantic.BaseModel):
        value: int
        label: str

    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        result: InnerModel = dspy.OutputField()

    adapter = XMLAdapter()
    # Format output fields as XML
    fields_with_values = {
        FieldInfoWithName(name="result", info=TestSignature.output_fields["result"]): InnerModel(value=5, label="foo")
    }
    xml = adapter.format_field_with_value(fields_with_values)
    # The output will be a JSON string inside the XML tag
    assert xml.strip().startswith("<result>")
    assert '"value": 5' in xml
    assert '"label": "foo"' in xml
    assert xml.strip().endswith("</result>")

    # Parse XML output (should parse as string, not as model)
    completion = '<result>{"value": 5, "label": "foo"}</result>'
    parsed = adapter.parse(TestSignature, completion)
    # The parse_value helper will try to cast to InnerModel
    assert isinstance(parsed["result"], InnerModel)
    assert parsed["result"].value == 5
    assert parsed["result"].label == "foo"


def test_xml_adapter_format_and_parse_list_of_models():
    class Item(pydantic.BaseModel):
        name: str
        score: float

    class TestSignature(dspy.Signature):
        items: list[Item] = dspy.OutputField()

    adapter = XMLAdapter()
    items = [Item(name="a", score=1.1), Item(name="b", score=2.2)]
    fields_with_values = {FieldInfoWithName(name="items", info=TestSignature.output_fields["items"]): items}
    xml = adapter.format_field_with_value(fields_with_values)
    assert xml.strip().startswith("<items>")
    assert '"name": "a"' in xml
    assert '"score": 2.2' in xml
    assert xml.strip().endswith("</items>")

    # Parse XML output
    import json

    completion = f"<items>{json.dumps([i.model_dump() for i in items])}</items>"
    parsed = adapter.parse(TestSignature, completion)
    assert isinstance(parsed["items"], list)
    assert all(isinstance(i, Item) for i in parsed["items"])
    assert parsed["items"][0].name == "a"
    assert parsed["items"][1].score == 2.2


def test_xml_adapter_with_tool_like_output():
    # XMLAdapter does not natively support tool calls, but we can test structured output
    class ToolCall(pydantic.BaseModel):
        name: str
        args: dict
        result: str

    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        tool_calls: list[ToolCall] = dspy.OutputField()
        answer: str = dspy.OutputField()

    adapter = XMLAdapter()
    tool_calls = [
        ToolCall(name="get_weather", args={"city": "Tokyo"}, result="Sunny"),
        ToolCall(name="get_population", args={"country": "Japan", "year": 2023}, result="125M"),
    ]
    fields_with_values = {
        FieldInfoWithName(name="tool_calls", info=TestSignature.output_fields["tool_calls"]): tool_calls,
        FieldInfoWithName(
            name="answer", info=TestSignature.output_fields["answer"]
        ): "The weather is Sunny. Population is 125M.",
    }
    xml = adapter.format_field_with_value(fields_with_values)
    assert xml.strip().startswith("<tool_calls>")
    assert '"name": "get_weather"' in xml
    assert '"result": "125M"' in xml
    assert xml.strip().endswith("</answer>")

    import json

    completion = (
        f"<tool_calls>{json.dumps([tc.model_dump() for tc in tool_calls])}</tool_calls>"
        f"\n<answer>The weather is Sunny. Population is 125M.</answer>"
    )
    parsed = adapter.parse(TestSignature, completion)
    assert isinstance(parsed["tool_calls"], list)
    assert parsed["tool_calls"][0].name == "get_weather"
    assert parsed["tool_calls"][1].result == "125M"
    assert parsed["answer"] == "The weather is Sunny. Population is 125M."


def test_xml_adapter_formats_nested_images():
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
    adapter = dspy.XMLAdapter()
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


def test_xml_adapter_with_code():
    # Test with code as input field
    class CodeAnalysis(dspy.Signature):
        """Analyze the time complexity of the code"""

        code: dspy.Code = dspy.InputField()
        result: str = dspy.OutputField()

    adapter = dspy.XMLAdapter()
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

    adapter = dspy.XMLAdapter()
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content='<code>print("Hello, world!")</code>'))],
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


def test_xml_adapter_full_prompt():
    class QA(dspy.Signature):
        query: str = dspy.InputField()
        context: str | None = dspy.InputField()
        answer: str = dspy.OutputField()

    adapter = dspy.XMLAdapter()
    messages = adapter.format(QA, [], {"query": "when was Marie Curie born"})

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    union_type_repr = "Union[str, NoneType]" if sys.version_info >= (3, 14) else "UnionType[str, NoneType]"

    expected_system = (
        "Your input fields are:\n"
        "1. `query` (str): \n"
        f"2. `context` ({union_type_repr}):\n"
        "Your output fields are:\n"
        "1. `answer` (str):\n"
        "All interactions will be structured in the following way, with the appropriate values filled in.\n\n"
        "<query>\n{query}\n</query>\n\n"
        "<context>\n{context}\n</context>\n\n"
        "<answer>\n{answer}\n</answer>\n"
        "In adhering to this structure, your objective is: \n"
        "        Given the fields `query`, `context`, produce the fields `answer`."
    )

    expected_user = (
        "<query>\nwhen was Marie Curie born\n</query>\n\n"
        "Respond with the corresponding output fields wrapped in XML tags `<answer>`."
    )

    assert messages[0]["content"] == expected_system
    assert messages[1]["content"] == expected_user


def test_xml_adapter_format_exact_messages_for_simple_signature():
    class StringSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(dspy.XMLAdapter(),
        StringSignature,
        demos=[],
        inputs={"question": "why did a chicken cross the kitchen?"},
    )

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

<question>
{question}
</question>

<answer>
{answer}
</answer>
In adhering to this structure, your objective is:\x20
        Given the fields `question`, produce the fields `answer`.""",
        },
        {
            "role": "user",
            "content": """<question>
why did a chicken cross the kitchen?
</question>

Respond with the corresponding output fields wrapped in XML tags `<answer>`.""",
        },
    ]


def test_xml_adapter_format_exact_non_native_tool_result_history_field():
    def search(query: str) -> str:
        return query

    class ToolHistorySignature(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        next_thought: str = dspy.OutputField()
        tool_calls: dspy.ToolCalls = dspy.OutputField()

    tool_call = dspy.ToolCalls.ToolCall(id="call_1", name="search", args={"query": "cats"})
    tool_call_results = dspy.ToolCallResults.from_tool_calls_and_values([tool_call], ["cat"])

    messages, _lm_kwargs = format_messages_and_lm_kwargs(
        dspy.XMLAdapter(use_native_function_calling=False),
        ToolHistorySignature,
        [],
        {
            "question": "Q2",
            "history": dspy.History(
                messages=[
                    {
                        "question": "Q1",
                        "next_thought": "I should search.",
                        "tool_calls": dspy.ToolCalls(tool_calls=[tool_call], tool_call_results=tool_call_results),
                    }
                ]
            ),
            "tools": [dspy.Tool(search)],
        },
    )

    assert messages[3]["content"] == (
        "<tool_call_results>\n"
        '{"tool_call_results": [{"call_id": "call_1", "name": "search", "value": "cat", "is_error": false}]}\n'
        "</tool_call_results>"
    )
    assert messages[4]["content"] == (
        "<question>\n"
        "Q2\n"
        "</question>\n"
        "\n"
        "<tools>\n"
        '["search. It takes arguments {\'query\': {\'type\': \'string\'}}."]\n'
        "</tools>\n"
        "\n"
        "Respond with the corresponding output fields wrapped in XML tags `<next_thought>`, then `<tool_calls>`."
    )


def test_xml_adapter_format_exact_messages_for_two_input_signature():
    class StringSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.InputField()
        judgement: str = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(dspy.XMLAdapter(),
        StringSignature,
        demos=[],
        inputs={"question": "why did a chicken cross the kitchen?", "answer": "To get to the other side!"},
    )

    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs

    assert messages == [
        {
            "role": "system",
            "content": """Your input fields are:
1. `question` (str):\x20
2. `answer` (str):
Your output fields are:
1. `judgement` (str):
All interactions will be structured in the following way, with the appropriate values filled in.

<question>
{question}
</question>

<answer>
{answer}
</answer>

<judgement>
{judgement}
</judgement>
In adhering to this structure, your objective is:\x20
        Given the fields `question`, `answer`, produce the fields `judgement`.""",
        },
        {
            "role": "user",
            "content": """<question>
why did a chicken cross the kitchen?
</question>

<answer>
To get to the other side!
</answer>

Respond with the corresponding output fields wrapped in XML tags `<judgement>`.""",
        },
    ]


def test_xml_adapter_format_exact_messages_with_demo_and_typed_output():
    class MultiAnswer(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        score: float = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(dspy.XMLAdapter(),
        MultiAnswer,
        demos=[{"question": "Q1", "answer": "A1", "score": 0.9}],
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
1. `answer` (str):\x20
2. `score` (float):
All interactions will be structured in the following way, with the appropriate values filled in.

<question>
{question}
</question>

<answer>
{answer}
</answer>

<score>
{score}        # note: the value you produce must be a single float value
</score>
In adhering to this structure, your objective is:\x20
        Given the fields `question`, produce the fields `answer`, `score`.""",
        },
        {"role": "user", "content": """<question>
Q1
</question>"""},
        {
            "role": "assistant",
            "content": """<answer>
A1
</answer>

<score>
0.9
</score>""",
        },
        {
            "role": "user",
            "content": """<question>
Q2
</question>

Respond with the corresponding output fields wrapped in XML tags `<answer>`, then `<score>`.""",
        },
    ]


def test_xml_adapter_format_exact_messages_with_history_demo_pydantic_tools_and_image():
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
    messages, lm_kwargs = format_messages_and_lm_kwargs(dspy.XMLAdapter(),
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
                 '<history>\n'
                 '{history}\n'
                 '</history>\n'
                 '\n'
                 '<image>\n'
                 '{image}\n'
                 '</image>\n'
                 '\n'
                 '<tools>\n'
                 '{tools}\n'
                 '</tools>\n'
                 '\n'
                 '<profile>\n'
                 '{profile}\n'
                 '</profile>\n'
                 '\n'
                 '<question>\n'
                 '{question}\n'
                 '</question>\n'
                 '\n'
                 '<answer>\n'
                 '{answer}        # note: the value you produce must adhere to the JSON schema: '
                 '{"type": "object", "properties": {"answer": {"type": "string", "title": "Answer"}, '
                 '"sources": {"type": "array", "items": {"type": "string"}, "title": "Sources"}}, '
                 '"required": ["answer", "sources"], "title": "AnswerCard"}\n'
                 '</answer>\n'
                 'In adhering to this structure, your objective is: \n'
                 '        Answer using all supplied context.'},
     {"role": "user",
      "content": [{"type": "text",
                   "text": "This is an example of the task, though some input or output fields are not "
                           "supplied.\n"
                           "\n"
                           "<image>\n"},
                  {"type": "image_url", "image_url": {"url": "https://example.com/demo.png"}},
                  {"type": "text",
                   "text": '\n'
                           '</image>\n'
                           '\n'
                           '<tools>\n'
                           '["search, whose description is <desc>Search for documents.</desc>. It '
                           "takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', "
                           '\'default\': 3}}."]\n'
                           '</tools>\n'
                           '\n'
                           '<profile>\n'
                           '{"name": "Ada", "location": {"city": "London", "country": "UK"}, '
                           '"interests": ["math", "machines"]}\n'
                           '</profile>\n'
                           '\n'
                           '<question>\n'
                           'What should we mention?\n'
                           '</question>'}]},
     {"role": "assistant",
      "content": '<answer>\n{"answer": "Mention analytical engines.", "sources": ["demo"]}\n</answer>'},
     {"role": "user",
      "content": '<profile>\n'
                 '{"name": "Ada", "location": {"city": "London", "country": "UK"}, "interests": '
                 '["math", "machines"]}\n'
                 '</profile>\n'
                 '\n'
                 '<question>\n'
                 'Who is Ada?\n'
                 '</question>'},
     {"role": "assistant",
      "content": '<answer>\n{"answer": "Ada is a mathematician.", "sources": ["memory"]}\n</answer>'},
     {"role": "user",
      "content": [{"type": "text", "text": "<image>\n"},
                  {"type": "image_url", "image_url": {"url": "https://example.com/current.png"}},
                  {"type": "text",
                   "text": '\n'
                           '</image>\n'
                           '\n'
                           '<tools>\n'
                           '["search, whose description is <desc>Search for documents.</desc>. It '
                           "takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', "
                           '\'default\': 3}}."]\n'
                           '</tools>\n'
                           '\n'
                           '<profile>\n'
                           '{"name": "Grace", "location": {"city": "Arlington", "country": "USA"}, '
                           '"interests": ["compilers", "navy"]}\n'
                           '</profile>\n'
                           '\n'
                           '<question>\n'
                           'What should the answer include?\n'
                           '</question>\n'
                           '\n'
                           'Respond with the corresponding output fields wrapped in XML tags '
                           '`<answer>`.'}]}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs

def test_xml_adapter_format_exact_messages_with_nested_pydantic_output():
    class XmlAddress(pydantic.BaseModel):
        city: str
        country: str

    class XmlSummary(pydantic.BaseModel):
        title: str
        address: XmlAddress

    class PydanticSignature(dspy.Signature):
        question: str = dspy.InputField()
        summary: XmlSummary = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(dspy.XMLAdapter(), PydanticSignature, [], {"question": "Summarize"})

    expected_messages = [{"role": "system",
      "content": 'Your input fields are:\n'
                 '1. `question` (str):\n'
                 'Your output fields are:\n'
                 '1. `summary` (XmlSummary):\n'
                 'All interactions will be structured in the following way, with the appropriate '
                 'values filled in.\n'
                 '\n'
                 '<question>\n'
                 '{question}\n'
                 '</question>\n'
                 '\n'
                 '<summary>\n'
                 '{summary}        # note: the value you produce must adhere to the JSON schema: '
                 '{"type": "object", "$defs": {"XmlAddress": {"type": "object", "properties": {"city": '
                 '{"type": "string", "title": "City"}, "country": {"type": "string", "title": '
                 '"Country"}}, "required": ["city", "country"], "title": "XmlAddress"}}, "properties": '
                 '{"address": {"$ref": "#/$defs/XmlAddress"}, "title": {"type": "string", "title": '
                 '"Title"}}, "required": ["title", "address"], "title": "XmlSummary"}\n'
                 '</summary>\n'
                 'In adhering to this structure, your objective is: \n'
                 '        Given the fields `question`, produce the fields `summary`.'},
     {"role": "user",
      "content": "<question>\n"
                 "Summarize\n"
                 "</question>\n"
                 "\n"
                 "Respond with the corresponding output fields wrapped in XML tags `<summary>`."}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_xml_adapter_format_exact_messages_with_incomplete_demo():
    class IncompleteDemoSignature(dspy.Signature):
        question: str = dspy.InputField()
        context: str = dspy.InputField()
        answer: str = dspy.OutputField()
        score: float = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(dspy.XMLAdapter(),
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
                 "2. `score` (float):\n"
                 "All interactions will be structured in the following way, with the appropriate "
                 "values filled in.\n"
                 "\n"
                 "<question>\n"
                 "{question}\n"
                 "</question>\n"
                 "\n"
                 "<context>\n"
                 "{context}\n"
                 "</context>\n"
                 "\n"
                 "<answer>\n"
                 "{answer}\n"
                 "</answer>\n"
                 "\n"
                 "<score>\n"
                 "{score}        # note: the value you produce must be a single float value\n"
                 "</score>\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Given the fields `question`, `context`, produce the fields `answer`, "
                 "`score`."},
     {"role": "user",
      "content": "This is an example of the task, though some input or output fields are not "
                 "supplied.\n"
                 "\n"
                 "<question>\n"
                 "Q1\n"
                 "</question>"},
     {"role": "assistant",
      "content": "<answer>\n"
                 "A1\n"
                 "</answer>\n"
                 "\n"
                 "<score>\n"
                 "Not supplied for this particular example. \n"
                 "</score>"},
     {"role": "user",
      "content": "<question>\n"
                 "Q2\n"
                 "</question>\n"
                 "\n"
                 "<context>\n"
                 "C2\n"
                 "</context>\n"
                 "\n"
                 "Respond with the corresponding output fields wrapped in XML tags `<answer>`, then "
                 "`<score>`."}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_format_system_message():
    class MySignature(dspy.Signature):
        """Answer the question with multiple answers and scores"""

        question: str = dspy.InputField()
        answers: list[str] = dspy.OutputField()
        scores: list[float] = dspy.OutputField()

    adapter = dspy.XMLAdapter()
    system_message = adapter.format_system_message(MySignature)

    expected_system_message = """Your input fields are:
1. `question` (str):
Your output fields are:
1. `answers` (list[str]):\x20
2. `scores` (list[float]):
All interactions will be structured in the following way, with the appropriate values filled in.

<question>
{question}
</question>

<answers>
{answers}        # note: the value you produce must adhere to the JSON schema: {"type": "array", "items": {"type": "string"}}
</answers>

<scores>
{scores}        # note: the value you produce must adhere to the JSON schema: {"type": "array", "items": {"type": "number"}}
</scores>
In adhering to this structure, your objective is:\x20
        Answer the question with multiple answers and scores"""
    assert system_message == expected_system_message


def test_xml_adapter_parse_reports_value_containing_its_own_closing_tag():
    class CodeSig(dspy.Signature):
        task: str = dspy.InputField()
        code: str = dspy.OutputField()

    # The wire format does not escape values, so `</code>` inside the value is indistinguishable
    # from the tag that ends it. XMLAdapter used to resolve that silently, returning the
    # truncated "if x:\n    print('". It now reports the ambiguity instead.
    completion = "<code>\nif x:\n    print('</code>')\n</code>\n"
    with pytest.raises(AdapterParseError, match="unmatched"):
        dspy.XMLAdapter().parse(CodeSig, completion)

    # ChatAdapter's delimiters cannot collide with the value, so it carries the string fine.
    # The point of the error is to send users here rather than hand back a short string.
    expected = "if x:\n    print('</code>')"
    chat_completion = dspy.ChatAdapter().format_assistant_message_content(CodeSig, {"code": expected})
    assert dspy.ChatAdapter().parse(CodeSig, chat_completion) == {"code": expected}


def test_xml_adapter_parse_reports_second_field_value_containing_its_own_closing_tag():
    class Two(dspy.Signature):
        q: str = dspy.InputField()
        reasoning: str = dspy.OutputField()
        answer: str = dspy.OutputField()

    # Previously returned {"reasoning": "think", "answer": "see"} with no error raised.
    completion = "<reasoning>\nthink\n</reasoning>\n<answer>\nsee </answer> above\n</answer>\n"
    with pytest.raises(AdapterParseError, match="`answer`"):
        dspy.XMLAdapter().parse(Two, completion)


def test_xml_adapter_format_parse_round_trip_reports_closing_tag_in_value():
    class CodeSig(dspy.Signature):
        task: str = dspy.InputField()
        code: str = dspy.OutputField()

    value = "before </code> after"
    adapter = dspy.XMLAdapter()

    # The wire format cannot express this value, so the adapter cannot read back what it wrote.
    # It says so instead of silently returning "before". Escaping the tag here would need a full
    # entity scheme: a half-measure that only rewrites `</code>` is not reversible, because a
    # value legitimately containing the escaped text would be corrupted on the way back.
    assistant_message = adapter.format_assistant_message_content(CodeSig, {"code": value})
    with pytest.raises(AdapterParseError, match="unmatched"):
        adapter.parse(CodeSig, assistant_message)

    # An entity-looking value survives untouched, which is what a rewrite would have broken.
    literal = "use &lt;/code> to escape it"
    formatted = adapter.format_field_with_value(
        {FieldInfoWithName(name="code", info=CodeSig.output_fields["code"]): literal}
    )
    assert adapter.parse(CodeSig, formatted) == {"code": literal}


def test_xml_adapter_parse_allows_closing_tag_nested_in_another_field():
    class Two(dspy.Signature):
        q: str = dspy.InputField()
        answer: str = dspy.OutputField()
        explanation: str = dspy.OutputField()

    # `</answer>` inside the explanation block belongs to the explanation's value and is fully
    # accounted for, so the answer field is not ambiguous and must still parse.
    completion = "<answer>42</answer>\n<explanation>the </answer> tag ends it</explanation>"
    assert dspy.XMLAdapter().parse(Two, completion) == {
        "answer": "42",
        "explanation": "the </answer> tag ends it",
    }


def test_xml_adapter_parse_keeps_first_of_duplicate_field_blocks():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    # A repeated block must not be merged into one value nor resolved to the last block:
    # the first complete block wins, matching ChatAdapter's "first header wins" rule.
    completion = "<answer>Paris</answer>\n<answer>Berlin</answer>\n<answer>Rome</answer>"
    assert dspy.XMLAdapter().parse(TestSignature, completion) == {"answer": "Paris"}


def test_xml_adapter_parse_value_containing_next_field_opening_tag():
    class Two(dspy.Signature):
        q: str = dspy.InputField()
        reasoning: str = dspy.OutputField()
        answer: str = dspy.OutputField()

    # The reasoning text mentions `<answer>`, which must not be mistaken for the start of the
    # answer field, and the answer itself must still be found afterwards.
    completion = "<reasoning>put it in <answer> tags</reasoning><answer>42</answer>"
    assert dspy.XMLAdapter().parse(Two, completion) == {
        "reasoning": "put it in <answer> tags",
        "answer": "42",
    }


def test_xml_adapter_parse_reports_trailing_prose_mentioning_the_closing_tag():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    # "final" is the intuitive reading, but the adapter cannot prove the sign-off is not part of
    # the value, so it reports rather than guesses. Guessing the other way silently corrupts the
    # value, which is the failure this module is trying to stop making.
    completion = "<answer>final</answer>\n\nRemember to close with `</answer>`."
    with pytest.raises(AdapterParseError, match="unmatched"):
        dspy.XMLAdapter().parse(TestSignature, completion)


def test_xml_adapter_parse_ignores_duplicated_closing_tag():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    # Emitting the closing tag twice is a common model slip. Only whitespace separates the copies,
    # so no content can be hidden between them and the value is unambiguous: no error.
    assert dspy.XMLAdapter().parse(TestSignature, "<answer>42</answer>\n</answer>") == {"answer": "42"}
    assert dspy.XMLAdapter().parse(TestSignature, "<answer>42</answer>\n</answer>\n</answer>") == {"answer": "42"}

    # But once real text sits between the copies, which one closes the value is unknowable.
    with pytest.raises(AdapterParseError, match="unmatched"):
        dspy.XMLAdapter().parse(TestSignature, "<answer>see </answer> here</answer>\n</answer>")


def test_xml_adapter_parse_ignores_duplicated_closing_tag_before_next_field():
    class Two(dspy.Signature):
        q: str = dspy.InputField()
        reasoning: str = dspy.OutputField()
        answer: str = dspy.OutputField()

    # Same slip in a non-final field: the duplicate sits between the real closing tag and the
    # next field's opening tag, so neither field may swallow it.
    completion = "<reasoning>think</reasoning>\n</reasoning>\n<answer>42</answer>\n</answer>"
    assert dspy.XMLAdapter().parse(Two, completion) == {"reasoning": "think", "answer": "42"}


def test_xml_adapter_parse_allows_bare_ampersand_and_unknown_tags():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    # The parser is text-anchored, not a strict XML parser: a bare `&` is not an entity error,
    # and tags that are not output fields are ignored rather than parsed.
    completion = "<thinking>hmm</thinking><answer>Tom & Jerry</answer>"
    assert dspy.XMLAdapter().parse(TestSignature, completion) == {"answer": "Tom & Jerry"}


def test_xml_adapter_parse_raises_when_closing_tag_is_missing():
    class Two(dspy.Signature):
        q: str = dspy.InputField()
        reasoning: str = dspy.OutputField()
        answer: str = dspy.OutputField()

    # An opening tag with no closing tag anywhere must fail loudly rather than guess a span.
    # The truncation bug was bad because it was silent; a field we cannot delimit stays absent.
    completion = "<reasoning>r</reasoning>\n<answer>unterminated"
    with pytest.raises(dspy.utils.exceptions.AdapterParseError) as e:
        dspy.XMLAdapter().parse(Two, completion)
    assert e.value.parsed_result == {"reasoning": "r"}
    assert "answer" in str(e.value)


def test_xml_adapter_parse_allows_closing_tag_nested_in_a_non_output_tag():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    # A tag that is not an output field still delimits its own content, so a copy of `</answer>`
    # inside it is accounted for and hides nothing from the answer's value.
    completion = "<answer>42</answer>\n<thinking>close with </answer></thinking>"
    assert dspy.XMLAdapter().parse(TestSignature, completion) == {"answer": "42"}


def test_xml_adapter_parse_reports_the_missing_field_not_the_stray_tag():
    class Two(dspy.Signature):
        q: str = dspy.InputField()
        reasoning: str = dspy.OutputField()
        answer: str = dspy.OutputField()

    # The `</answer>` sits in a block that is never closed, so the real fault is the missing
    # `reasoning` field. Reporting the ambiguity first would point at the wrong field entirely.
    completion = "<answer>a</answer>\n<reasoning>oops </answer> unclosed"
    with pytest.raises(dspy.utils.exceptions.AdapterParseError) as e:
        dspy.XMLAdapter().parse(Two, completion)
    assert e.value.parsed_result == {"answer": "a"}
    assert "unmatched" not in str(e.value)


def test_xml_adapter_parse_reports_ambiguity_after_a_benign_duplicate():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    # The first surplus tag is a harmless duplicate, but the scan must carry on past it: the next
    # copy has real text before it, so content is being dropped and that has to be reported.
    completion = "<answer>42</answer>\n</answer> rest of value</answer>"
    with pytest.raises(dspy.utils.exceptions.AdapterParseError, match="unmatched"):
        dspy.XMLAdapter().parse(TestSignature, completion)


def test_xml_adapter_parse_ignores_duplicated_closing_tag_after_a_balanced_block():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    # A block between the value and the surplus tag is closed, so its text belongs to that block
    # and the longer reading hides nothing. Treating the block as content would reject a common
    # slip - trailing duplicate tag after later output - that has always parsed.
    completion = "<answer>42</answer>\n<thinking>done</thinking>\n</answer>"
    assert dspy.XMLAdapter().parse(TestSignature, completion) == {"answer": "42"}

    # Text sitting outside those blocks is still unaccounted for, so it is still reported.
    with pytest.raises(dspy.utils.exceptions.AdapterParseError, match="unmatched"):
        dspy.XMLAdapter().parse(TestSignature, "<answer>42</answer>\n<thinking>done</thinking> and</answer>")


def test_xml_adapter_parse_ignores_earlier_field_closing_tag_after_a_balanced_block():
    class Two(dspy.Signature):
        q: str = dspy.InputField()
        reasoning: str = dspy.OutputField()
        answer: str = dspy.OutputField()

    # Same slip for a field the later blocks are nested after rather than before: the answer
    # block accounts for everything between reasoning's real closing tag and the stray copy.
    completion = "<reasoning>think</reasoning>\n<answer>42</answer>\n</reasoning>"
    assert dspy.XMLAdapter().parse(Two, completion) == {"reasoning": "think", "answer": "42"}


def test_xml_adapter_parse_reports_ambiguity_despite_a_mentioned_opening_tag():
    class Two(dspy.Signature):
        q: str = dspy.InputField()
        answer: str = dspy.OutputField()
        thinking: str = dspy.OutputField()

    # The `<answer>` inside the thinking block is part of that block's value, not the start of a
    # second answer block, so it cannot explain away the trailing `</answer>`. Letting a mere
    # mention account for it would hand back "42" and drop the correction after it.
    completion = (
        "<answer>42</answer>\n<thinking>wrap it in <answer> tags</thinking>\nActually the answer is 43</answer>"
    )
    with pytest.raises(AdapterParseError, match="unmatched"):
        dspy.XMLAdapter().parse(Two, completion)


def test_xml_adapter_parse_allows_a_repeated_block_nesting_the_same_tag():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    # The first block wins, and the trailing `</answer>` closes the repeated block rather than
    # hanging off the value: every byte between the two is inside that block, so nothing is
    # hidden and the parse must stay silent.
    completion = "<answer>a</answer>\n<answer>x<answer>y</answer></answer>"
    assert dspy.XMLAdapter().parse(TestSignature, completion) == {"answer": "a"}


def test_xml_adapter_parse_rejects_a_nested_span_that_is_not_self_contained():
    class CodeSig(dspy.Signature):
        task: str = dspy.InputField()
        code: str = dspy.OutputField()

    class CodeExplanation(dspy.Signature):
        task: str = dspy.InputField()
        code: str = dspy.OutputField()
        explanation: str = dspy.OutputField()

    adapter = dspy.XMLAdapter()

    # The depth-balanced partner here lives inside `<note>`, so widening to it would build the
    # value out of another element's markup, trailing `<note>` and all. The span is not
    # self-contained, so the lazy reading stands.
    completion = "<code>\n<code>x</code>\n<note></code></note>"
    assert adapter.parse(CodeSig, completion) == {"code": "<code>x"}

    # Here the span *is* balanced but contains another output field, so widening would eat
    # `explanation` whole and report it missing. Both fields must still be found.
    completion = "<code>\n<code>x</code>\n<explanation>hi</explanation>\n</code>"
    assert adapter.parse(CodeExplanation, completion) == {"code": "<code>x", "explanation": "hi"}

    # A nested value that closes before the next field keeps the widened reading.
    completion = "<code>\n<code>x</code>\n</code>\n<explanation>hi</explanation>"
    assert adapter.parse(CodeExplanation, completion) == {"code": "<code>x</code>", "explanation": "hi"}


def test_xml_adapter_parse_recovers_a_value_that_nests_its_own_tag():
    class CodeSig(dspy.Signature):
        task: str = dspy.InputField()
        code: str = dspy.OutputField()

    class HtmlSig(dspy.Signature):
        task: str = dspy.InputField()
        html: str = dspy.OutputField()

    adapter = dspy.XMLAdapter()

    # A value that is itself a same-named element is NOT ambiguous: the tags are depth-balanced,
    # so exactly one reading exists and it can be recovered outright rather than reported.
    assert adapter.parse(CodeSig, "<code>\n<code>x</code>\n</code>") == {"code": "<code>x</code>"}
    assert adapter.parse(HtmlSig, "<html>\n<html><body>x</body></html>\n</html>") == {
        "html": "<html><body>x</body></html>",
    }
    assert adapter.parse(CodeSig, "<code>\n<code>\n<code>x</code>\n</code>\n</code>") == {
        "code": "<code>\n<code>x</code>\n</code>",
    }

    # KNOWN LIMITATION: a value that merely *ends* with its closing tag has no nested opening tag
    # to balance against, so the scan closes the block early and only whitespace follows -- nothing
    # surplus is left in open text to detect. Representing it needs escaping, which this wire
    # format does not have.
    assert adapter.parse(CodeSig, "<code>\nbefore </code>\n</code>") == {"code": "before"}

    # Nesting is gated on the inner opening tag following after whitespace only. A value that
    # merely mentions its own opening tag mid-prose keeps the lazy reading, so this still parses
    # -- an earlier attempt at a broader rule rejected it, which was a regression against main.
    class Reasoning(dspy.Signature):
        task: str = dspy.InputField()
        reasoning: str = dspy.OutputField()

    completion = "<reasoning>wrap in <reasoning> tags</reasoning>\n</reasoning>"
    assert adapter.parse(Reasoning, completion) == {"reasoning": "wrap in <reasoning> tags"}
    assert adapter.parse(CodeSig, "<code>use <code> tags</code>") == {"code": "use <code> tags"}


def test_xml_adapter_parse_scans_surplus_closing_tags_in_linear_time():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    # A degenerate completion that repeats one tag is cheap model output to produce, so the scan
    # for it has to stay linear. Re-measuring the text before each surplus tag made this
    # quadratic: 50k copies took tens of seconds before, and milliseconds now.
    completion = "<answer>42</answer>" + "</answer>" * 50_000
    start = time.perf_counter()
    assert dspy.XMLAdapter().parse(TestSignature, completion) == {"answer": "42"}
    assert time.perf_counter() - start < 5


@pytest.mark.parametrize(
    ("label", "tail"),
    [
        # A tag inside the span that pairs with nothing.
        ("unpaired", "<b>" + "</code>" * 20_000),
        # A tag inside the span that pairs after it.
        ("crossing", "<x>" + "</code>" * 20_000 + "</x>"),
        # Another output field inside the span.
        ("other field", "<answer>y</answer>" + "</code>" * 20_000),
    ],
)
def test_xml_adapter_parse_rejects_unwidenable_spans_in_linear_time(label, tail):
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    # Every `<code><code>` pair below looks like a nested value worth widening, and each tail then
    # rules the widened span out. Walking a span to reject it cost its length, so these took half a
    # minute of CPU before and milliseconds now; the answer itself is unaffected either way.
    completion = "<answer>42</answer>" + "<code><code></code>" * 20_000 + tail
    start = time.perf_counter()
    assert dspy.XMLAdapter().parse(TestSignature, completion) == {"answer": "42"}
    assert time.perf_counter() - start < 5
