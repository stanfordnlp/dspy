import sys
from unittest import mock

import pydantic
import pytest
from litellm import Choices, Message, ModelResponse
from typing_extensions import TypedDict

import dspy
from dspy.adapters.chat_adapter import FieldInfoWithName
from dspy.adapters.xml_adapter import XMLAdapter
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


@pytest.mark.parametrize(
    "code",
    [
        "if len(page) < 20:\n    print('R&B')",
        "if n<limit and n>0 and mask & 1:\n    print(n << 2)",
        "print('<unfinished attr=\"x\"> &amp; &#65; </unrelated>')",
        "    print('</code>')\n    print('R&B')  ",
    ],
)
def test_xml_adapter_predict_accepts_python_comparisons_without_fallback(code):
    class Generate(dspy.Signature):
        question: str = dspy.InputField()
        reasoning: str = dspy.OutputField()
        code: str = dspy.OutputField()

    completion = f"<reasoning>\nKeep R&B tracks\n</reasoning>\n<code>\n{code}\n</code>"
    with mock.patch("litellm.completion") as completion_mock:
        completion_mock.return_value = ModelResponse(
            choices=[Choices(message=Message(content=completion))], model="openai/gpt-4o-mini"
        )
        with dspy.context(
            lm=dspy.LM("openai/gpt-4o-mini", engine="litellm", cache=False),
            adapter=XMLAdapter(use_json_adapter_fallback=False),
        ):
            result = dspy.Predict(Generate)(question="Write a pagination check")
    assert result.code == code
    assert result.reasoning == "Keep R&B tracks"
    assert completion_mock.call_count == 1


def test_xml_adapter_raw_text_preserves_entities_cdata_and_nested_types():
    class Result(dspy.Signature):
        code: str = dspy.OutputField()
        literal: str = dspy.OutputField()
        counts: list[int] = dspy.OutputField()

    completion = (
        "<code>if n <= 20 and mask & 1: print('&amp;')</code>"
        "<literal><![CDATA[<raw> &amp;]]></literal>"
        "<counts><item>2</item><item>7</item></counts>"
    )
    assert XMLAdapter().parse(Result, completion) == {
        "code": "if n <= 20 and mask & 1: print('&amp;')",
        "literal": "<![CDATA[<raw> &amp;]]>",
        "counts": [2, 7],
    }
    with pytest.raises(dspy.utils.exceptions.AdapterParseError):
        XMLAdapter().parse(Result, completion.replace("</code>", ""))


@pytest.mark.parametrize(
    "value",
    [
        "if x<y and y>0: print('R&B &amp;')",
        '<unfinished attr="x"> &#65; </unrelated>',
        "</code> </item> ]]> <![CDATA[literal]]>",
    ],
)
def test_xml_adapter_leaf_text_has_identical_semantics_at_every_depth(value):
    class Program(pydantic.BaseModel):
        code: str
        alternatives: list[str]
        metadata: dict[str, str]

    class Result(dspy.Signature):
        code: str = dspy.OutputField()
        program: Program = dspy.OutputField()
        programs: list[Program] = dspy.OutputField()

    program = (
        f"<code>{value}</code><alternatives><item>{value}</item></alternatives>"
        f"<metadata><source>{value}</source></metadata>"
    )
    completion = f"<code>{value}</code><program>{program}</program><programs><item>{program}</item></programs>"
    expected = Program(code=value, alternatives=[value], metadata={"source": value})
    adapter = XMLAdapter()
    if "</code>" not in value:
        assert adapter.parse(Result, completion) == {"code": value, "program": expected, "programs": [expected]}
    else:
        with pytest.raises(dspy.utils.exceptions.AdapterParseError):
            adapter.parse(Result, completion)
    formatted = adapter.format_assistant_message_content(
        Result, {"code": value, "program": expected, "programs": [expected]}
    )
    assert adapter.parse(Result, formatted) == {"code": value, "program": expected, "programs": [expected]}


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_xml_adapter_nested_tag_lines_preserve_literal_payload(newline):
    class Program(pydantic.BaseModel):
        code: str
        checks: list[str]

    class Result(dspy.Signature):
        program: Program = dspy.OutputField()

    code = f"    if n < 20:{newline}        print('</code> &amp;')  "
    completion = newline.join(
        [
            "<program>",
            "  <code>",
            code,
            "  </code>",
            "  <checks>",
            "    <item>",
            "print('</item>')",
            "    </item>",
            "    <item>",
            "R&B",
            "    </item>",
            "  </checks>",
            "</program>",
        ]
    )
    assert XMLAdapter().parse(Result, completion) == {"program": Program(code=code, checks=["print('</item>')", "R&B"])}


@pytest.mark.parametrize(
    "completion",
    [
        "<code>unterminated",
        "<code>x</wrong>",
        "<code>x</code></code>",
        "<code>x</code><code>y</code>",
        "<code>x</code><unknown>y</unknown>",
        "<code>x</code><counts><item>2</counts>",
        "<code>\nx\n</code>\nsurplus\n</code>",
        "<code>prefix <![CDATA[</code>]]>suffix</code>",
    ],
)
def test_xml_adapter_validates_outer_fields_and_structured_nesting(completion):
    class Result(dspy.Signature):
        code: str = dspy.OutputField()
        counts: list[int] = dspy.OutputField(default=[])

    with pytest.raises(dspy.utils.exceptions.AdapterParseError):
        XMLAdapter().parse(Result, completion)


def test_xml_adapter_nullable_scalar_cannot_repeat():
    class Result(dspy.Signature):
        text: str | None = dspy.OutputField()

    assert XMLAdapter().parse(Result, "<text />") == {"text": None}
    with pytest.raises(dspy.utils.exceptions.AdapterParseError):
        XMLAdapter().parse(Result, "<text /><text />")


@pytest.mark.parametrize("value", [None, ""])
def test_xml_adapter_nullable_text_round_trip_distinguishes_empty_from_null(value):
    class Result(dspy.Signature):
        text: str | None = dspy.OutputField()

    adapter = XMLAdapter()
    formatted = adapter.format_assistant_message_content(Result, {"text": value})
    assert adapter.parse(Result, formatted) == {"text": value}
    assert adapter.parse(Result, "<text>\n\n</text>") == {"text": ""}


@pytest.mark.parametrize("values", [None, [], ["", None, "R&B"]])
def test_xml_adapter_nullable_containers_round_trip(values):
    class Result(dspy.Signature):
        values: list[str | None] | None = dspy.OutputField()
        mapping: dict[str, str | None] | None = dspy.OutputField()

    expected = {
        "values": values,
        "mapping": None if values is None else {str(i): value for i, value in enumerate(values)},
    }
    adapter = XMLAdapter()
    formatted = adapter.format_assistant_message_content(Result, expected)
    assert adapter.parse(Result, formatted) == expected


def test_xml_adapter_structured_union_uses_structural_tag_lines():
    class Person(pydantic.BaseModel):
        name: str
        age: int

    class Result(dspy.Signature):
        person: str | Person = dspy.OutputField()

    adapter = XMLAdapter()
    body = "<name>Ada</name><age>not an integer</age>"
    with pytest.raises(dspy.utils.exceptions.AdapterParseError):
        adapter.parse(Result, f"<person>{body}</person>")
    with pytest.raises(dspy.utils.exceptions.AdapterParseError):
        adapter.parse(Result, "<person>\n<name>\nAda\n</name>\n<age>\nbad\n</age>\n</person>")
    assert adapter.parse(Result, f"<person>\n{body}\n</person>") == {"person": body}
    assert adapter.parse(Result, adapter.format_assistant_message_content(Result, {"person": body})) == {"person": body}
    assert adapter.parse(Result, "<person><name>Ada</name><age>36</age></person>") == {
        "person": Person(name="Ada", age=36)
    }


def test_xml_adapter_parse_raises_on_type_error():
    class TestSignature(dspy.Signature):
        number: int = dspy.OutputField()

    adapter = XMLAdapter()
    completion = "<number>not_a_number</number>"
    with pytest.raises(dspy.utils.exceptions.AdapterParseError) as e:
        adapter.parse(TestSignature, completion)
    assert "Failed to parse field" in str(e.value)


def test_xml_adapter_repeated_dict_elements_and_empty_lists():
    class TestSignature(dspy.Signature):
        counts: dict[str, list[int]] = dspy.OutputField()

    adapter = XMLAdapter()
    completion = "<counts><first>3</first><first>4</first><second>5</second></counts>"
    assert adapter.parse(TestSignature, completion) == {
        "counts": {"first": [3, 4], "second": [5]},
    }

    counts = {"postal code": [3, 4], 'quoted "key" & more': [5], "line\nbreak": [6], "-status": [7], ".status": [8]}
    field = FieldInfoWithName(name="counts", info=TestSignature.output_fields["counts"])
    formatted = adapter.format_field_with_value({field: counts})
    assert '<entry key="postal code">\n<item>\n3\n</item>\n<item>\n4\n</item>\n</entry>' in formatted
    assert adapter.parse(TestSignature, formatted) == {"counts": counts}
    assert adapter.parse(TestSignature, "<counts />") == {"counts": {}}

    class EmptySignature(dspy.Signature):
        items: list[str] = dspy.OutputField()

    field = FieldInfoWithName(name="items", info=EmptySignature.output_fields["items"])
    assert adapter.format_field_with_value({field: []}) == "<items>\n\n</items>"
    assert adapter.parse(EmptySignature, "<items />") == {"items": []}


def test_xml_adapter_uses_pydantic_field_names_as_xml_tags():
    class Address(pydantic.BaseModel):
        postal_code: str = pydantic.Field(alias="postal code")
        country_code: str = pydantic.Field(alias="country-code")

    class TestSignature(dspy.Signature):
        address: Address = dspy.OutputField()

    adapter = XMLAdapter()
    address = Address(**{"postal code": "94305", "country-code": "US"})
    xml = adapter.format_assistant_message_content(TestSignature, {"address": address})

    assert xml == "<address>\n<postal_code>\n94305\n</postal_code>\n<country_code>\nUS\n</country_code>\n</address>"
    assert "<address>\n<postal_code>\n...\n</postal_code>\n<country_code>\n...\n</country_code>\n</address>" in (
        adapter.format_field_structure(TestSignature)
    )
    assert adapter.parse(TestSignature, xml) == {"address": address}


def test_xml_adapter_typed_dict_schema_and_parsing():
    class Address(TypedDict):
        city: str

    class Order(TypedDict):
        order_id: int
        address: Address | None
        labels: list[str]

    class TestSignature(dspy.Signature):
        order: Order = dspy.OutputField()
        orders: list[Order] = dspy.OutputField()

    adapter = XMLAdapter()
    order_schema = (
        "<order>\n<order_id>\n...\n</order_id>\n<address>\n<city>\n...\n</city>\n</address>\n"
        "<labels>\n<item>\n...\n</item>\n</labels>\n</order>"
    )
    orders_schema = (
        "<orders>\n<item>\n<order_id>\n...\n</order_id>\n<address>\n<city>\n...\n</city>\n</address>\n"
        "<labels>\n<item>\n...\n</item>\n</labels>\n</item>\n</orders>"
    )
    system_instructions = adapter.format_field_structure(TestSignature)
    assert f"{order_schema}\n\n{orders_schema}" in system_instructions
    assert f"Use this nested XML structure: {order_schema} {orders_schema}" in (
        adapter.user_message_output_requirements(TestSignature)
    )

    completion = (
        "<order><order_id>1</order_id><address><city>London</city></address><labels>new</labels></order>"
        "<orders><order_id>2</order_id><address><city>Paris</city></address><labels>paid</labels></orders>"
    )
    assert adapter.parse(TestSignature, completion) == {
        "order": {"order_id": 1, "address": {"city": "London"}, "labels": ["new"]},
        "orders": [{"order_id": 2, "address": {"city": "Paris"}, "labels": ["paid"]}],
    }


def test_xml_adapter_parses_nullable_and_structured_union_fields():
    class Profile(pydantic.BaseModel):
        name: str

    class Details(TypedDict):
        label: str

    class First(pydantic.BaseModel):
        count: int

    class Second(pydantic.BaseModel):
        labels: list[str]

    class FirstDict(TypedDict):
        count: int

    class SecondDict(TypedDict):
        labels: list[str]

    class NumberList(pydantic.BaseModel):
        value: list[int]

    class TextValue(pydantic.BaseModel):
        value: str

    class ScalarLabels(pydantic.BaseModel):
        labels: str

    class ListLabels(pydantic.BaseModel):
        labels: list[str]

    class Envelope(pydantic.BaseModel):
        choice: ScalarLabels | ListLabels

    class StringFirstEnvelope(pydantic.BaseModel):
        child: str | ListLabels

    class TestSignature(dspy.Signature):
        text: str | None = dspy.OutputField()
        profile: Profile | None = dspy.OutputField()
        details: Details | None = dspy.OutputField()
        model: First | Second = dspy.OutputField()
        mapping: FirstDict | SecondDict = dspy.OutputField()
        tied: NumberList | TextValue = dspy.OutputField()
        nested: Envelope = dspy.OutputField()
        string_first: StringFirstEnvelope = dspy.OutputField()

    completion = (
        "<text /><profile /><details />"
        "<model><labels>one</labels><labels>two</labels></model>"
        "<mapping><labels>three</labels></mapping>"
        "<tied><value>later-branch-text</value></tied>"
        "<nested><choice><labels>first</labels><labels>second</labels></choice></nested>"
        "<string_first><child><labels>one</labels><labels>two</labels></child></string_first>"
    )
    assert XMLAdapter().parse(TestSignature, completion) == {
        "text": None,
        "profile": None,
        "details": None,
        "model": Second(labels=["one", "two"]),
        "mapping": {"labels": ["three"]},
        "tied": TextValue(value="later-branch-text"),
        "nested": Envelope(choice=ListLabels(labels=["first", "second"])),
        "string_first": StringFirstEnvelope(child=ListLabels(labels=["one", "two"])),
    }


def test_xml_adapter_recursive_model_schema_terminates():
    class Node(pydantic.BaseModel):
        value: str
        children: list["Node"]

    class TestSignature(dspy.Signature):
        root: Node = dspy.OutputField()

    adapter = XMLAdapter()
    assert "<root>\n<value>\n...\n</value>\n<children>\n<item>\n...\n</item>\n</children>\n</root>" in (
        adapter.format_field_structure(TestSignature)
    )
    completion = "<root><value>parent</value><children><value>child</value><children /></children></root>"
    assert adapter.parse(TestSignature, completion) == {
        "root": Node(value="parent", children=[Node(value="child", children=[])])
    }


@pytest.mark.parametrize("value", ["print('</code> & done ]]>')", "<![CDATA[literal &amp;]]>"])
def test_xml_adapter_literal_delimiters_need_no_quoting(value):
    class TestSignature(dspy.Signature):
        code: str = dspy.OutputField()

    adapter = XMLAdapter()
    formatted = adapter.format_assistant_message_content(TestSignature, {"code": value})
    assert formatted == f"<code>\n{value}\n</code>"
    assert adapter.parse(TestSignature, formatted) == {"code": value}

    with pytest.raises(dspy.utils.exceptions.AdapterParseError, match="Failed to parse XML"):
        adapter.parse(TestSignature, "<code>print('</code>')</code>")


def test_xml_adapter_format_and_parse_nested_model():
    class Address(pydantic.BaseModel):
        city: str

    class InnerModel(pydantic.BaseModel):
        value: int
        label: str
        address: Address

    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        result: InnerModel = dspy.OutputField()

    adapter = XMLAdapter()
    # Format output fields as XML
    result = InnerModel(value=5, label="foo", address=Address(city="London"))
    fields_with_values = {FieldInfoWithName(name="result", info=TestSignature.output_fields["result"]): result}
    xml = adapter.format_field_with_value(fields_with_values)
    assert xml == "<result>\n<value>\n5\n</value>\n<label>\nfoo\n</label>\n<address>\n<city>\nLondon\n</city>\n</address>\n</result>"
    assert adapter.parse(TestSignature, xml) == {"result": result}

    # Legacy JSON values inside the outer XML field remain supported.
    completion = '<result>{"value": 5, "label": "foo", "address": {"city": "London"}}</result>'
    parsed = adapter.parse(TestSignature, completion)
    assert parsed == {"result": result}


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
    assert xml == (
        "<items>\n<item>\n<name>\na\n</name>\n<score>\n1.1\n</score>\n</item>\n"
        "<item>\n<name>\nb\n</name>\n<score>\n2.2\n</score>\n</item>\n</items>"
    )
    assert adapter.parse(TestSignature, xml) == {"items": items}

    # Legacy JSON lists inside the outer XML field remain supported.
    import json

    completion = f"<items>{json.dumps([i.model_dump() for i in items])}</items>"
    assert adapter.parse(TestSignature, completion) == {"items": items}


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
    assert "<name>\nget_weather\n</name>" in xml
    assert "<result>\n125M\n</result>" in xml
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
            dspy.LM(engine="litellm", model="openai/gpt-4o-mini", cache=False),
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
        " Put structural tags on separate lines; leaf values are literal text, without escaping."
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

Respond with the corresponding output fields wrapped in XML tags `<answer>`. Put structural tags on separate lines; leaf values are literal text, without escaping.""",
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
        " Put structural tags on separate lines; leaf values are literal text, without escaping."
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

Respond with the corresponding output fields wrapped in XML tags `<judgement>`. Put structural tags on separate lines; leaf values are literal text, without escaping.""",
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

Respond with the corresponding output fields wrapped in XML tags `<answer>`, then `<score>`. Put structural tags on separate lines; leaf values are literal text, without escaping.""",
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
      "content": "Your input fields are:\n"
                 "1. `history` (History): \n"
                 "2. `image` (Image): \n"
                 "3. `tools` (list[Tool]): \n"
                 "4. `profile` (Profile): \n"
                 "5. `question` (str):\n"
                 "Your output fields are:\n"
                 "1. `answer` (AnswerCard):\n"
                 "All interactions will be structured in the following way, with the appropriate "
                 "values filled in.\n"
                 "\n"
                 "<history>\n"
                 "{history}\n"
                 "</history>\n"
                 "\n"
                 "<image>\n"
                 "{image}\n"
                 "</image>\n"
                 "\n"
                 "<tools>\n"
                 "{tools}\n"
                 "</tools>\n"
                 "\n"
                 "<profile>\n"
                 "{profile}\n"
                 "</profile>\n"
                 "\n"
                 "<question>\n"
                 "{question}\n"
                 "</question>\n"
                 "\n"
                 "<answer>\n<answer>\n...\n</answer>\n<sources>\n<item>\n...\n</item>\n</sources>\n</answer>\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Answer using all supplied context."},
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
      "content": "<answer>\n<answer>\nMention analytical engines.\n</answer>\n<sources>\n<item>\ndemo\n</item>\n</sources>\n</answer>"},
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
      "content": "<answer>\n<answer>\nAda is a mathematician.\n</answer>\n<sources>\n<item>\nmemory\n</item>\n</sources>\n</answer>"},
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
                           '`<answer>`. Put structural tags on separate lines; '
                           'leaf values are literal text, without escaping. Use this nested XML structure: '
                               '<answer>\n<answer>\n...\n</answer>\n<sources>\n<item>\n...\n</item>\n</sources>\n</answer>'}]}]
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
      "content": "Your input fields are:\n"
                 "1. `question` (str):\n"
                 "Your output fields are:\n"
                 "1. `summary` (XmlSummary):\n"
                 "All interactions will be structured in the following way, with the appropriate "
                 "values filled in.\n"
                 "\n"
                 "<question>\n"
                 "{question}\n"
                 "</question>\n"
                 "\n"
                 "<summary>\n<title>\n...\n</title>\n<address>\n<city>\n...\n</city>\n<country>\n...\n</country>\n"
                 "</address>\n</summary>\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Given the fields `question`, produce the fields `summary`."},
     {"role": "user",
      "content": "<question>\n"
                 "Summarize\n"
                 "</question>\n"
                 "\n"
                 "Respond with the corresponding output fields wrapped in XML tags `<summary>`. "
                    "Put structural tags on separate lines; leaf values are literal text, without escaping. "
                    "Use this nested XML structure: "
                    "<summary>\n<title>\n...\n</title>\n<address>\n<city>\n...\n</city>\n<country>\n...\n</country>\n"
                    "</address>\n</summary>"}]
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
                 "`<score>`. Put structural tags on separate lines; "
                 "leaf values are literal text, without escaping."}]
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
<item>
...
</item>
</answers>

<scores>
<item>
...
</item>
</scores>
In adhering to this structure, your objective is:\x20
        Answer the question with multiple answers and scores"""
    assert system_message == expected_system_message


def test_xml_adapter_missing_optional_output_fields_fall_back_to_defaults():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        note: str | None = dspy.OutputField(default="No note")
        maybe: str | None = dspy.OutputField()

    adapter = XMLAdapter()
    parsed = adapter.parse(TestSignature, "<answer>Paris</answer>")
    assert parsed == {"answer": "Paris", "note": "No note", "maybe": None}
