import pydantic
import pytest
import dspy
from dspy.adapters.chat_adapter import FieldInfoWithName
from dspy.adapters.xml_adapter import XMLAdapter


def test_xml_adapter_format_and_parse_basic():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    adapter = XMLAdapter()
    # Format output fields as XML
    fields_with_values = {FieldInfoWithName(name="answer", info=TestSignature.output_fields["answer"]): "Paris"}
    xml = adapter.format_field_with_value(fields_with_values)
    assert xml.strip() == "<answer>Paris</answer>"

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
<answer>Paris</answer><explanation>The capital of France is Paris.</explanation>"""
    parsed = adapter.parse(TestSignature, completion)
    assert parsed == {"answer": "Paris", "explanation": "The capital of France is Paris."}


def test_xml_adapter_parse_raises_on_missing_field():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        explanation: str = dspy.OutputField()

    adapter = XMLAdapter()
    completion = "<answer>Paris</answer>"
    with pytest.raises(dspy.utils.exceptions.AdapterParseError):
        adapter.parse(TestSignature, completion)


def test_xml_adapter_parse_casts_types():
    class TestSignature(dspy.Signature):
        number: int = dspy.OutputField()
        flag: bool = dspy.OutputField()

    adapter = XMLAdapter()
    completion = """
<number>42</number><flag>true</flag>"""
    parsed = adapter.parse(TestSignature, completion)
    assert parsed == {"number": 42, "flag": True}


def test_xml_adapter_parse_raises_on_type_error():
    class TestSignature(dspy.Signature):
        number: int = dspy.OutputField()

    adapter = XMLAdapter()
    completion = "<number>not_a_number</number>"
    with pytest.raises(dspy.utils.exceptions.AdapterParseError):
        adapter.parse(TestSignature, completion)


def test_xml_adapter_handles_true_nested_xml_parsing():
    class InnerModel(pydantic.BaseModel):
        value: int
        label: str

    class TestSignature(dspy.Signature):
        result: InnerModel = dspy.OutputField()

    adapter = XMLAdapter()
    completion = """
<result>
    <value>5</value>
    <label>foo</label>
</result>
"""
    parsed = adapter.parse(TestSignature, completion)
    assert isinstance(parsed["result"], InnerModel)
    assert parsed["result"].value == 5
    assert parsed["result"].label == "foo"


def test_xml_adapter_formats_true_nested_xml():
    class InnerModel(pydantic.BaseModel):
        value: int
        label: str

    class TestSignature(dspy.Signature):
        result: InnerModel = dspy.OutputField()

    adapter = XMLAdapter()
    fields_with_values = {
        FieldInfoWithName(name="result", info=TestSignature.output_fields["result"]): InnerModel(value=5, label="foo")
    }
    xml = adapter.format_field_with_value(fields_with_values)
    
    # The output should be a true nested XML string
    expected_xml = "<result><value>5</value><label>foo</label></result>"
    assert xml.strip() == expected_xml.strip()


def test_xml_adapter_handles_lists_as_repeated_tags():
    class Item(pydantic.BaseModel):
        name: str
        score: float

    class TestSignature(dspy.Signature):
        items: list[Item] = dspy.OutputField()

    adapter = XMLAdapter()
    
    # Test parsing repeated tags into a list
    completion = """
<items>
    <name>a</name>
    <score>1.1</score>
</items>
<items>
    <name>b</name>
    <score>2.2</score>
</items>
"""
    parsed = adapter.parse(TestSignature, completion)
    assert isinstance(parsed["items"], list)
    assert len(parsed["items"]) == 2
    assert all(isinstance(i, Item) for i in parsed["items"])
    assert parsed["items"][0].name == "a"
    assert parsed["items"][1].score == 2.2

    # Test formatting a list into repeated tags
    items = [Item(name="x", score=3.3), Item(name="y", score=4.4)]
    fields_with_values = {FieldInfoWithName(name="items", info=TestSignature.output_fields["items"]): items}
    xml = adapter.format_field_with_value(fields_with_values)
    
    expected_xml = "<items><name>x</name><score>3.3</score></items><items><name>y</name><score>4.4</score></items>"
    assert xml.strip() == expected_xml.strip()


def test_parse_malformed_xml():
    class TestSignature(dspy.Signature):
        data: str = dspy.OutputField()

    adapter = XMLAdapter()
    completion = "<root><child>text</root>"
    with pytest.raises(dspy.utils.exceptions.AdapterParseError):
        adapter.parse(TestSignature, completion)


def test_format_and_parse_deeply_nested_model():
    class Inner(pydantic.BaseModel):
        text: str

    class Middle(pydantic.BaseModel):
        inner: Inner
        num: int

    class TestSignature(dspy.Signature):
        middle: Middle = dspy.OutputField()

    adapter = XMLAdapter()
    data = Middle(inner=Inner(text="deep"), num=123)
    fields_with_values = {FieldInfoWithName(name="middle", info=TestSignature.output_fields["middle"]): data}
    
    # Test formatting
    xml = adapter.format_field_with_value(fields_with_values)
    expected_xml = "<middle><inner><text>deep</text></inner><num>123</num></middle>"
    assert xml.strip() == expected_xml

    # Test parsing
    parsed = adapter.parse(TestSignature, xml)
    assert isinstance(parsed["middle"], Middle)
    assert parsed["middle"].inner.text == "deep"
    assert parsed["middle"].num == 123


def test_format_and_parse_empty_list():
    class TestSignature(dspy.Signature):
        items: list[str] = dspy.OutputField()

    adapter = XMLAdapter()
    
    # Test formatting
    fields_with_values = {FieldInfoWithName(name="items", info=TestSignature.output_fields["items"]): []}
    xml = adapter.format_field_with_value(fields_with_values)
    assert xml.strip() in ["<items></items>", "<items />"]

    # Test parsing
    parsed = adapter.parse(TestSignature, xml)
    assert parsed["items"] == []


def test_end_to_end_with_predict():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    # Mock LM
    class MockLM(dspy.LM):
        def __init__(self):
            self.history = []
            self.kwargs = {}

        def __call__(self, messages, **kwargs):
            self.history.append(messages)
            completion = "<answer>mocked answer</answer>"
            return [completion]

    lm = MockLM()
    lm.model = "mock-model"
    dspy.settings.configure(lm=lm, adapter=XMLAdapter())

    predict = dspy.Predict(TestSignature)
    result = predict(question="test question")

    assert result.answer == "mocked answer"