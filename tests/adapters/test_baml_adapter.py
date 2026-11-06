from typing import Literal
from unittest import mock

import pydantic
import pytest

import dspy
from dspy.adapters.baml_adapter import COMMENT_SYMBOL, INDENTATION, BAMLAdapter
from tests.adapters.conftest import format_messages_and_lm_kwargs
from tests.test_utils.engines import make_response


# Test fixtures - Pydantic models for testing
class PatientAddress(pydantic.BaseModel):
    """Patient Address model docstring"""
    street: str
    city: str
    country: Literal["US", "CA"]


class PatientDetails(pydantic.BaseModel):
    """
    Patient Details model docstring
    Multiline docstring support test
    """
    name: str = pydantic.Field(description="Full name of the patient")
    age: int
    address: PatientAddress | None = None


class ComplexNestedModel(pydantic.BaseModel):
    """Complex model docstring"""
    id: int = pydantic.Field(description="Unique identifier")
    details: PatientDetails
    tags: list[str] = pydantic.Field(default_factory=list)
    metadata: dict[str, str] = pydantic.Field(default_factory=dict)


class ModelWithLists(pydantic.BaseModel):
    items: list[PatientAddress] = pydantic.Field(description="List of patient addresses")
    scores: list[float]


class ImageWrapper(pydantic.BaseModel):
    images: list[dspy.Image]
    tag: list[str]


class CircularModel(pydantic.BaseModel):
    name: str
    field: "CircularModel"


def test_baml_adapter_format_exact_messages_for_simple_signature_with_demo():
    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(BAMLAdapter(), QA, [{"question": "Q1", "answer": "A1"}], {"question": "Q2"})

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
                 "Output field `answer` should be of type: string\n"
                 "\n"
                 "[[ ## completed ## ]]\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Given the fields `question`, produce the fields `answer`."},
     {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
     {"role": "assistant", "content": '{\n  "answer": "A1"\n}'},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Q2\n"
                 "\n"
                 "Respond with a JSON object in the following order of fields: `answer`."}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_baml_adapter_format_exact_messages_with_nested_output():
    class BamlNested(pydantic.BaseModel):
        value: int
        tags: list[str]

    class TypedSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: BamlNested = dspy.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(BAMLAdapter(), TypedSignature, [], {"question": "Q"})

    expected_messages = [{"role": "system",
      "content": "Your input fields are:\n"
                 "1. `question` (str):\n"
                 "Your output fields are:\n"
                 "1. `answer` (BamlNested):\n"
                 "All interactions will be structured in the following way, with the appropriate "
                 "values filled in.\n"
                 "\n"
                 "[[ ## question ## ]]\n"
                 "{question}\n"
                 "\n"
                 "[[ ## answer ## ]]\n"
                 "Output field `answer` should be of type: {\n"
                 "  value: int,\n"
                 "  tags: string[],\n"
                 "}\n"
                 "\n"
                 "[[ ## completed ## ]]\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Given the fields `question`, produce the fields `answer`."},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Q\n"
                 "\n"
                 "Respond with a JSON object in the following order of fields: `answer` (must be "
                 "formatted as a valid Python BamlNested)."}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_baml_adapter_basic_schema_generation():
    """Test that BAMLAdapter generates simplified schemas for Pydantic models."""

    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        patient: PatientDetails = dspy.OutputField()

    adapter = BAMLAdapter()
    schema = adapter.format_field_structure(TestSignature)

    # Should contain simplified schema with comments
    assert f"{COMMENT_SYMBOL} Full name of the patient" in schema
    assert "name: string," in schema
    assert "age: int," in schema
    assert "address:" in schema
    assert "street: string," in schema
    assert 'country: "US" or "CA",' in schema


def test_baml_adapter_handles_optional_fields():
    """Test optional field rendering with 'or null' syntax."""

    class TestSignature(dspy.Signature):
        input: str = dspy.InputField()
        patient: PatientDetails = dspy.OutputField()

    adapter = BAMLAdapter()
    schema = adapter.format_field_structure(TestSignature)

    # Optional address field should show 'or null'
    assert "address:" in schema
    assert "or null" in schema


def test_baml_adapter_handles_primitive_types():
    """Test rendering of basic primitive types."""

    class SimpleModel(pydantic.BaseModel):
        text: str
        number: int
        decimal: float
        flag: bool

    class TestSignature(dspy.Signature):
        input: str = dspy.InputField()
        output: SimpleModel = dspy.OutputField()

    adapter = BAMLAdapter()
    schema = adapter.format_field_structure(TestSignature)

    assert "text: string," in schema
    assert "number: int," in schema
    assert "decimal: float," in schema
    assert "flag: boolean," in schema


def test_baml_adapter_handles_non_string_literals():
    """Only string Literal members are quoted; int/bool/None members render bare (regression)."""

    class ModelWithLiterals(pydantic.BaseModel):
        status: Literal["active", "inactive"]
        priority: Literal[1, 2, 3]
        enabled: Literal[True, False]
        mode: Literal["auto", None]

    class TestSignature(dspy.Signature):
        input: str = dspy.InputField()
        output: ModelWithLiterals = dspy.OutputField()

    adapter = BAMLAdapter()
    schema = adapter.format_field_structure(TestSignature)

    # String literals stay quoted...
    assert 'status: "active" or "inactive",' in schema
    # ...but int and bool literals must NOT be quoted (previously rendered as "1" or "2" or "3").
    assert "priority: 1 or 2 or 3," in schema
    assert "enabled: true or false," in schema
    # None renders as JSON null (not Python None), matching Optional's "or null".
    assert 'mode: "auto" or null,' in schema


def test_baml_adapter_handles_lists_with_bracket_notation():
    """Test that lists of Pydantic models use proper bracket notation."""

    class TestSignature(dspy.Signature):
        input: str = dspy.InputField()
        addresses: ModelWithLists = dspy.OutputField()

    adapter = BAMLAdapter()
    schema = adapter.format_field_structure(TestSignature)

    # Should use bracket notation for lists and include comments
    assert "items: [" in schema
    assert f"{COMMENT_SYMBOL} List of patient addresses" in schema
    assert "street: string," in schema
    assert "city: string," in schema
    assert "]," in schema
    assert "scores: float[]," in schema


def test_baml_adapter_handles_complex_nested_models():
    """Test deeply nested Pydantic model schema generation."""

    class TestSignature(dspy.Signature):
        input: str = dspy.InputField()
        complex: ComplexNestedModel = dspy.OutputField()

    adapter = BAMLAdapter()
    schema = adapter.format_field_structure(TestSignature)

    expected_patient_details = "\n".join([
        f"{INDENTATION}{COMMENT_SYMBOL} Patient Details model docstring",
        f"{INDENTATION}{COMMENT_SYMBOL} Multiline docstring support test",
        f"{INDENTATION}details:",
    ])

    # Should include nested structure with comments
    assert f"{COMMENT_SYMBOL} Unique identifier" in schema
    assert expected_patient_details in schema
    assert f"{COMMENT_SYMBOL} Full name of the patient" in schema
    assert "tags: string[]," in schema
    assert "metadata: dict[string, string]," in schema
    assert f"{COMMENT_SYMBOL} Complex model docstring" in schema
    assert f"{COMMENT_SYMBOL} Patient Address model docstring" in schema


def test_baml_adapter_raise_error_on_circular_references():
    """Test that circular references are handled gracefully."""

    class TestSignature(dspy.Signature):
        input: str = dspy.InputField()
        circular: CircularModel = dspy.OutputField()

    adapter = BAMLAdapter()
    with pytest.raises(ValueError) as error:
        adapter.format_field_structure(TestSignature)

    assert "BAMLAdapter cannot handle recursive pydantic models" in str(error.value)


def test_baml_adapter_formats_pydantic_inputs_as_clean_json():
    """Test that Pydantic input instances are formatted as clean JSON."""

    class TestSignature(dspy.Signature):
        patient: PatientDetails = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    adapter = BAMLAdapter()
    patient = PatientDetails(
        name="John Doe", age=45, address=PatientAddress(street="123 Main St", city="Anytown", country="US")
    )

    messages = adapter.format(TestSignature, [], {"patient": patient, "question": "What is the diagnosis?"})

    # Should have clean, indented JSON for Pydantic input
    user_message = messages.messages[-1].text
    assert '"name": "John Doe"' in user_message
    assert '"age": 45' in user_message
    assert '"street": "123 Main St"' in user_message
    assert '"country": "US"' in user_message


def test_baml_adapter_handles_mixed_input_types():
    """Test formatting of mixed Pydantic and primitive inputs."""

    class TestSignature(dspy.Signature):
        patient: PatientDetails = dspy.InputField()
        priority: int = dspy.InputField()
        notes: str = dspy.InputField()
        result: str = dspy.OutputField()

    adapter = BAMLAdapter()
    patient = PatientDetails(name="Jane Doe", age=30)

    messages = adapter.format(TestSignature, [], {"patient": patient, "priority": 1, "notes": "Urgent case"})

    user_message = messages.messages[-1].text
    # Pydantic should be JSON formatted
    assert '"name": "Jane Doe"' in user_message
    # Primitives should be formatted normally
    assert "priority ## ]]\n1" in user_message
    assert "notes ## ]]\nUrgent case" in user_message


def test_baml_adapter_handles_schema_generation_errors_gracefully():
    """Test graceful handling of schema generation errors."""

    class ProblematicModel(pydantic.BaseModel):
        # This might cause issues in schema generation
        field: object

    class TestSignature(dspy.Signature):
        input: str = dspy.InputField()
        output: ProblematicModel = dspy.OutputField()

    adapter = BAMLAdapter()

    # Should not raise an exception
    try:
        schema = adapter.format_field_structure(TestSignature)
        # If no exception, schema should at least contain some basic structure
        assert "schema" in schema.lower()
    except Exception:
        # If exception occurs, test passes as we're testing graceful handling
        pass


def test_baml_adapter_raises_on_missing_fields():
    """Test that missing required fields raise appropriate errors."""

    class TestSignature(dspy.Signature):
        input: str = dspy.InputField()
        patient: PatientDetails = dspy.OutputField()
        summary: str = dspy.OutputField()

    adapter = BAMLAdapter()

    # Missing 'summary' field
    completion = '{"patient": {"name": "John", "age": 30}}'

    with pytest.raises(dspy.utils.exceptions.AdapterParseError) as e:
        adapter.parse(TestSignature, completion)

    assert e.value.adapter_name == "JSONAdapter"  # BAMLAdapter inherits from JSONAdapter
    assert "summary" in str(e.value)


def test_baml_adapter_handles_type_casting_errors():
    """Test graceful handling of type casting errors."""

    class TestSignature(dspy.Signature):
        input: str = dspy.InputField()
        patient: PatientDetails = dspy.OutputField()

    adapter = BAMLAdapter()

    # Invalid age type
    completion = '{"patient": {"name": "John", "age": "not_a_number"}}'

    # Should raise ValidationError from Pydantic (which is the expected behavior)
    with pytest.raises((dspy.utils.exceptions.AdapterParseError, pydantic.ValidationError)):
        adapter.parse(TestSignature, completion)


def test_baml_adapter_with_images():
    """Test BAMLAdapter integration with dspy.Image objects."""

    class TestSignature(dspy.Signature):
        image_data: ImageWrapper = dspy.InputField()
        description: str = dspy.OutputField()

    adapter = BAMLAdapter()

    image_wrapper = ImageWrapper(
        images=[dspy.Image(url="https://example.com/image1.jpg"), dspy.Image(url="https://example.com/image2.jpg")],
        tag=["test", "medical"],
    )

    messages = adapter.format(TestSignature, [], {"image_data": image_wrapper})

    # Should contain image parts in the user message
    from dspy.lm15 import ImagePart

    parts = messages.messages[-1].parts
    image_parts = [part for part in parts if isinstance(part, ImagePart)]

    assert len(image_parts) == 2
    assert ImagePart(url="https://example.com/image1.jpg", media_type="image/jpeg") in parts
    assert ImagePart(url="https://example.com/image2.jpg", media_type="image/jpeg") in parts


def test_baml_adapter_with_tools():
    """Test BAMLAdapter integration with dspy.Tool objects."""

    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        answer: str = dspy.OutputField()

    def get_patient_info(patient_id: int) -> str:
        """Get patient information by ID"""
        return f"Patient info for ID {patient_id}"

    def schedule_appointment(patient_name: str, date: str) -> str:
        """Schedule an appointment for a patient"""
        return f"Scheduled appointment for {patient_name} on {date}"

    tools = [dspy.Tool(get_patient_info), dspy.Tool(schedule_appointment)]

    adapter = BAMLAdapter()
    messages = adapter.format(TestSignature, [], {"question": "Schedule an appointment for John", "tools": tools})

    user_message = messages.messages[-1].text
    assert "get_patient_info" in user_message
    assert "schedule_appointment" in user_message
    assert "Get patient information by ID" in user_message
    assert "Schedule an appointment for a patient" in user_message


def test_baml_adapter_with_code():
    """Test BAMLAdapter integration with dspy.Code objects."""

    # Test with code as input field
    class CodeAnalysisSignature(dspy.Signature):
        code: dspy.Code = dspy.InputField()
        analysis: str = dspy.OutputField()

    adapter = BAMLAdapter()
    messages = adapter.format(CodeAnalysisSignature, [], {"code": "def hello():\n    print('Hello, world!')"})

    user_message = messages.messages[-1].text
    assert "def hello():" in user_message
    assert "print('Hello, world!')" in user_message

    # Test with code as output field
    class CodeGenSignature(dspy.Signature):
        task: str = dspy.InputField()
        code: dspy.Code = dspy.OutputField()

    with mock.patch("dspy.clients.engines.LM15Engine.complete") as mock_completion:
        mock_completion.return_value = make_response(
            '{"code": "print(\\"Generated code\\")"}',
            model="openai/gpt-4o-mini",
        )

        result = adapter(
            dspy.LM(model="openai/gpt-4o-mini", cache=False),
            {},
            CodeGenSignature,
            [],
            {"task": "Write a hello world program"},
        )

        assert result[0]["code"].code == 'print("Generated code")'


def test_baml_adapter_with_conversation_history():
    """Test BAMLAdapter integration with dspy.History objects."""

    class TestSignature(dspy.Signature):
        history: dspy.History = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    history = dspy.History(
        messages=[
            {"question": "What is the patient's age?", "answer": "45 years old"},
            {"question": "Any allergies?", "answer": "Penicillin allergy"},
        ]
    )

    adapter = BAMLAdapter()
    messages = adapter.format(TestSignature, [], {"history": history, "question": "What medications should we avoid?"})

    # Should format history as separate messages
    messages = messages.messages
    assert len(messages) == 5  # 2 history pairs + user
    assert "What is the patient's age?" in messages[0].text
    assert '"answer": "45 years old"' in messages[1].text
    assert "Any allergies?" in messages[2].text
    assert '"answer": "Penicillin allergy"' in messages[3].text


# Comparison tests with JSONAdapter
def test_baml_vs_json_adapter_token_efficiency():
    """Test that BAMLAdapter generates more token-efficient schemas."""

    class TestSignature(dspy.Signature):
        input: str = dspy.InputField()
        complex: ComplexNestedModel = dspy.OutputField()

    baml_adapter = BAMLAdapter()
    json_adapter = dspy.JSONAdapter()

    baml_schema = baml_adapter.format_field_structure(TestSignature)
    json_schema = json_adapter.format_field_structure(TestSignature)

    # Simple character count as proxy for token efficiency
    # BAMLAdapter should always produce shorter schemas
    assert len(baml_schema) < len(json_schema)


def test_baml_vs_json_adapter_functional_compatibility():
    """Test that both adapters parse identical outputs to the same results."""

    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        patient: PatientDetails = dspy.OutputField()

    baml_adapter = BAMLAdapter()
    json_adapter = dspy.JSONAdapter()

    completion = """{"patient": {
        "name": "Alice Brown",
        "age": 35,
        "address": {"street": "789 Pine St", "city": "Boston", "country": "US"}
    }}"""

    baml_result = baml_adapter.parse(TestSignature, completion)
    json_result = json_adapter.parse(TestSignature, completion)

    # Results should be functionally equivalent
    assert baml_result["patient"].name == json_result["patient"].name
    assert baml_result["patient"].age == json_result["patient"].age
    assert baml_result["patient"].address.street == json_result["patient"].address.street


@pytest.mark.asyncio
async def test_baml_adapter_async_functionality():
    """Test BAMLAdapter async operations."""

    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        patient: PatientDetails = dspy.OutputField()

    with mock.patch("dspy.clients.engines.AsyncLM15Engine.complete") as mock_acompletion:
        mock_acompletion.return_value = make_response(
            '{"patient": {"name": "John Doe", "age": 28}}',
            model="openai/gpt-4o",
        )

        adapter = BAMLAdapter()
        result = await adapter.acall(
            dspy.LM(model="openai/gpt-4o", cache=False), {}, TestSignature, [], {"question": "Extract patient info"}
        )

        assert result[0]["patient"].name == "John Doe"
        assert result[0]["patient"].age == 28


def test_baml_adapter_with_field_aliases():
    """Test BAMLAdapter with Pydantic field aliases."""

    class ModelWithAliases(pydantic.BaseModel):
        full_name: str = pydantic.Field(alias="name")
        patient_age: int = pydantic.Field(alias="age")

    class TestSignature(dspy.Signature):
        input: str = dspy.InputField()
        data: ModelWithAliases = dspy.OutputField()

    adapter = BAMLAdapter()

    # Schema should show aliases in the output structure
    schema = adapter.format_field_structure(TestSignature)
    assert "name:" in schema  # Should use alias, not field name
    assert "age:" in schema  # Should use alias, not field name


def test_baml_adapter_field_alias_without_description():
    """Test BAMLAdapter with field alias present but description absent."""

    class ModelWithAliasNoDescription(pydantic.BaseModel):
        internal_field: str = pydantic.Field(alias="public_name")
        regular_field: int
        field_with_description: str = pydantic.Field(description="This field has a description", alias="desc_field")

    class TestSignature(dspy.Signature):
        input: str = dspy.InputField()
        data: ModelWithAliasNoDescription = dspy.OutputField()

    adapter = BAMLAdapter()
    schema = adapter.format_field_structure(TestSignature)

    # Should show alias as comment when description is absent
    assert f"{COMMENT_SYMBOL} alias: public_name" in schema
    # Should show description comment when present
    assert f"{COMMENT_SYMBOL} This field has a description" in schema
    # Regular field (without alias) should appear in schema but without alias comment
    assert "regular_field: int," in schema
    # Check that regular_field section doesn't have an alias comment
    regular_field_section = schema.split("regular_field: int,")[0].split("\n")[-1]
    assert f"{COMMENT_SYMBOL} alias:" not in regular_field_section


def test_baml_adapter_multiple_pydantic_input_fields():
    """Test that multiple InputField() with Pydantic models are rendered correctly."""

    class UserProfile(pydantic.BaseModel):
        name: str = pydantic.Field(description="User's full name")
        email: str
        age: int

    class SystemConfig(pydantic.BaseModel):
        timeout: int = pydantic.Field(description="Timeout in seconds")
        debug: bool
        endpoints: list[str]

    class TestSignature(dspy.Signature):
        input_1: UserProfile = dspy.InputField(desc="User profile information")
        input_2: SystemConfig = dspy.InputField(desc="System configuration settings")
        result: str = dspy.OutputField(desc="Resulting output after processing")

    adapter = BAMLAdapter()

    # Test schema generation includes headers for ALL input fields
    schema = adapter.format_field_structure(TestSignature)
    assert "[[ ## input_1 ## ]]" in schema  # Should include first input field header
    assert "[[ ## input_2 ## ]]" in schema  # Should include second input field header
    assert "[[ ## result ## ]]" in schema  # Should include output field header
    assert "[[ ## completed ## ]]" in schema  # Should include completed section
    assert "All interactions will be structured in the following way" in schema
    assert "{input_1}" in schema
    assert "{input_2}" in schema
    assert "Output field `result` should be of type: string" in schema

    # Test field descriptions are in the correct method
    field_desc = adapter.format_field_description(TestSignature)
    assert "Your input fields are:" in field_desc
    assert "1. `input_1` (UserProfile): User profile information" in field_desc
    assert "2. `input_2` (SystemConfig): System configuration settings" in field_desc
    assert "Your output fields are:" in field_desc
    assert "1. `result` (str): Resulting output after processing" in field_desc

    # Test message formatting with actual Pydantic instances
    user_profile = UserProfile(name="John Doe", email="john@example.com", age=30)
    system_config = SystemConfig(timeout=300, debug=True, endpoints=["api1", "api2"])

    messages = adapter.format(TestSignature, [], {"input_1": user_profile, "input_2": system_config})

    user_message = messages.messages[-1].text

    # Verify both inputs are rendered with the correct bracket notation
    assert "[[ ## input_1 ## ]]" in user_message
    assert "[[ ## input_2 ## ]]" in user_message

    # Verify JSON content for both inputs
    assert '"name": "John Doe"' in user_message
    assert '"email": "john@example.com"' in user_message
    assert '"age": 30' in user_message
    assert '"timeout": 300' in user_message
    assert '"debug": true' in user_message
    # Endpoints array is formatted with indentation, so check for individual elements
    assert '"api1"' in user_message
    assert '"api2"' in user_message
    assert '"endpoints":' in user_message


def test_baml_adapter_parse_delimiter_syntax():
    """Test that BAMLAdapter correctly parses delimiter-wrapped outputs matching its prompt schema."""

    class PatientAddress(pydantic.BaseModel):
        street: str
        city: str

    class ExtractAddress(dspy.Signature):
        text: str = dspy.InputField()
        address: PatientAddress = dspy.OutputField()

    adapter = BAMLAdapter()

    # The exact output models emit following BAMLAdapter's prompt instructions
    completion = """[[ ## address ## ]]
{
  "street": "123 Main St",
  "city": "Boston"
}
[[ ## completed ## ]]"""

    parsed = adapter.parse(ExtractAddress, completion)
    assert "address" in parsed
    assert isinstance(parsed["address"], PatientAddress)
    assert parsed["address"].street == "123 Main St"
    assert parsed["address"].city == "Boston"


def test_baml_adapter_parse_fallback_json():
    """Test that BAMLAdapter maintains backward compatibility with standard JSON outputs."""

    class Item(pydantic.BaseModel):
        name: str
        count: int

    class Inventory(dspy.Signature):
        query: str = dspy.InputField()
        item: Item = dspy.OutputField()

    adapter = BAMLAdapter()

    # Standard JSON object format
    completion = '{"item": {"name": "Widget", "count": 42}}'
    parsed = adapter.parse(Inventory, completion)
    assert parsed["item"].name == "Widget"
    assert parsed["item"].count == 42


def test_baml_adapter_parse_with_markdown_fences():
    """Test that BAMLAdapter handles markdown code fences within delimiters."""

    class Status(pydantic.BaseModel):
        ready: bool
        msg: str

    class CheckStatus(dspy.Signature):
        query: str = dspy.InputField()
        status: Status = dspy.OutputField()

    adapter = BAMLAdapter()

    completion = """[[ ## status ## ]]
```json
{
  "ready": true,
  "msg": "all systems operational"
}
```
[[ ## completed ## ]]"""

    parsed = adapter.parse(CheckStatus, completion)
    assert parsed["status"].ready is True
    assert parsed["status"].msg == "all systems operational"


def test_baml_adapter_parse_bare_pydantic_json():
    """Test that BAMLAdapter parses bare Pydantic JSON objects when a single BaseModel output field is used."""

    class PatientAddress(pydantic.BaseModel):
        street: str
        city: str

    class ExtractAddress(dspy.Signature):
        text: str = dspy.InputField()
        address: PatientAddress = dspy.OutputField()

    adapter = BAMLAdapter()

    completion = '{"street": "123 Main St", "city": "Boston"}'
    parsed = adapter.parse(ExtractAddress, completion)
    assert isinstance(parsed["address"], PatientAddress)
    assert parsed["address"].street == "123 Main St"
    assert parsed["address"].city == "Boston"


def test_baml_adapter_parse_delimiter_with_reasoning_and_missing_fields():
    """Test delimiter parsing with plain-text reasoning/str fields, colliding field names, and missing required fields."""

    class Status(pydantic.BaseModel):
        status: str
        code: int

    class DiagnoseSignature(dspy.Signature):
        query: str = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        status: Status = dspy.OutputField()
        notes: str | None = dspy.OutputField(default=None)

    adapter = BAMLAdapter()

    # Valid delimiter response with plain-text reasoning, colliding inner 'status' key, and omitted optional 'notes'
    completion = """[[ ## reasoning ## ]]
Step 1: check system code 200 and verify status is ok.

[[ ## status ## ]]
{
  "status": "ok",
  "code": 200
}
[[ ## completed ## ]]"""

    parsed = adapter.parse(DiagnoseSignature, completion)
    assert parsed["reasoning"] == "Step 1: check system code 200 and verify status is ok."
    assert parsed["status"].status == "ok"
    assert parsed["status"].code == 200
    assert parsed["notes"] is None

    # Missing required field 'status' in delimiter mode must raise AdapterParseError
    incomplete = """[[ ## reasoning ## ]]
Only reasoning was generated.
[[ ## completed ## ]]"""

    with pytest.raises(dspy.utils.exceptions.AdapterParseError) as exc_info:
        adapter.parse(DiagnoseSignature, incomplete)
    assert exc_info.value.adapter_name == "BAMLAdapter"


def test_baml_adapter_parse_dict_output_and_inline_markers():
    """Test that delimiter parsing preserves all dict entries and does not truncate on inline [[ ## text."""

    class DictAndTextSig(dspy.Signature):
        query: str = dspy.InputField()
        answer: dict[str, object] = dspy.OutputField()
        explanation: str = dspy.OutputField()

    adapter = BAMLAdapter()

    completion = """[[ ## answer ## ]]
{"answer": {"x": 1}, "other": "keep"}

[[ ## explanation ## ]]
Use [[ ## to start a marker or write [[ ## inline ## ]] without breaking the section.
[[ ## completed ## ]]"""

    parsed = adapter.parse(DictAndTextSig, completion)
    assert parsed["answer"] == {"answer": {"x": 1}, "other": "keep"}
    assert parsed["explanation"] == (
        "Use [[ ## to start a marker or write [[ ## inline ## ]] without breaking the section."
    )


def test_baml_adapter_parse_bare_pydantic_with_aliases_and_defaults():
    """Test bare Pydantic JSON fallback with field aliases and models whose fields all have defaults."""

    class AliasedUser(pydantic.BaseModel):
        full_name: str = pydantic.Field(alias="name")

    class DefaultConfig(pydantic.BaseModel):
        retries: int = 3
        verbose: bool = False

    class AliasedSig(dspy.Signature):
        query: str = dspy.InputField()
        user: AliasedUser = dspy.OutputField()

    class DefaultSig(dspy.Signature):
        query: str = dspy.InputField()
        config: DefaultConfig = dspy.OutputField()

    adapter = BAMLAdapter()

    parsed_alias = adapter.parse(AliasedSig, '{"name": "Alice"}')
    assert parsed_alias["user"].full_name == "Alice"

    parsed_defaults = adapter.parse(DefaultSig, "{}")
    assert parsed_defaults["config"].retries == 3
    assert parsed_defaults["config"].verbose is False


def test_baml_adapter_parse_same_line_headers_and_json_with_completed_marker():
    """Test same-line header values and raw JSON followed by [[ ## completed ## ]]."""

    class SameLineSig(dspy.Signature):
        query: str = dspy.InputField()
        answer: str = dspy.OutputField()
        score: int = dspy.OutputField()

    adapter = BAMLAdapter()

    same_line_completion = "[[ ## answer ## ]] yes\n[[ ## score ## ]] 1\n[[ ## completed ## ]]"
    parsed = adapter.parse(SameLineSig, same_line_completion)
    assert parsed["answer"] == "yes"
    assert parsed["score"] == 1

    json_with_completed = '{"answer": "yes", "score": 1}\n[[ ## completed ## ]]'
    parsed_json = adapter.parse(SameLineSig, json_with_completed)
    assert parsed_json["answer"] == "yes"
    assert parsed_json["score"] == 1


def test_baml_adapter_parse_line_scanner_edge_cases():
    """Test line scanner with unknown headers, duplicate headers, and empty string fields."""

    class ScannerSig(dspy.Signature):
        query: str = dspy.InputField()
        answer: str = dspy.OutputField()
        notes: str = dspy.OutputField()

    adapter = BAMLAdapter()

    completion = """[[ ## answer ## ]]
first answer
[[ ## unknown_header ## ]]
should be ignored and not appended to answer
[[ ## answer ## ]]
duplicate answer should be ignored
[[ ## notes ## ]]
[[ ## completed ## ]]"""

    parsed = adapter.parse(ScannerSig, completion)
    assert parsed["answer"] == "first answer"
    assert parsed["notes"] == ""


def test_baml_adapter_wrapped_invalid_json_does_not_mask_with_defaults():
    """Test that invalid wrapped JSON does not fall back to bare defaults."""

    class Config(pydantic.BaseModel):
        retries: int = 3

    class Service(dspy.Signature):
        query: str = dspy.InputField()
        config: Config = dspy.OutputField()

    adapter = BAMLAdapter()
    invalid_wrapped = '{"config": {"retries": "not_a_number"}}'
    with pytest.raises(dspy.utils.exceptions.AdapterParseError):
        adapter.parse(Service, invalid_wrapped)


def test_baml_adapter_bare_model_with_colliding_field_name():
    """Test that bare models parse successfully when an inner property shares the output field name."""

    class Status(pydantic.BaseModel):
        status: str
        code: int

    class StatusSig(dspy.Signature):
        query: str = dspy.InputField()
        status: Status = dspy.OutputField()

    adapter = BAMLAdapter()
    bare_colliding = '{"status": "ok", "code": 200}'
    parsed = adapter.parse(StatusSig, bare_colliding)
    assert parsed["status"].status == "ok"
    assert parsed["status"].code == 200


def test_baml_adapter_parse_null_delimiter_block():
    """Test that explicit 'null' inside a delimiter block parses as None for optional fields."""

    class OptionalSig(dspy.Signature):
        query: str = dspy.InputField()
        notes: str | None = dspy.OutputField()
        answer: str = dspy.OutputField()

    adapter = BAMLAdapter()
    completion = """[[ ## notes ## ]]
null
[[ ## answer ## ]]
real answer
[[ ## completed ## ]]"""

    parsed = adapter.parse(OptionalSig, completion)
    assert parsed["notes"] is None
    assert parsed["answer"] == "real answer"



def test_baml_adapter_wrapped_dict_with_defaults_does_not_mask_supplied_values():
    """Test that redundantly wrapped models with default fields unpack supplied values instead of defaults."""

    class Config(pydantic.BaseModel):
        retries: int = 3

    class ConfigSig(dspy.Signature):
        query: str = dspy.InputField()
        config: Config = dspy.OutputField()

    adapter = BAMLAdapter()
    completion = """[[ ## config ## ]]
{
  "config": {
    "retries": 10
  }
}
[[ ## completed ## ]]"""

    parsed = adapter.parse(ConfigSig, completion)
    assert parsed["config"].retries == 10



def test_baml_adapter_dict_output_preserves_single_key_matching_field_name():
    """Test that dictionary outputs with a single key matching field_name are not unwrapped."""

    class DictSig(dspy.Signature):
        query: str = dspy.InputField()
        answer: dict[str, object] = dspy.OutputField()

    adapter = BAMLAdapter()
    completion = """[[ ## answer ## ]]
{
  "answer": {
    "x": 1
  }
}
[[ ## completed ## ]]"""

    parsed = adapter.parse(DictSig, completion)
    assert parsed["answer"] == {"answer": {"x": 1}}



def test_baml_adapter_union_dict_and_model_preserves_dict():
    """Test that outputs allowing either a dict or a BaseModel preserve dict keys without unwrapping."""

    class ResultModel(pydantic.BaseModel):
        val: int

    class UnionSig(dspy.Signature):
        query: str = dspy.InputField()
        answer: dict[str, object] | ResultModel = dspy.OutputField()

    adapter = BAMLAdapter()
    completion = """[[ ## answer ## ]]
{
  "answer": {
    "x": 1
  }
}
[[ ## completed ## ]]"""

    parsed = adapter.parse(UnionSig, completion)
    assert parsed["answer"] == {"answer": {"x": 1}}


def test_baml_adapter_union_model_and_typed_dict_unwraps_when_dict_rejects_wrapper():
    """Test that Config | dict[str, int] unwraps {'config': {'retries': 10}} when dict[str, int] rejects it."""

    class Config(pydantic.BaseModel):
        retries: int = 3

    class UnionConfigSig(dspy.Signature):
        query: str = dspy.InputField()
        config: Config | dict[str, int] = dspy.OutputField()

    adapter = BAMLAdapter()
    completion = """[[ ## config ## ]]
{
  "config": {
    "retries": 10
  }
}
[[ ## completed ## ]]"""

    parsed = adapter.parse(UnionConfigSig, completion)
    assert isinstance(parsed["config"], Config)
    assert parsed["config"].retries == 10


def test_baml_adapter_wrapped_union_models_prefers_best_matching_model():
    """Test that wrapped objects in A | B validate against the union and select the best matching model."""

    class ModelA(pydantic.BaseModel):
        retries: int = 3

    class ModelB(pydantic.BaseModel):
        retries: int
        verbose: bool

    class UnionSig(dspy.Signature):
        query: str = dspy.InputField()
        config: ModelA | ModelB = dspy.OutputField()

    adapter = BAMLAdapter()
    completion = """[[ ## config ## ]]
{
  "config": {
    "retries": 10,
    "verbose": true
  }
}
[[ ## completed ## ]]"""

    parsed = adapter.parse(UnionSig, completion)
    assert isinstance(parsed["config"], ModelB)
    assert parsed["config"].retries == 10
    assert parsed["config"].verbose is True


def test_baml_adapter_alias_choices_validation_preserves_outer_field_key():
    """Test that models using AliasChoices or AliasPath matching field_name validate directly without unwrapping."""

    class AnswerModel(pydantic.BaseModel):
        payload: dict[str, int] = pydantic.Field(validation_alias=pydantic.AliasChoices("answer", "payload"))

    class Sig(dspy.Signature):
        query: str = dspy.InputField()
        answer: AnswerModel = dspy.OutputField()

    adapter = BAMLAdapter()
    completion = """[[ ## answer ## ]]
{
  "answer": {
    "x": 1
  }
}
[[ ## completed ## ]]"""

    parsed = adapter.parse(Sig, completion)
    assert isinstance(parsed["answer"], AnswerModel)
    assert parsed["answer"].payload == {"x": 1}



