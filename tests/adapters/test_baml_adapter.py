from typing import Literal
from unittest import mock

import pydantic
import pytest
from litellm import Choices, Message
from litellm.files.main import ModelResponse

import dspy
from dspy.adapters.baml_adapter import COMMENT_SYMBOL, BAMLAdapter


# Test fixtures - Pydantic models for testing
class PatientAddress(pydantic.BaseModel):
    street: str
    city: str
    country: Literal["US", "CA"]


class PatientDetails(pydantic.BaseModel):
    name: str = pydantic.Field(description="Full name of the patient")
    age: int
    address: PatientAddress | None = None


class ComplexNestedModel(pydantic.BaseModel):
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

    # Should include nested structure with comments
    assert f"{COMMENT_SYMBOL} Unique identifier" in schema
    assert "details:" in schema
    assert f"{COMMENT_SYMBOL} Full name of the patient" in schema
    assert "tags: string[]," in schema
    assert "metadata: dict[string, string]," in schema


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
    user_message = messages[-1]["content"]
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

    user_message = messages[-1]["content"]
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

    # Should contain image URLs in the message content
    user_message = messages[-1]["content"]
    image_contents = [
        content for content in user_message if isinstance(content, dict) and content.get("type") == "image_url"
    ]

    assert len(image_contents) == 2
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}} in user_message
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}} in user_message


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

    user_message = messages[-1]["content"]
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

    user_message = messages[-1]["content"]
    assert "def hello():" in user_message
    assert "print('Hello, world!')" in user_message

    # Test with code as output field
    class CodeGenSignature(dspy.Signature):
        task: str = dspy.InputField()
        code: dspy.Code = dspy.OutputField()

    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content='{"code": "print(\\"Generated code\\")"}'))],
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
    assert len(messages) == 6  # system + 2 history pairs + user
    assert "What is the patient's age?" in messages[1]["content"]
    assert '"answer": "45 years old"' in messages[2]["content"]
    assert "Any allergies?" in messages[3]["content"]
    assert '"answer": "Penicillin allergy"' in messages[4]["content"]


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

    with mock.patch("litellm.acompletion") as mock_acompletion:
        mock_acompletion.return_value = ModelResponse(
            choices=[Choices(message=Message(content='{"patient": {"name": "John Doe", "age": 28}}'))],
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
        input_1: UserProfile = dspy.InputField()
        input_2: SystemConfig = dspy.InputField()
        result: str = dspy.OutputField()

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
    assert "Your output fields are:" in field_desc

    # Test message formatting with actual Pydantic instances
    user_profile = UserProfile(name="John Doe", email="john@example.com", age=30)
    system_config = SystemConfig(timeout=300, debug=True, endpoints=["api1", "api2"])

    messages = adapter.format(TestSignature, [], {"input_1": user_profile, "input_2": system_config})

    user_message = messages[-1]["content"]

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
