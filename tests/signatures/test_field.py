import dspy
from dspy.signatures.field import InputField, OutputField


def test_input_field_preserves_dict_json_schema_extra():
    user_extra = {
        "examples": ["What is the capital of France?"],
        "custom_attribute": "custom_value",
    }

    class QA(dspy.Signature):
        """Answer queries."""

        query: str = InputField(desc="User question", json_schema_extra=user_extra)
        answer: str = OutputField(desc="Model answer", json_schema_extra={"examples": ["Paris"]})

    # Test that DSPy input and output fields preserve user extra keys
    query_extra = QA.input_fields["query"].json_schema_extra
    answer_extra = QA.output_fields["answer"].json_schema_extra

    assert query_extra["examples"] == ["What is the capital of France?"]
    assert query_extra["custom_attribute"] == "custom_value"
    assert query_extra["desc"] == "User question"
    assert query_extra["__dspy_field_type"] == "input"

    assert answer_extra["examples"] == ["Paris"]
    assert answer_extra["desc"] == "Model answer"
    assert answer_extra["__dspy_field_type"] == "output"

    # Test that generated Pydantic JSON schema includes user keys
    schema = QA.model_json_schema()
    query_prop = schema["properties"]["query"]
    answer_prop = schema["properties"]["answer"]

    assert query_prop["examples"] == ["What is the capital of France?"]
    assert query_prop["custom_attribute"] == "custom_value"
    assert query_prop["desc"] == "User question"

    assert answer_prop["examples"] == ["Paris"]
    assert answer_prop["desc"] == "Model answer"


def test_input_field_supports_callable_json_schema_extra():
    def custom_schema_mutator(schema: dict):
        schema["x-custom-extension"] = "dynamic_value"

    class DynamicSignature(dspy.Signature):
        text: str = InputField(desc="Input text", json_schema_extra=custom_schema_mutator)
        summary: str = OutputField(desc="Summary output")

    text_extra = DynamicSignature.input_fields["text"].json_schema_extra
    assert text_extra["desc"] == "Input text"
    assert text_extra["__dspy_field_type"] == "input"

    schema = DynamicSignature.model_json_schema()
    text_prop = schema["properties"]["text"]
    assert text_prop["x-custom-extension"] == "dynamic_value"
    assert text_prop["desc"] == "Input text"


def test_field_without_json_schema_extra_unchanged():
    class StandardSig(dspy.Signature):
        q: str = InputField(desc="Question", gt=0)
        a: str = OutputField(desc="Answer")

    q_extra = StandardSig.input_fields["q"].json_schema_extra
    a_extra = StandardSig.output_fields["a"].json_schema_extra

    assert q_extra["desc"] == "Question"
    assert q_extra["__dspy_field_type"] == "input"
    assert "constraints" in q_extra

    assert a_extra["desc"] == "Answer"
    assert a_extra["__dspy_field_type"] == "output"
