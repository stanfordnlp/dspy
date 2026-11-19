import pytest

import dspy
from dspy.signatures.field import InputField, OutputField, _SchemaExtraCallable


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


def test_callable_json_schema_extra_mapping_and_updates():
    def custom_mutator(schema: dict):
        schema["x-custom-extension"] = "dynamic_value"

    class DynamicSignature(dspy.Signature):
        text: str = InputField(desc="Input text", json_schema_extra=custom_mutator)
        summary: str = OutputField(desc="Summary output", prefix="Out:")

    field = DynamicSignature.input_fields["text"]
    extra = field.json_schema_extra

    # Test Mapping interface (keys, items, values, contains, unpack)
    assert "desc" in extra
    assert list(extra.keys()) == ["desc", "__dspy_field_type", "prefix"]
    assert ("desc", "Input text") in extra.items()
    assert "Input text" in list(extra.values())
    unpacked = {**extra, "extra_key": "val"}
    assert unpacked["desc"] == "Input text"
    assert unpacked["extra_key"] == "val"

    # Test compatibility with sandbox_guest signature_spec
    from dspy.primitives.sandbox_guest import signature_spec

    spec = signature_spec(DynamicSignature)
    assert spec["extra"]["text"]["desc"] == "Input text"

    # Test with_updated_fields preserves callback and applies updates
    updated_sig = DynamicSignature.with_updated_fields("text", desc="Updated input text", prefix="CustomPrefix:")
    assert updated_sig.input_fields["text"].json_schema_extra["desc"] == "Updated input text"
    assert updated_sig.input_fields["text"].json_schema_extra["prefix"] == "CustomPrefix:"

    schema = updated_sig.model_json_schema()
    text_prop = schema["properties"]["text"]
    assert text_prop["desc"] == "Updated input text"
    assert text_prop["prefix"] == "CustomPrefix:"
    assert text_prop["x-custom-extension"] == "dynamic_value"


def test_callable_json_schema_extra_does_not_retry_callback_errors():
    calls = 0

    def failing_mutator(schema: dict):
        nonlocal calls
        calls += 1
        raise TypeError("callback failure")

    extra = _SchemaExtraCallable({}, failing_mutator)

    with pytest.raises(TypeError, match="callback failure"):
        extra({})

    assert calls == 1


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
