"""Signature formatting for the native lm15 TypeSafe System One route."""

import json

from pydantic import TypeAdapter

from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types.base_type import split_message_content_for_custom_types
from dspy.adapters.utils import get_field_description_string, serialize_for_json


class TypeSafeAdapter(JSONAdapter):
    """Use ``dspy.LM('typesafe:jev-1.14.0', engine='lm15')`` with Predict.

    Jev judges a single state rather than a chat conversation. Instructions,
    demonstrations and named inputs become that state; output descriptions
    become native questions. Use top-level BoundingBox, bool and Literal fields
    (string choices or consecutive integer levels starting at zero). Other
    output schemas, including lists and nullable boxes, are refused by lm15
    before transport. Use JSONAdapter for generative providers instead.

    Images must contain PNG/JPEG bytes (Image.from_path/from_url), not remote
    URLs. lm15 sends TypeSafe's experimental images opt-in header. Confidence
    and measured distributions are retained in the canonical LM response in
    ``lm.history[-1]['response']``; they are not generated prediction fields.
    """

    def _call_preprocess(self, lm, lm_kwargs, signature, inputs):
        if "response_format" in lm_kwargs:
            raise ValueError("TypeSafeAdapter derives response_format from the signature.")
        properties = {}
        for name, field in signature.output_fields.items():
            # Generate each field's schema independently so a top-level box is
            # an inline object, not a $ref the native judgment reader cannot see.
            schema = TypeAdapter(field.rebuild_annotation()).json_schema()
            description = field.json_schema_extra.get("desc", "")
            schema["description"] = name if description == f"${{{name}}}" else description or name
            properties[name] = schema
        lm_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "DSPyJudgments", "schema": {
                "type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False,
            }},
        }
        return signature

    def format(self, signature, demos, inputs):
        state = {
            "instructions": signature.instructions,
            "input_fields": get_field_description_string(signature.input_fields),
            "inputs": {name: value for name, value in inputs.items() if name in signature.input_fields},
        }
        if demos:
            state["demos"] = [{name: value for name, value in demo.items() if name in signature.fields} for demo in demos]
        # Reuse DSPy's media serialization: image markers become actual image
        # parts, including images nested in containers and demonstrations.
        messages = [{"role": "user", "content": json.dumps(serialize_for_json(state), ensure_ascii=False)}]
        return split_message_content_for_custom_types(messages)
