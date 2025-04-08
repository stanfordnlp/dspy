import json
import logging
import litellm
import pydantic
import json_repair

from typing import Any, Dict, Type
from pydantic.fields import FieldInfo

from dspy.clients.lm import LM
from dspy.adapters.utils import (
    format_field_value,
    get_annotation_name,
    parse_value,
    serialize_for_json,
    translate_field_type,
)
from dspy.signatures.signature import Signature, SignatureMeta
from dspy.adapters.chat_adapter import ChatAdapter, FieldInfoWithName

logger = logging.getLogger(__name__)


class JSONAdapter(ChatAdapter):
    def __call__(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: Type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        provider = lm.model.split("/", 1)[0] or "openai"
        params = litellm.get_supported_openai_params(model=lm.model, custom_llm_provider=provider)

        # If response_format is not supported, use basic call
        if not params or "response_format" not in params:
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)

        # Try structured output first, fall back to basic json if it fails
        try:
            structured_output_model = _get_structured_outputs_response_format(signature)
            lm_kwargs["response_format"] = structured_output_model
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)
        except Exception as e:
            logger.warning(f"Failed to use structured output format. Falling back to JSON mode. Error: {e}")
            try:
                lm_kwargs["response_format"] = {"type": "json_object"}
                return super().__call__(lm, lm_kwargs, signature, demos, inputs)
            except Exception as e:
                raise RuntimeError(
                    "Both structured output format and JSON mode failed. Please choose a model that supports "
                    f"`response_format` argument. Original error: {e}"
                ) from e

    def format_field_structure(self, signature: Type[Signature]) -> str:
        parts = []
        parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

        def format_signature_fields_for_instructions(fields: Dict[str, FieldInfo]):
            return self.format_field_with_value(
                fields_with_values={
                    FieldInfoWithName(name=field_name, info=field_info): translate_field_type(field_name, field_info)
                    for field_name, field_info in fields.items()
                },
            )

        parts.append("Inputs will have the following structure:")
        parts.append(format_signature_fields_for_instructions(signature.input_fields))
        parts.append("Outputs will be a JSON object with the following fields.")
        parts.append(format_signature_fields_for_instructions(signature.output_fields))
        return "\n\n".join(parts).strip()

    def user_message_output_requirements(self, signature: Type[Signature]) -> str:
        def type_info(v):
            return (
                f" (must be formatted as a valid Python {get_annotation_name(v.annotation)})"
                if v.annotation is not str
                else ""
            )

        message = "Respond with a JSON object in the following order of fields: "
        message += ", then ".join(f"`{f}`{type_info(v)}" for f, v in signature.output_fields.items())
        message += "."
        return message

    def format_assistant_message_content(
        self,
        signature: Type[Signature],
        outputs: dict[str, Any],
        missing_field_message=None,
    ) -> str:
        fields_with_values = {
            FieldInfoWithName(name=k, info=v): outputs.get(k, missing_field_message)
            for k, v in signature.output_fields.items()
        }
        return self.format_field_with_value(fields_with_values, role="assistant")

    def parse(self, signature: Type[Signature], completion: str) -> dict[str, Any]:
        fields = json_repair.loads(completion)
        fields = {k: v for k, v in fields.items() if k in signature.output_fields}

        # Attempt to cast each value to type signature.output_fields[k].annotation.
        for k, v in fields.items():
            if k in signature.output_fields:
                fields[k] = parse_value(v, signature.output_fields[k].annotation)

        if fields.keys() != signature.output_fields.keys():
            raise ValueError(f"Expected {signature.output_fields.keys()} but got {fields.keys()}")

        return fields

    def format_field_with_value(self, fields_with_values: Dict[FieldInfoWithName, Any], role: str = "user") -> str:
        """
        Formats the values of the specified fields according to the field's DSPy type (input or output),
        annotation (e.g. str, int, etc.), and the type of the value itself. Joins the formatted values
        into a single string, which is a multiline string if there are multiple fields.

        Args:
            fields_with_values: A dictionary mapping information about a field to its corresponding value.
        Returns:
            The joined formatted values of the fields, represented as a string.
        """
        if role == "user":
            output = []
            for field, field_value in fields_with_values.items():
                formatted_field_value = format_field_value(field_info=field.info, value=field_value)
                output.append(f"[[ ## {field.name} ## ]]\n{formatted_field_value}")
            return "\n\n".join(output).strip()
        else:
            d = fields_with_values.items()
            d = {k.name: v for k, v in d}
            return json.dumps(serialize_for_json(d), indent=2)

    def format_finetune_data(
        self, signature: Type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any], outputs: dict[str, Any]
    ) -> dict[str, list[Any]]:
        # TODO: implement format_finetune_data method in JSONAdapter
        raise NotImplementedError


def _get_structured_outputs_response_format(signature: SignatureMeta) -> type[pydantic.BaseModel]:
    """
    Builds a Pydantic model from a DSPy signature's output_fields,
    ensuring that internal DSPy metadata is not leaked in the model's schema.
    """
    fields = {
        name: (field.annotation, field.default if hasattr(field, "default") else ...)
        for name, field in signature.output_fields.items()
    }
    Model = pydantic.create_model(
        "DSPyProgramOutputs",
        **fields,
        __config__=type("Config", (), {"extra": "forbid"})
    )
    
    # Remove any internal DSPy metadata from each model field.
    for field in Model.__fields__.values():
        field.field_info.json_schema_extra = {}
    
    # Build the schema and ensure it doesn't include DSPy metadata.
    schema = Model.schema()
    
    # Remove DSPy-specific metadata (e.g. "json_schema_extra") from each property's schema.
    for prop in schema.get("properties", {}).values():
        prop.pop("json_schema_extra", None)
    
    schema["required"] = list(schema.get("properties", {}).keys())
    schema["additionalProperties"] = False

    # Override model_json_schema to return our precomputed schema (avoiding recursion).
    Model.model_json_schema = lambda *args, **kwargs: schema
    return Model
