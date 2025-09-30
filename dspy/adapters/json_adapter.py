import json
import logging
from typing import Any, get_origin

import json_repair
import litellm
import pydantic
import regex
from pydantic.fields import FieldInfo

from dspy.adapters.chat_adapter import ChatAdapter, FieldInfoWithName
from dspy.adapters.types.tool import ToolCalls
from dspy.adapters.utils import (
    format_field_value,
    get_annotation_name,
    parse_value,
    serialize_for_json,
    translate_field_type,
)
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature, SignatureMeta
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)


def _has_open_ended_mapping(signature: SignatureMeta) -> bool:
    """
    Check whether any output field in the signature has an open-ended mapping type,
    such as dict[str, Any]. Structured Outputs require explicit properties, so such fields
    are incompatible.
    """
    for field in signature.output_fields.values():
        annotation = field.annotation
        if get_origin(annotation) is dict:
            return True
    return False


class JSONAdapter(ChatAdapter):
    def __init__(self, callbacks: list[BaseCallback] | None = None, use_native_function_calling: bool = True):
        # JSONAdapter uses native function calling by default.
        super().__init__(callbacks=callbacks, use_native_function_calling=use_native_function_calling)

    def _json_adapter_call_common(self, lm, lm_kwargs, signature, demos, inputs, call_fn):
        """Common call logic to be used for both sync and async calls."""
        provider = lm.model.split("/", 1)[0] or "openai"
        params = litellm.get_supported_openai_params(model=lm.model, custom_llm_provider=provider)

        if not params or "response_format" not in params:
            return call_fn(lm, lm_kwargs, signature, demos, inputs)

        has_tool_calls = any(field.annotation == ToolCalls for field in signature.output_fields.values())
        # Some models support json mode but not structured outputs
        # Follows guidance from: https://docs.litellm.ai/docs/completion/json_mode#check-model-support
        supports_structured_outputs = litellm.supports_response_schema(model=lm.model, custom_llm_provider=provider)

        if _has_open_ended_mapping(signature) or (not self.use_native_function_calling and has_tool_calls) or not supports_structured_outputs:
            # We found that structured output mode doesn't work well with dspy.ToolCalls as output field.
            # So we fall back to json mode if native function calling is disabled and ToolCalls is present.
            lm_kwargs["response_format"] = {"type": "json_object"}
            return call_fn(lm, lm_kwargs, signature, demos, inputs)

    def __call__(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = self._json_adapter_call_common(lm, lm_kwargs, signature, demos, inputs, super().__call__)
        if result:
            return result

        try:
            structured_output_model = _get_structured_outputs_response_format(
                signature, self.use_native_function_calling
            )
            lm_kwargs["response_format"] = structured_output_model
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)
        except Exception:
            logger.warning("Failed to use structured output format, falling back to JSON mode.")
            lm_kwargs["response_format"] = {"type": "json_object"}
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)

    async def acall(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = self._json_adapter_call_common(lm, lm_kwargs, signature, demos, inputs, super().acall)
        if result:
            return await result

        try:
            structured_output_model = _get_structured_outputs_response_format(signature)
            lm_kwargs["response_format"] = structured_output_model
            return await super().acall(lm, lm_kwargs, signature, demos, inputs)
        except Exception:
            logger.warning("Failed to use structured output format, falling back to JSON mode.")
            lm_kwargs["response_format"] = {"type": "json_object"}
            return await super().acall(lm, lm_kwargs, signature, demos, inputs)

    def format_field_structure(self, signature: type[Signature]) -> str:
        parts = []
        parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

        def format_signature_fields_for_instructions(fields: dict[str, FieldInfo], role: str):
            return self.format_field_with_value(
                fields_with_values={
                    FieldInfoWithName(name=field_name, info=field_info): translate_field_type(field_name, field_info)
                    for field_name, field_info in fields.items()
                },
                role=role,
            )

        parts.append("Inputs will have the following structure:")
        parts.append(format_signature_fields_for_instructions(signature.input_fields, role="user"))
        parts.append("Outputs will be a JSON object with the following fields.")
        parts.append(format_signature_fields_for_instructions(signature.output_fields, role="assistant"))
        return "\n\n".join(parts).strip()

    def user_message_output_requirements(self, signature: type[Signature]) -> str:
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
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message=None,
    ) -> str:
        fields_with_values = {
            FieldInfoWithName(name=k, info=v): outputs.get(k, missing_field_message)
            for k, v in signature.output_fields.items()
        }
        return self.format_field_with_value(fields_with_values, role="assistant")

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        pattern = r"\{(?:[^{}]|(?R))*\}"
        match = regex.search(pattern, completion, regex.DOTALL)
        if match:
            completion = match.group(0)
        fields = json_repair.loads(completion)

        if not isinstance(fields, dict):
            raise AdapterParseError(
                adapter_name="JSONAdapter",
                signature=signature,
                lm_response=completion,
                message="LM response cannot be serialized to a JSON object.",
            )

        fields = {k: v for k, v in fields.items() if k in signature.output_fields}

        # Attempt to cast each value to type signature.output_fields[k].annotation.
        for k, v in fields.items():
            if k in signature.output_fields:
                fields[k] = parse_value(v, signature.output_fields[k].annotation)

        if fields.keys() != signature.output_fields.keys():
            raise AdapterParseError(
                adapter_name="JSONAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=fields,
            )

        return fields

    def format_field_with_value(self, fields_with_values: dict[FieldInfoWithName, Any], role: str = "user") -> str:
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
        self, signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any], outputs: dict[str, Any]
    ) -> dict[str, list[Any]]:
        # TODO: implement format_finetune_data method in JSONAdapter
        raise NotImplementedError


def _get_structured_outputs_response_format(
    signature: SignatureMeta,
    use_native_function_calling: bool = True,
) -> type[pydantic.BaseModel]:
    """
    Builds a Pydantic model from a DSPy signature's output_fields and ensures the generated JSON schema
    is compatible with OpenAI Structured Outputs (all objects have a "required" key listing every property,
    and additionalProperties is always false).

    IMPORTANT: If any field's annotation is an open-ended mapping (e.g. dict[str, Any]), then a structured
    schema cannot be generated since all properties must be explicitly declared. In that case, an exception
    is raised so that the caller can fall back to using a plain "json_object" response_format.
    """
    # Although we've already performed an early check, we keep this here as a final guard.
    for name, field in signature.output_fields.items():
        annotation = field.annotation
        if get_origin(annotation) is dict:
            raise ValueError(
                f"Field '{name}' has an open-ended mapping type which is not supported by Structured Outputs."
            )

    fields = {}
    for name, field in signature.output_fields.items():
        annotation = field.annotation
        if use_native_function_calling and annotation == ToolCalls:
            # Skip ToolCalls field if native function calling is enabled.
            continue
        default = field.default if hasattr(field, "default") else ...
        fields[name] = (annotation, default)

    # Build the model with extra fields forbidden.
    pydantic_model = pydantic.create_model(
        "DSPyProgramOutputs",
        __config__=pydantic.ConfigDict(extra="forbid"),
        **fields,
    )

    # Generate the initial schema.
    schema = pydantic_model.model_json_schema()

    # Remove any DSPy-specific metadata.
    for prop in schema.get("properties", {}).values():
        prop.pop("json_schema_extra", None)

    def enforce_required(schema_part: dict):
        """
        Recursively ensure that:
            - for any object schema, a "required" key is added with all property names (or [] if no properties)
            - additionalProperties is set to False regardless of the previous value.
            - the same enforcement is run for nested arrays and definitions.
        """
        if schema_part.get("type") == "object":
            props = schema_part.get("properties")
            if props is not None:
                # For objects with explicitly declared properties:
                schema_part["required"] = list(props.keys())
                schema_part["additionalProperties"] = False
                for sub_schema in props.values():
                    if isinstance(sub_schema, dict):
                        enforce_required(sub_schema)
            else:
                # For objects with no properties (should not happen normally but a fallback).
                schema_part["properties"] = {}
                schema_part["required"] = []
                schema_part["additionalProperties"] = False
        if schema_part.get("type") == "array" and isinstance(schema_part.get("items"), dict):
            enforce_required(schema_part["items"])
        # Also enforce in any nested definitions / $defs.
        for key in ("$defs", "definitions"):
            if key in schema_part:
                for def_schema in schema_part[key].values():
                    enforce_required(def_schema)

    enforce_required(schema)

    # Override the model's JSON schema generation to return our precomputed schema.
    pydantic_model.model_json_schema = lambda *args, **kwargs: schema

    return pydantic_model
