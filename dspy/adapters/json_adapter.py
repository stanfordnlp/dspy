import json
import logging
from copy import deepcopy
from typing import Any, Dict, Type

import json_repair
import litellm
import pydantic
from pydantic import create_model
from pydantic.fields import FieldInfo

from dspy.adapters.chat_adapter import ChatAdapter, FieldInfoWithName
from dspy.adapters.utils import (
    format_field_value,
    get_annotation_name,
    parse_value,
    serialize_for_json,
    translate_field_type,
)
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature, SignatureMeta

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
            structured_output_format = self._get_structured_outputs_response_format(signature)
            lm_kwargs["response_format"] = structured_output_format
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

        # attempt to cast each value to type signature.output_fields[k].annotation
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
        into a single string, which is is a multiline string if there are multiple fields.

        Args:
        fields_with_values: A dictionary mapping information about a field to its corresponding value.
        Returns:
            The joined formatted values of the fields, represented as a string
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

    def _get_structured_outputs_response_format(self, signature: SignatureMeta) -> pydantic.BaseModel:
        """
        Obtains the LiteLLM / OpenAI `response_format` parameter for generating structured outputs from
        an LM request, based on the output fields of the specified DSPy signature.

        Args:
            signature: The DSPy signature for which to obtain the `response_format` request parameter.
        Returns:
            A Pydantic model representing the `response_format` parameter for the LM request.
        """

        def filter_json_schema_extra(field_name: str, field_info: FieldInfo) -> FieldInfo:
            """
            Recursively filter the `json_schema_extra` of a FieldInfo to exclude DSPy internal attributes
            (e.g. `__dspy_field_type`) and remove descriptions that are placeholders for the field name.
            """
            field_copy = deepcopy(field_info)  # Make a copy to avoid mutating the original

            # Update `json_schema_extra` for the copied field
            if field_copy.json_schema_extra:
                field_copy.json_schema_extra = {
                    key: value
                    for key, value in field_info.json_schema_extra.items()
                    if key not in ("desc", "__dspy_field_type")
                }
                field_desc = field_info.json_schema_extra.get("desc")
                if field_desc is not None and field_desc != f"${{{field_name}}}":
                    field_copy.json_schema_extra["desc"] = field_desc

            # Handle nested fields
            if hasattr(field_copy.annotation, "__pydantic_model__"):
                # Recursively update fields of the nested model
                nested_model = field_copy.annotation.__pydantic_model__
                updated_fields = {
                    key: filter_json_schema_extra(key, value) for key, value in nested_model.__fields__.items()
                }
                # Create a new model with the same name and updated fields
                field_copy.annotation = create_model(nested_model.__name__, **updated_fields)

            return field_copy

        output_pydantic_fields = {
            key: (value.annotation, filter_json_schema_extra(key, value))
            for key, value in signature.output_fields.items()
        }
        return create_model("DSPyProgramOutputs", **output_pydantic_fields)
