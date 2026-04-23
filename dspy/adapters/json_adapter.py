import json
import logging
from collections.abc import Mapping
from typing import Any, Generator, Literal, get_args, get_origin

import json_repair
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
from dspy.clients.base_lm import BaseLM
from dspy.signatures.signature import Signature, SignatureMeta
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)


def _open_ended_mapping_field_names(signature: SignatureMeta) -> Generator[str, None, None]:
    """Yield names of output fields with open-ended mapping annotations.

    "Open-ended" here means the key-set is unbounded at the type level: an annotation
    like ``dict[str, Any]`` declares "some string keys to values" without naming which
    keys. A strict JSON Schema requires every property enumerated (``properties`` +
    ``required`` + ``additionalProperties: false``) -- impossible without a known
    key-set -- so these fields force the prompted fallback.

    Covers any ``collections.abc.Mapping`` subclass: ``dict``, ``Mapping``,
    ``MutableMapping``, ``OrderedDict``, ``defaultdict``, ``Counter``, ``ChainMap``,
    and any user class inheriting from ``Mapping``. Correctly excludes ``TypedDict``
    subclasses -- they declare a fixed named key-set at the type level and *are*
    strict-schema-expressible; Python treats them as ``dict``-at-runtime but not as
    ``Mapping`` subclasses, so ``issubclass`` returns False.
    """
    for name, field in signature.output_fields.items():
        origin = get_origin(field.annotation) or field.annotation
        if isinstance(origin, type) and issubclass(origin, Mapping):
            yield name


def _non_native_tool_call_field_names(signature: SignatureMeta) -> Generator[str, None, None]:
    """Yield names of output fields with ToolCalls annotations."""
    for name, field in signature.output_fields.items():
        if field.annotation == ToolCalls:
            yield name


SchemaEnforcement = Literal["auto", "strict", "prompted"]
"""Which ``response_format`` payload ``JSONAdapter`` ships.

``"auto"`` delegates to ``lm.supports_response_schema`` unless the signature has open-ended mapping fields or non-native tool calls.
``"strict"`` forces a Pydantic class (serialized to strict ``json_schema``) -- use for known openai-compatible endpoints.
``"prompted"`` forces ``{"type": "json_object"}`` with schema hints in the prompt.
"""


class JSONAdapter(ChatAdapter):
    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = True,
        schema_enforcement_mode: SchemaEnforcement = "auto",
    ):
        super().__init__(callbacks=callbacks, use_native_function_calling=use_native_function_calling)
        assert schema_enforcement_mode in get_args(
            SchemaEnforcement
        ), f"schema_enforcement_mode must be one of {SchemaEnforcement}; got {schema_enforcement_mode!r}"
        self.schema_enforcement_mode: SchemaEnforcement = schema_enforcement_mode

    def _json_adapter_call_common(self, lm, lm_kwargs, signature, demos, inputs, call_fn):
        """Common call logic to be used for both sync and async calls."""
        if "response_format" not in lm.supported_params:
            return call_fn(lm, lm_kwargs, signature, demos, inputs)

        if self._effective_schema_enforcement_mode(lm, signature) == "prompted":
            lm_kwargs["response_format"] = {"type": "json_object"}
            return call_fn(lm, lm_kwargs, signature, demos, inputs)

    def _effective_schema_enforcement_mode(self, lm: BaseLM, signature: SignatureMeta) -> SchemaEnforcement:
        open_ended_mapping_fields = _open_ended_mapping_field_names(signature)
        has_non_native_tool_calls = not self.use_native_function_calling and any(
            _non_native_tool_call_field_names(signature)
        )
        if self.schema_enforcement_mode == "auto":
            # open-ended dicts aren't JSON-Schema-expressible; ToolCalls + non-native fcall misbehaves (see warnings below).
            if has_non_native_tool_calls or any(open_ended_mapping_fields):
                return "prompted"
            return "strict" if lm.supports_response_schema else "prompted"
        if self.schema_enforcement_mode == "strict":
            if any(open_ended_mapping_fields):
                names = ", ".join(_open_ended_mapping_field_names(signature))
                logger.warning(
                    f"Signature {signature.__name__} has open-ended output mapping field(s) {names}; most structured output inference engines"
                    f" cannot express these, so the {self.__class__.__name__} will fall back to prompted mode. "
                    f"Either consider using different Signature types or use {self.__class__.__name__}.schema_enforcement_mode='prompted'.",
                )
            if has_non_native_tool_calls:
                names = ", ".join(_non_native_tool_call_field_names(signature))
                logger.warning(
                    f"Signature {signature.__name__} has dspy.ToolCalls output field(s) {names} but {self.__class__.__name__}.use_native_function_calling=False and {self.__class__.__name__}.schema_enforcement_mode='strict'; "
                    "strict schema mode is known to compose poorly with JSON-mode tool calling and may produce "
                    f"unreliable output. Consider {self.__class__.__name__}.use_native_function_calling=True or "
                    f"{self.__class__.__name__}.schema_enforcement_mode='prompted' for more reliable behavior from {signature.__name__}.",
                )
        return self.schema_enforcement_mode

    def __call__(
        self,
        lm: BaseLM,
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
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = self._json_adapter_call_common(lm, lm_kwargs, signature, demos, inputs, super().acall)
        if result:
            return await result

        try:
            structured_output_model = _get_structured_outputs_response_format(
                signature, self.use_native_function_calling
            )
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
        fields = json_repair.loads(completion)

        if not isinstance(fields, dict):
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
            return json.dumps(serialize_for_json(d), indent=2, ensure_ascii=False)

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
