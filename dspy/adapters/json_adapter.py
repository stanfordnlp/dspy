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


EffectiveEnforcement = Literal["json_schema", "json_object"]

SchemaEnforcement = Literal["auto"] | EffectiveEnforcement
"""Which ``response_format`` payload ``JSONAdapter`` ships. Values name the OpenAI-compatible ``response_format.type`` that reaches the wire.

``"auto"`` delegates to ``lm.supports_response_schema`` unless the signature has open-ended mapping fields or non-native tool calls.
``"json_schema"`` forces the Pydantic-class response_format, shipped as ``{"type": "json_schema", "strict": true, "schema": {...}}``; use for known openai-compatible endpoints that can use structured outputs.
``"json_object"`` forces ``{"type": "json_object"}`` with schema hints carried in the prompt; escape hatch for LM-known models whose strict-schema backend is broken.
"""


class JSONAdapter(ChatAdapter):
    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = True,
        schema_enforcement_mode: SchemaEnforcement = "auto",
    ):
        super().__init__(callbacks=callbacks, use_native_function_calling=use_native_function_calling)
        # typing.get_args doesn't flatten Literal | Literal unions, so build the flat tuple explicitly.
        valid_modes = ("auto", *get_args(EffectiveEnforcement))
        assert (
            schema_enforcement_mode in valid_modes
        ), f"schema_enforcement_mode must be one of {valid_modes}; got {schema_enforcement_mode!r}"
        self.schema_enforcement_mode: SchemaEnforcement = schema_enforcement_mode

    def response_format(self, lm: BaseLM, signature: SignatureMeta) -> type[pydantic.BaseModel] | dict[str, str] | None:
        """Returns the ``response_format`` value to set on this call, or ``None`` if no response_format is needed."""
        if self.schema_enforcement_mode == "auto" and "response_format" not in lm.supported_params:
            return None
        match self._effective_enforcement_mode(lm, signature):
            case "json_object":
                return {"type": "json_object"}
            case "json_schema":
                try:
                    return _get_structured_outputs_response_format(signature, self.use_native_function_calling)
                except Exception as e:
                    logger.warning(f"Failed to build structured output format, falling back to JSON mode: {e}")
                    return {"type": "json_object"}
            case unknown:
                raise ValueError(f"Invalid schema enforcement mode: {unknown}")

    def _effective_enforcement_mode(self, lm: BaseLM, signature: SignatureMeta) -> EffectiveEnforcement:
        open_ended_mapping_fields = _open_ended_mapping_field_names(signature)
        has_non_native_tool_calls = not self.use_native_function_calling and any(
            _non_native_tool_call_field_names(signature)
        )
        if self.schema_enforcement_mode == "auto":
            # open-ended dicts aren't JSON-Schema-expressible; ToolCalls + non-native fcall misbehaves (see warnings below).
            if has_non_native_tool_calls or any(open_ended_mapping_fields):
                return "json_object"
            return "json_schema" if lm.supports_response_schema else "json_object"
        if self.schema_enforcement_mode == "json_schema":
            if any(open_ended_mapping_fields):
                names = ", ".join(_open_ended_mapping_field_names(signature))
                logger.warning(
                    f"Signature {signature.__name__} has open-ended output mapping field(s) {names}; most structured output "
                    f"inference engines cannot express these. Consider replacing the open-ended mapping with a typed schema "
                    f"(e.g., a pydantic.BaseModel with explicit fields, or typing.TypedDict) for more reliable behavior from {signature.__name__} when using {self.__class__.__name__}.",
                )
            if has_non_native_tool_calls:
                names = ", ".join(_non_native_tool_call_field_names(signature))
                logger.warning(
                    f"Signature {signature.__name__} has dspy.ToolCalls output field(s) {names} but {self.__class__.__name__}.use_native_function_calling=False and {self.__class__.__name__}.schema_enforcement_mode='json_schema'; "
                    "json_schema mode is known to compose poorly with JSON-mode tool calling and may produce unreliable output. "
                    f" Consider {self.__class__.__name__}.use_native_function_calling=True or "
                    f"{self.__class__.__name__}.schema_enforcement_mode='json_object' for more reliable behavior from {signature.__name__} when using {self.__class__.__name__}.",
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
        rf = self.response_format(lm, signature)
        if rf is not None:
            lm_kwargs["response_format"] = rf
        try:
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)
        except Exception as e:
            if not isinstance(rf, type):
                raise
            logger.error(f"Structured output call failed; automatically retrying with JSON mode. Reason: {e}")
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
        rf = self.response_format(lm, signature)
        if rf is not None:
            lm_kwargs["response_format"] = rf
        try:
            return await super().acall(lm, lm_kwargs, signature, demos, inputs)
        except Exception as e:
            if not isinstance(rf, type):
                raise
            logger.error(f"Structured output call failed; automatically retrying with JSON mode. Reason: {e}")
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
