import enum
import inspect
import json
import logging
import textwrap
from copy import deepcopy
from typing import Any, Dict, KeysView, Literal, NamedTuple, Type

import json_repair
import litellm
import pydantic
from pydantic import create_model
from pydantic.fields import FieldInfo

from dspy.adapters.base import Adapter
from dspy.adapters.types.history import History
from dspy.adapters.types.image import try_expand_image_tags
from dspy.adapters.utils import format_field_value, get_annotation_name, parse_value, serialize_for_json
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature, SignatureMeta
from dspy.signatures.utils import get_dspy_field_type

logger = logging.getLogger(__name__)


class FieldInfoWithName(NamedTuple):
    name: str
    info: FieldInfo


class JSONAdapter(Adapter):
    def __init__(self):
        pass

    def __call__(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: Type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        inputs = self.format(signature, demos, inputs)
        inputs = dict(prompt=inputs) if isinstance(inputs, str) else dict(messages=inputs)

        try:
            provider = lm.model.split("/", 1)[0] or "openai"
            params = litellm.get_supported_openai_params(model=lm.model, custom_llm_provider=provider)
            if params and "response_format" in params:
                try:
                    response_format = _get_structured_outputs_response_format(signature)
                    outputs = lm(**inputs, **lm_kwargs, response_format=response_format)
                except Exception as e:
                    logger.debug(
                        f"Failed to obtain response using signature-based structured outputs"
                        f" response format: Falling back to default 'json_object' response format."
                        f" Exception: {e}"
                    )
                    outputs = lm(**inputs, **lm_kwargs, response_format={"type": "json_object"})
            else:
                outputs = lm(**inputs, **lm_kwargs)

        except litellm.UnsupportedParamsError:
            outputs = lm(**inputs, **lm_kwargs)

        values = []

        for output in outputs:
            value = self.parse(signature, output)
            assert set(value.keys()) == set(
                signature.output_fields.keys()
            ), f"Expected {signature.output_fields.keys()} but got {value.keys()}"
            values.append(value)

        return values

    def format(
        self, signature: Type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]
    ) -> list[dict[str, Any]]:
        messages = []

        # Extract demos where some of the output_fields are not filled in.
        incomplete_demos = [demo for demo in demos if not all(k in demo for k in signature.fields)]
        complete_demos = [demo for demo in demos if demo not in incomplete_demos]
        incomplete_demos = [
            demo
            for demo in incomplete_demos
            if any(k in demo for k in signature.input_fields) and any(k in demo for k in signature.output_fields)
        ]

        demos = incomplete_demos + complete_demos

        messages.append({"role": "system", "content": prepare_instructions(signature)})

        for demo in demos:
            messages.append(self.format_turn(signature, demo, role="user", incomplete=demo in incomplete_demos))
            messages.append(self.format_turn(signature, demo, role="assistant", incomplete=demo in incomplete_demos))

        # Add the chat history after few-shot examples
        if any(field.annotation == History for field in signature.input_fields.values()):
            messages.extend(self.format_conversation_history(signature, inputs))
        else:
            messages.append(self.format_turn(signature, inputs, role="user"))

        messages = try_expand_image_tags(messages)
        return messages

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

    def format_fields(self, signature: Type[Signature], values: dict[str, Any], role: str) -> str:
        fields_with_values = {
            FieldInfoWithName(name=field_name, info=field_info): values.get(
                field_name, "Not supplied for this particular example."
            )
            for field_name, field_info in signature.fields.items()
            if field_name in values
        }
        return format_fields(role=role, fields_with_values=fields_with_values)

    def format_turn(
        self,
        signature: Type[Signature],
        values,
        role: str,
        incomplete: bool = False,
        is_conversation_history: bool = False,
    ) -> dict[str, Any]:
        return format_turn(signature, values, role, incomplete, is_conversation_history)

    def format_finetune_data(
        self, signature: Type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any], outputs: dict[str, Any]
    ) -> dict[str, list[Any]]:
        # TODO: implement format_finetune_data method in JSONAdapter
        raise NotImplementedError


def format_fields(role: str, fields_with_values: Dict[FieldInfoWithName, Any]) -> str:
    """
    Formats the values of the specified fields according to the field's DSPy type (input or output),
    annotation (e.g. str, int, etc.), and the type of the value itself. Joins the formatted values
    into a single string, which is a multiline string if there are multiple fields.

    Args:
      role: The role of the message ('user' or 'assistant')
      fields_with_values: A dictionary mapping information about a field to its corresponding value.

    Returns:
      The joined formatted values of the fields, represented as a string.
    """

    if role == "assistant":
        d = fields_with_values.items()
        d = {k.name: v for k, v in d}
        return json.dumps(serialize_for_json(d), indent=2)

    output = []
    for field, field_value in fields_with_values.items():
        formatted_field_value = format_field_value(field_info=field.info, value=field_value)
        output.append(f"[[ ## {field.name} ## ]]\n{formatted_field_value}")

    return "\n\n".join(output).strip()


def format_turn(
    signature: SignatureMeta,
    values: Dict[str, Any],
    role: str,
    incomplete=False,
    is_conversation_history=False,
) -> Dict[str, str]:
    """
    Constructs a new message ("turn") to append to a chat thread. The message is carefully formatted
    so that it can instruct an LLM to generate responses conforming to the specified DSPy signature.

    Args:
        signature: The DSPy signature to which future LLM responses should conform.
        values: A dictionary mapping field names (from the DSPy signature) to corresponding values
            that should be included in the message.
        role: The role of the message, which can be either "user" or "assistant".
        incomplete: If True, indicates that output field values are present in the set of specified
            `values`. If False, indicates that `values` only contains input field values. Only
            relevant if `is_conversation_history` is False.
        is_conversation_history: If True, indicates that the message is part of a chat history instead of a
            few-shot example.
    Returns:
        A chat message that can be appended to a chat thread. The message contains two string fields:
        ``role`` ("user" or "assistant") and ``content`` (the message text).
    """
    content = []

    if role == "user":
        fields: Dict[str, FieldInfo] = signature.input_fields
        if incomplete and not is_conversation_history:
            content.append("This is an example of the task, though some input or output fields are not supplied.")
    else:
        fields: Dict[str, FieldInfo] = signature.output_fields

    if not incomplete and not is_conversation_history:
        # For complete few-shot examples, ensure that the values contain all the fields.
        field_names: KeysView = fields.keys()
        if not set(values).issuperset(set(field_names)):
            raise ValueError(f"Expected {field_names} but got {values.keys()}")

    fields_with_values = {}
    for field_name, field_info in fields.items():
        if is_conversation_history:
            fields_with_values[FieldInfoWithName(name=field_name, info=field_info)] = values.get(
                field_name, "Not supplied for this conversation history message. "
            )
        else:
            fields_with_values[FieldInfoWithName(name=field_name, info=field_info)] = values.get(
                field_name, "Not supplied for this particular example. "
            )

    formatted_fields = format_fields(role=role, fields_with_values=fields_with_values)
    content.append(formatted_fields)

    if role == "user":

        def type_info(v):
            return (
                f" (must be formatted as a valid Python {get_annotation_name(v.annotation)})"
                if v.annotation is not str
                else ""
            )

        # TODO: Consider if not incomplete:
        content.append(
            "Respond with a JSON object in the following order of fields: "
            + ", then ".join(f"`{f}`{type_info(v)}" for f, v in signature.output_fields.items())
            + "."
        )

    return {"role": role, "content": "\n\n".join(content).strip()}


def enumerate_fields(fields):
    parts = []
    for idx, (k, v) in enumerate(fields.items()):
        parts.append(f"{idx+1}. `{k}`")
        parts[-1] += f" ({get_annotation_name(v.annotation)})"
        parts[-1] += f": {v.json_schema_extra['desc']}" if v.json_schema_extra["desc"] != f"${{{k}}}" else ""
        parts[-1] += (
            f"\nConstraints: {v.json_schema_extra['constraints']}" if v.json_schema_extra.get("constraints") else ""
        )

    return "\n".join(parts).strip()


def prepare_instructions(signature: SignatureMeta):
    parts = []
    parts.append("Your input fields are:\n" + enumerate_fields(signature.input_fields))
    parts.append("Your output fields are:\n" + enumerate_fields(signature.output_fields))
    parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

    def field_metadata(field_name, field_info):
        type_ = field_info.annotation

        if get_dspy_field_type(field_info) == "input" or type_ is str:
            desc = ""
        elif type_ is bool:
            desc = "must be True or False"
        elif type_ in (int, float):
            desc = f"must be a single {type_.__name__} value"
        elif inspect.isclass(type_) and issubclass(type_, enum.Enum):
            desc = f"must be one of: {'; '.join(type_.__members__)}"
        elif hasattr(type_, "__origin__") and type_.__origin__ is Literal:
            desc = f"must be one of: {'; '.join([str(x) for x in type_.__args__])}"
        else:
            desc = "must adhere to the JSON schema: "
            desc += json.dumps(pydantic.TypeAdapter(type_).json_schema())

        desc = (" " * 8) + f"# note: the value you produce {desc}" if desc else ""
        return f"{{{field_name}}}{desc}"

    def format_signature_fields_for_instructions(role, fields: Dict[str, FieldInfo]):
        return format_fields(
            role=role,
            fields_with_values={
                FieldInfoWithName(name=field_name, info=field_info): field_metadata(field_name, field_info)
                for field_name, field_info in fields.items()
            },
        )

    parts.append("Inputs will have the following structure:")
    parts.append(format_signature_fields_for_instructions("user", signature.input_fields))
    parts.append("Outputs will be a JSON object with the following fields.")
    parts.append(format_signature_fields_for_instructions("assistant", signature.output_fields))
    # parts.append(format_fields({BuiltInCompletedOutputFieldInfo: ""}))

    instructions = textwrap.dedent(signature.instructions)
    objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
    parts.append(f"In adhering to this structure, your objective is: {objective}")

    # parts.append("You will receive some input fields in each interaction. " +
    #              "Respond only with the corresponding output fields, starting with the field " +
    #              ", then ".join(f"`{f}`" for f in signature.output_fields) +
    #              ", and then ending with the marker for `completed`.")

    return "\n\n".join(parts).strip()


def _get_structured_outputs_response_format(signature: SignatureMeta) -> pydantic.BaseModel:
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
        key: (value.annotation, filter_json_schema_extra(key, value)) for key, value in signature.output_fields.items()
    }
    return create_model("DSPyProgramOutputs", **output_pydantic_fields)
