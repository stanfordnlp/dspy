import ast
import enum
import inspect
import json
import textwrap
from typing import Any, Dict, KeysView, Literal, NamedTuple, get_args, get_origin

import json_repair
import litellm
import pydantic
from pydantic import TypeAdapter
from pydantic.fields import FieldInfo

from dspy.adapters.base import Adapter
from dspy.adapters.utils import find_enum_member, format_field_value, serialize_for_json

from ..adapters.image_utils import Image
from ..signatures.signature import SignatureMeta
from ..signatures.utils import get_dspy_field_type


class FieldInfoWithName(NamedTuple):
    name: str
    info: FieldInfo


class JSONAdapter(Adapter):
    def __init__(self):
        pass

    def __call__(self, lm, lm_kwargs, signature, demos, inputs, _parse_values=True):
        inputs = self.format(signature, demos, inputs)
        inputs = dict(prompt=inputs) if isinstance(inputs, str) else dict(messages=inputs)

        try:
            provider = lm.model.split("/", 1)[0] or "openai"
            if "response_format" in litellm.get_supported_openai_params(model=lm.model, custom_llm_provider=provider):
                outputs = lm(**inputs, **lm_kwargs, response_format={"type": "json_object"})
            else:
                outputs = lm(**inputs, **lm_kwargs)

        except litellm.UnsupportedParamsError:
            outputs = lm(**inputs, **lm_kwargs)

        values = []

        for output in outputs:
            value = self.parse(signature, output, _parse_values=_parse_values)
            assert set(value.keys()) == set(
                signature.output_fields.keys()
            ), f"Expected {signature.output_fields.keys()} but got {value.keys()}"
            values.append(value)

        return values

    def format(self, signature, demos, inputs):
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
            messages.append(format_turn(signature, demo, role="user", incomplete=demo in incomplete_demos))
            messages.append(format_turn(signature, demo, role="assistant", incomplete=demo in incomplete_demos))

        messages.append(format_turn(signature, inputs, role="user"))

        return messages

    def parse(self, signature, completion, _parse_values=True):
        fields = json_repair.loads(completion)
        fields = {k: v for k, v in fields.items() if k in signature.output_fields}

        # attempt to cast each value to type signature.output_fields[k].annotation
        for k, v in fields.items():
            if k in signature.output_fields:
                fields[k] = parse_value(v, signature.output_fields[k].annotation)

        if fields.keys() != signature.output_fields.keys():
            raise ValueError(f"Expected {signature.output_fields.keys()} but got {fields.keys()}")

        return fields

    def format_turn(self, signature, values, role, incomplete=False):
        return format_turn(signature, values, role, incomplete)

    def format_fields(self, signature, values, role):
        fields_with_values = {
            FieldInfoWithName(name=field_name, info=field_info): values.get(
                field_name, "Not supplied for this particular example."
            )
            for field_name, field_info in signature.fields.items()
            if field_name in values
        }

        return format_fields(role=role, fields_with_values=fields_with_values)


def parse_value(value, annotation):
    if annotation is str:
        return str(value)

    parsed_value = value

    if isinstance(annotation, enum.EnumMeta):
        parsed_value = find_enum_member(annotation, value)
    elif isinstance(value, str):
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed_value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parsed_value = value

    return TypeAdapter(annotation).validate_python(parsed_value)


def _format_field_value(field_info: FieldInfo, value: Any) -> str:
    """
    Formats the value of the specified field according to the field's DSPy type (input or output),
    annotation (e.g. str, int, etc.), and the type of the value itself.

    Args:
      field_info: Information about the field, including its DSPy field type and annotation.
      value: The value of the field.
    Returns:
      The formatted value of the field, represented as a string.
    """
    if field_info.annotation is Image:
        raise NotImplementedError("Images are not yet supported in JSON mode.")

    return format_field_value(field_info=field_info, value=value, assume_text=True)


def format_fields(role: str, fields_with_values: Dict[FieldInfoWithName, Any]) -> str:
    """
    Formats the values of the specified fields according to the field's DSPy type (input or output),
    annotation (e.g. str, int, etc.), and the type of the value itself. Joins the formatted values
    into a single string, which is is a multiline string if there are multiple fields.

    Args:
      fields_with_values: A dictionary mapping information about a field to its corresponding
                          value.
    Returns:
      The joined formatted values of the fields, represented as a string.
    """

    if role == "assistant":
        d = fields_with_values.items()
        d = {k.name: v for k, v in d}
        return json.dumps(serialize_for_json(d), indent=2)

    output = []
    for field, field_value in fields_with_values.items():
        formatted_field_value = _format_field_value(field_info=field.info, value=field_value)
        output.append(f"[[ ## {field.name} ## ]]\n{formatted_field_value}")

    return "\n\n".join(output).strip()


def format_turn(signature: SignatureMeta, values: Dict[str, Any], role, incomplete=False) -> Dict[str, str]:
    """
    Constructs a new message ("turn") to append to a chat thread. The message is carefully formatted
    so that it can instruct an LLM to generate responses conforming to the specified DSPy signature.

    Args:
      signature: The DSPy signature to which future LLM responses should conform.
      values: A dictionary mapping field names (from the DSPy signature) to corresponding values
              that should be included in the message.
      role: The role of the message, which can be either "user" or "assistant".
      incomplete: If True, indicates that output field values are present in the set of specified
                  ``values``. If False, indicates that ``values`` only contains input field values.
    Returns:
      A chat message that can be appended to a chat thread. The message contains two string fields:
      ``role`` ("user" or "assistant") and ``content`` (the message text).
    """
    content = []

    if role == "user":
        fields: Dict[str, FieldInfo] = signature.input_fields
        if incomplete:
            content.append("This is an example of the task, though some input or output fields are not supplied.")
    else:
        fields: Dict[str, FieldInfo] = signature.output_fields

    if not incomplete:
        field_names: KeysView = fields.keys()
        if not set(values).issuperset(set(field_names)):
            raise ValueError(f"Expected {field_names} but got {values.keys()}")

    formatted_fields = format_fields(
        role=role,
        fields_with_values={
            FieldInfoWithName(name=field_name, info=field_info): values.get(
                field_name, "Not supplied for this particular example."
            )
            for field_name, field_info in fields.items()
        },
    )
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


def get_annotation_name(annotation):
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is None:
        if hasattr(annotation, "__name__"):
            return annotation.__name__
        else:
            return str(annotation)
    else:
        args_str = ", ".join(get_annotation_name(arg) for arg in args)
        return f"{get_annotation_name(origin)}[{args_str}]"


def enumerate_fields(fields):
    parts = []
    for idx, (k, v) in enumerate(fields.items()):
        parts.append(f"{idx+1}. `{k}`")
        parts[-1] += f" ({get_annotation_name(v.annotation)})"
        parts[-1] += f": {v.json_schema_extra['desc']}" if v.json_schema_extra["desc"] != f"${{{k}}}" else ""

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
            desc = "must be pareseable according to the following JSON schema: "
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
