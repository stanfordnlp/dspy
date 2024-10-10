import ast
import json
import re
import textwrap
from typing import Any, Dict, KeysView, List, Literal, NamedTuple, get_args, get_origin

import pydantic
from pydantic import TypeAdapter
from pydantic.fields import FieldInfo

from ..signatures.field import OutputField
from ..signatures.signature import SignatureMeta
from ..signatures.utils import get_dspy_field_type
from .base import Adapter

field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")


class FieldInfoWithName(NamedTuple):
    """
    A tuple containing a field name and its corresponding FieldInfo object.
    """

    name: str
    info: FieldInfo


# Built-in field indicating that a chat turn (i.e. a user or assistant reply to a chat
# thread) has been completed.
BuiltInCompletedOutputFieldInfo = FieldInfoWithName(name="completed", info=OutputField())


class ChatAdapter(Adapter):
    def __init__(self):
        pass

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
        sections = [(None, [])]

        for line in completion.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]

        fields = {}
        for k, v in sections:
            if (k not in fields) and (k in signature.output_fields):
                try:
                    fields[k] = parse_value(v, signature.output_fields[k].annotation) if _parse_values else v
                except Exception as e:
                    raise ValueError(
                        f"Error parsing field {k}: {e}.\n\n\t\tOn attempting to parse the value\n```\n{v}\n```"
                    )

        if fields.keys() != signature.output_fields.keys():
            raise ValueError(f"Expected {signature.output_fields.keys()} but got {fields.keys()}")

        return fields


def format_blob(blob):
    if "\n" not in blob and "«" not in blob and "»" not in blob:
        return f"«{blob}»"

    modified_blob = blob.replace("\n", "\n    ")
    return f"«««\n    {modified_blob}\n»»»"


def format_input_list_field_value(value: List[Any]) -> str:
    """
    Formats the value of an input field of type List[Any].

    Args:
      value: The value of the list-type input field.
    Returns:
      A string representation of the input field's list value.
    """
    if len(value) == 0:
        return "N/A"
    if len(value) == 1:
        return format_blob(value[0])

    return "\n".join([f"[{idx+1}] {format_blob(txt)}" for idx, txt in enumerate(value)])


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
    dspy_field_type: Literal["input", "output"] = get_dspy_field_type(field_info)
    if isinstance(value, list):
        if dspy_field_type == "input" and field_info.annotation is not str:
            # If the field is an input field of type List[Any], format the value as a numbered list
            # to make it easier for the LLM to parse individual elements from the list by index,
            # which is helpful for certain tasks (e.g. entity extraction)
            return format_input_list_field_value(value)
        else:
            # If the field is an output field or an input field of type str, format the value as
            # a stringified JSON Array. This ensures that downstream routines can parse the
            # field value correctly using methods from the `ujson` or `json` packages
            return json.dumps(value)
    elif isinstance(value, pydantic.BaseModel):
        return value.model_dump_json()
    else:
        return str(value)


def format_fields(fields_with_values: Dict[FieldInfoWithName, Any]) -> str:
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
    output = []
    for field, field_value in fields_with_values.items():
        formatted_field_value = _format_field_value(field_info=field.info, value=field_value)
        output.append(f"[[ ## {field.name} ## ]]\n{formatted_field_value}")

    return "\n\n".join(output).strip()


def parse_value(value, annotation):
    if annotation is str:
        return str(value)
    parsed_value = value
    if isinstance(value, str):
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed_value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parsed_value = value
    return TypeAdapter(annotation).validate_python(parsed_value)


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
        # Add the built-in field indicating that the chat turn has been completed
        fields[BuiltInCompletedOutputFieldInfo.name] = BuiltInCompletedOutputFieldInfo.info
        values = {**values, BuiltInCompletedOutputFieldInfo.name: ""}

    if not incomplete:
        field_names: KeysView = fields.keys()
        if not set(values).issuperset(set(field_names)):
            raise ValueError(f"Expected {field_names} but got {values.keys()}")

    formatted_fields = format_fields(
        fields_with_values={
            FieldInfoWithName(name=field_name, info=field_info): values.get(
                field_name, "Not supplied for this particular example."
            )
            for field_name, field_info in fields.items()
        }
    )
    content.append(formatted_fields)

    if role == "user":
        content.append(
            "Respond with the corresponding output fields, starting with the field "
            + ", then ".join(f"`{f}`" for f in signature.output_fields)
            + ", and then ending with the marker for `completed`."
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

    def format_signature_fields_for_instructions(fields: Dict[str, FieldInfo]):
        return format_fields(
            fields_with_values={
                FieldInfoWithName(name=field_name, info=field_info): f"{{{field_name}}}"
                for field_name, field_info in fields.items()
            }
        )

    parts.append(format_signature_fields_for_instructions(signature.input_fields))
    parts.append(format_signature_fields_for_instructions(signature.output_fields))
    parts.append(format_fields({BuiltInCompletedOutputFieldInfo: ""}))

    instructions = textwrap.dedent(signature.instructions)
    objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
    parts.append(f"In adhering to this structure, your objective is: {objective}")

    # parts.append("You will receive some input fields in each interaction. " +
    #              "Respond only with the corresponding output fields, starting with the field " +
    #              ", then ".join(f"`{f}`" for f in signature.output_fields) +
    #              ", and then ending with the marker for `completed`.")

    return "\n\n".join(parts).strip()
