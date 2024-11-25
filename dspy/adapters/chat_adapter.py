import ast
import enum
import inspect
import json
import re
import textwrap
from collections.abc import Mapping
from itertools import chain
from typing import Any, Dict, List, Literal, NamedTuple, Union, get_args, get_origin

import pydantic
from pydantic import TypeAdapter
from pydantic.fields import FieldInfo

from dsp.adapters.base_template import Field
from dspy.adapters.base import Adapter
from dspy.adapters.utils import find_enum_member, format_field_value
from dspy.signatures.field import OutputField
from dspy.signatures.signature import Signature, SignatureMeta
from dspy.signatures.utils import get_dspy_field_type

field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")


class FieldInfoWithName(NamedTuple):
    name: str
    info: FieldInfo


# Built-in field indicating that a chat turn has been completed.
BuiltInCompletedOutputFieldInfo = FieldInfoWithName(name="completed", info=OutputField())


class ChatAdapter(Adapter):
    def format(self, signature: Signature, demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        # Extract demos where some of the output_fields are not filled in.
        incomplete_demos = [
            demo for demo in demos if not all(k in demo and demo[k] is not None for k in signature.fields)
        ]
        complete_demos = [demo for demo in demos if demo not in incomplete_demos]
        # Filter out demos that don't have at least one input and one output field.
        incomplete_demos = [
            demo
            for demo in incomplete_demos
            if any(k in demo for k in signature.input_fields) and any(k in demo for k in signature.output_fields)
        ]

        demos = incomplete_demos + complete_demos

        prepared_instructions = prepare_instructions(signature)
        messages.append({"role": "system", "content": prepared_instructions})

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

    # TODO(PR): Looks ok?
    def format_finetune_data(self, signature, demos, inputs, outputs):
        # Get system + user messages
        messages = self.format(signature, demos, inputs)

        # Add the assistant message
        role = "assistant"
        incomplete = False
        assistant_message = format_turn(signature, outputs, role, incomplete)
        messages.append(assistant_message)

        # Wrap the messages in a dictionary with a "messages" key
        return dict(messages=messages)

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

        return format_fields(fields_with_values)


def format_fields(fields_with_values: Dict[FieldInfoWithName, Any], assume_text=True) -> Union[str, List[dict]]:
    """
    Formats the values of the specified fields according to the field's DSPy type (input or output),
    annotation (e.g. str, int, etc.), and the type of the value itself. Joins the formatted values
    into a single string, which is is a multiline string if there are multiple fields.

    Args:
      fields_with_values: A dictionary mapping information about a field to its corresponding
                          value.
    Returns:
      The joined formatted values of the fields, represented as a string or a list of dicts
    """
    output = []
    for field, field_value in fields_with_values.items():
        formatted_field_value = format_field_value(field_info=field.info, value=field_value, assume_text=assume_text)
        if assume_text:
            output.append(f"[[ ## {field.name} ## ]]\n{formatted_field_value}")
        else:
            output.append({"type": "text", "text": f"[[ ## {field.name} ## ]]\n"})
            if isinstance(formatted_field_value, dict) and formatted_field_value.get("type") == "image_url":
                output.append(formatted_field_value)
            else:
                output[-1]["text"] += formatted_field_value["text"]
    if assume_text:
        return "\n\n".join(output).strip()
    else:
        return output


def parse_value(value, annotation):
    if annotation is str:
        return str(value)

    parsed_value = value

    if isinstance(annotation, enum.EnumMeta):
        return find_enum_member(annotation, value)
    elif isinstance(value, str):
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed_value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parsed_value = value

    return TypeAdapter(annotation).validate_python(parsed_value)


def format_turn(signature, values, role, incomplete=False):
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
    fields_to_collapse = []
    content = []

    if role == "user":
        fields = signature.input_fields
        if incomplete:
            fields_to_collapse.append(
                {
                    "type": "text",
                    "text": "This is an example of the task, though some input or output fields are not supplied.",
                }
            )
    else:
        fields = signature.output_fields
        # Add the built-in field indicating that the chat turn has been completed
        fields[BuiltInCompletedOutputFieldInfo.name] = BuiltInCompletedOutputFieldInfo.info
        values = {**values, BuiltInCompletedOutputFieldInfo.name: ""}
    field_names = fields.keys()
    if not incomplete:
        if not set(values).issuperset(set(field_names)):
            raise ValueError(f"Expected {field_names} but got {values.keys()}")

    fields_to_collapse.extend(
        format_fields(
            fields_with_values={
                FieldInfoWithName(name=field_name, info=field_info): values.get(
                    field_name, "Not supplied for this particular example."
                )
                for field_name, field_info in fields.items()
            },
            assume_text=False,
        )
    )

    if role == "user":
        output_fields = list(signature.output_fields.keys())

        def type_info(v):
            return (
                f" (must be formatted as a valid Python {get_annotation_name(v.annotation)})"
                if v.annotation is not str
                else ""
            )

        if output_fields:
            fields_to_collapse.append(
                {
                    "type": "text",
                    "text": "Respond with the corresponding output fields, starting with the field "
                    + ", then ".join(f"`[[ ## {f} ## ]]`{type_info(v)}" for f, v in signature.output_fields.items())
                    + ", and then ending with the marker for `[[ ## completed ## ]]`.",
                }
            )

    # flatmap the list if any items are lists otherwise keep the item
    flattened_list = list(
        chain.from_iterable(item if isinstance(item, list) else [item] for item in fields_to_collapse)
    )

    if all(message.get("type", None) == "text" for message in flattened_list):
        content = "\n\n".join(message.get("text") for message in flattened_list)
        return {"role": role, "content": content}

    # Collapse all consecutive text messages into a single message.
    collapsed_messages = []
    for item in flattened_list:
        # First item is always added
        if not collapsed_messages:
            collapsed_messages.append(item)
            continue

        # If the current item is image, add to collapsed_messages
        if item.get("type") == "image_url":
            if collapsed_messages[-1].get("type") == "text":
                collapsed_messages[-1]["text"] += "\n"
            collapsed_messages.append(item)
        # If the previous item is text and current item is text, append to the previous item
        elif collapsed_messages[-1].get("type") == "text":
            collapsed_messages[-1]["text"] += "\n\n" + item["text"]
        # If the previous item is not text(aka image), add the current item as a new item
        else:
            item["text"] = "\n\n" + item["text"]
            collapsed_messages.append(item)

    return {"role": role, "content": collapsed_messages}


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


def enumerate_fields(fields: dict[str, Field]) -> str:
    parts = []
    for idx, (k, v) in enumerate(fields.items()):
        parts.append(f"{idx+1}. `{k}`")
        parts[-1] += f" ({get_annotation_name(v.annotation)})"
        parts[-1] += f": {v.json_schema_extra['desc']}" if v.json_schema_extra["desc"] != f"${{{k}}}" else ""

    return "\n".join(parts).strip()


def move_type_to_front(d):
    # Move the 'type' key to the front of the dictionary, recursively, for LLM readability/adherence.
    if isinstance(d, Mapping):
        return {k: move_type_to_front(v) for k, v in sorted(d.items(), key=lambda item: (item[0] != "type", item[0]))}
    elif isinstance(d, list):
        return [move_type_to_front(item) for item in d]
    return d


def prepare_schema(type_):
    schema = pydantic.TypeAdapter(type_).json_schema()
    schema = move_type_to_front(schema)
    return schema


def prepare_instructions(signature: SignatureMeta):
    parts = []
    parts.append("Your input fields are:\n" + enumerate_fields(signature.input_fields))
    parts.append("Your output fields are:\n" + enumerate_fields(signature.output_fields))
    parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

    def field_metadata(field_name, field_info):
        field_type = field_info.annotation

        if get_dspy_field_type(field_info) == "input" or field_type is str:
            desc = ""
        elif field_type is bool:
            desc = "must be True or False"
        elif field_type in (int, float):
            desc = f"must be a single {field_type.__name__} value"
        elif inspect.isclass(field_type) and issubclass(field_type, enum.Enum):
            desc = f"must be one of: {'; '.join(field_type.__members__)}"
        elif hasattr(field_type, "__origin__") and field_type.__origin__ is Literal:
            desc = f"must be one of: {'; '.join([str(x) for x in field_type.__args__])}"
        else:
            desc = "must be pareseable according to the following JSON schema: "
            desc += json.dumps(prepare_schema(field_type), ensure_ascii=False)

        desc = (" " * 8) + f"# note: the value you produce {desc}" if desc else ""
        return f"{{{field_name}}}{desc}"

    def format_signature_fields_for_instructions(fields: Dict[str, FieldInfo]):
        return format_fields(
            fields_with_values={
                FieldInfoWithName(name=field_name, info=field_info): field_metadata(field_name, field_info)
                for field_name, field_info in fields.items()
            },
            assume_text=True,
        )

    parts.append(format_signature_fields_for_instructions(signature.input_fields))
    parts.append(format_signature_fields_for_instructions(signature.output_fields))
    parts.append(format_fields({BuiltInCompletedOutputFieldInfo: ""}, assume_text=True))
    instructions = textwrap.dedent(signature.instructions)
    objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
    parts.append(f"In adhering to this structure, your objective is: {objective}")

    return "\n\n".join(parts).strip()
