import re
import ast
import json
import enum
import inspect
import pydantic
import textwrap

from pydantic import TypeAdapter
from collections.abc import Mapping
from pydantic.fields import FieldInfo
from typing import Any, Dict, KeysView, List, Literal, NamedTuple, get_args, get_origin

from dspy.adapters.base import Adapter
from ..signatures.field import OutputField
from ..signatures.signature import SignatureMeta
from ..signatures.utils import get_dspy_field_type

field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")


class FieldInfoWithName(NamedTuple):
    name: str
    info: FieldInfo


# Built-in field indicating that a chat turn has been completed.
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

    def format_turn(self, signature, values, role, incomplete=False):
        return format_turn(signature, values, role, incomplete)

    def format_fields(self, signature, values):
        fields_with_values = {
            FieldInfoWithName(name=field_name, info=field_info): values.get(
                field_name, "Not supplied for this particular example."
            )
            for field_name, field_info in signature.fields.items()
            if field_name in values
        }

        return format_fields(fields_with_values)
        
        

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


def _serialize_for_json(value):
    if isinstance(value, pydantic.BaseModel):
        return value.model_dump()
    elif isinstance(value, list):
        return [_serialize_for_json(item) for item in value]
    elif isinstance(value, dict):
        return {key: _serialize_for_json(val) for key, val in value.items()}
    else:
        return value

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

    if isinstance(value, list) and field_info.annotation is str:
        # If the field has no special type requirements, format it as a nice numbere list for the LM.
        return format_input_list_field_value(value)
    elif isinstance(value, pydantic.BaseModel) or isinstance(value, dict) or isinstance(value, list):
        return json.dumps(_serialize_for_json(value))
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

    if isinstance(annotation, enum.EnumMeta):
        parsed_value = annotation[value]
    elif isinstance(value, str):
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
        def type_info(v):
            return f" (must be formatted as a valid Python {get_annotation_name(v.annotation)})" \
                if v.annotation is not str else ""
        
        if not incomplete:
            content.append(
                "Respond with the corresponding output fields, starting with the field "
                + ", then ".join(f"`[[ ## {f} ## ]]`{type_info(v)}" for f, v in signature.output_fields.items())
                + ", and then ending with the marker for `[[ ## completed ## ]]`."
            )

            # content.append(
            #     "Respond with the corresponding output fields, starting with the field "
            #     + ", then ".join(f"`{f}`" for f in signature.output_fields)
            #     + ", and then ending with the marker for `completed`."
            # )

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


def move_type_to_front(d):
    # Move the 'type' key to the front of the dictionary, recursively, for LLM readability/adherence.
    if isinstance(d, Mapping):
        return {k: move_type_to_front(v) for k, v in sorted(d.items(), key=lambda item: (item[0] != 'type', item[0]))}
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
        type_ = field_info.annotation

        if get_dspy_field_type(field_info) == 'input' or type_ is str:
            desc = ""
        elif type_ is bool:
            desc = "must be True or False"
        elif type_ in (int, float):
            desc = f"must be a single {type_.__name__} value"
        elif inspect.isclass(type_) and issubclass(type_, enum.Enum):
            desc= f"must be one of: {'; '.join(type_.__members__)}"
        elif hasattr(type_, '__origin__') and type_.__origin__ is Literal:
            desc = f"must be one of: {'; '.join([str(x) for x in type_.__args__])}"
        else:
            desc = "must be pareseable according to the following JSON schema: "
            desc += json.dumps(prepare_schema(type_))

        desc = (" " * 8) + f"# note: the value you produce {desc}" if desc else ""
        return f"{{{field_name}}}{desc}"

    def format_signature_fields_for_instructions(fields: Dict[str, FieldInfo]):
        return format_fields(
            fields_with_values={
                FieldInfoWithName(name=field_name, info=field_info): field_metadata(field_name, field_info)
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
