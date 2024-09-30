import re
from typing import Any
from dsp.adapters.base_template import Field
from dspy.signatures.field import Image
from dspy.signatures.signature import Signature
from .base import Adapter
from .image_utils import encode_image


import ast
import json
import textwrap

from pydantic import TypeAdapter
from typing import get_origin, get_args

field_header_pattern = re.compile(r'\[\[ ## (\w+) ## \]\]')
# Which is correct? 
# field_header_pattern = re.compile(r"\[\[\[ ### (\w+) ### \]\]\]")

class ChatAdapter(Adapter):
    """
    ChatAdapter is used to format and parse data for chat-based LLMs.
    """

    def format(self, signature: Signature, demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        # Extract demos where some of the output_fields are not filled in.
        incomplete_demos = [demo for demo in demos if not all(k in demo for k in signature.fields)]
        complete_demos = [demo for demo in demos if demo not in incomplete_demos]
        incomplete_demos = [demo for demo in incomplete_demos \
                            if any(k in demo for k in signature.input_fields) and \
                                any(k in demo for k in signature.output_fields)]

        demos = incomplete_demos + complete_demos

        prepared_instructions = prepare_instructions(signature)
        messages.append({"role": "system", "content": prepared_instructions})

        # messages.append({"role": "system", "content": prepare_instructions(signature, raw_demos)})

        # TODO: Remove the raw_demos from demos.
        input_field_types = [field.annotation for field in signature.input_fields.values()]
        output_field_types = [field.annotation for field in signature.output_fields.values()]

        for demo in demos:
            output_fields_, demo_ = list(signature.output_fields.keys()) + ["completed"], {**demo, "completed": ""}

            # signature
            messages.append(
                {
                    "role": "user",
                    "content": format_chat_turn(signature.input_fields.keys(), input_field_types, demo),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": format_chat_turn(output_fields_, output_field_types, demo_),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": format_chat_turn(signature.input_fields.keys(), input_field_types, inputs),
            }
        )

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
                    raise ValueError(f"Error parsing field {k}: {e}.\n\n\t\tOn attempting to parse the value\n```\n{v}\n```")

        if fields.keys() != signature.output_fields.keys():
            print("Expected", signature.output_fields.keys(), "but got", fields.keys(), "from", completion)
            raise ValueError(f"Expected {signature.output_fields.keys()} but got {fields.keys()}")

        return fields

def format_blob(blob):
    if '\n' not in blob and "«" not in blob and "»" not in blob: return f"«{blob}»"

    modified_blob = blob.replace('\n', '\n    ')
    return f"«««\n    {modified_blob}\n»»»"


def format_list(items):
    if len(items) == 0: return "N/A"
    if len(items) == 1: return format_blob(items[0])

    return "\n".join([f"[{idx+1}] {format_blob(txt)}" for idx, txt in enumerate(items)])


def format_fields(fields):
    output = []
    for k, v in fields.items():
        v = v if not isinstance(v, list) else format_list(v)
        output.append(f"[[ ## {k} ## ]]\n{v}")

    return '\n\n'.join(output).strip()
        

def parse_value(value, annotation):
    if annotation is str: return str(value)
    parsed_value = value
    if isinstance(value, str):
        try: parsed_value = json.loads(value)
        except json.JSONDecodeError:
            try: parsed_value = ast.literal_eval(value)
            except (ValueError, SyntaxError): parsed_value = value
    return TypeAdapter(annotation).validate_python(parsed_value)


def format_turn(signature, values, role, incomplete=False):       
    content = []

    if role == "user":
        field_names = signature.input_fields.keys()
        if incomplete:
            content.append("This is an example of the task, though some input or output fields are not supplied.")
    else:
        field_names, values = list(signature.output_fields.keys()) + ['completed'], {**values, 'completed': ''}

    if not incomplete:
        if not set(values).issuperset(set(field_names)):
            raise ValueError(f"Expected {field_names} but got {values.keys()}")
    
    content.append(format_fields({k: values.get(k, "Not supplied for this particular example.") for k in field_names}))

    if role == "user":
        content.append("Respond with the corresponding output fields, starting with the field " +
                       ", then ".join(f"`{f}`" for f in signature.output_fields) +
                       ", and then ending with the marker for `completed`.")

    return {"role": role, "content": '\n\n'.join(content).strip()}


def get_annotation_name(annotation):
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is None:
        if hasattr(annotation, '__name__'):
            return annotation.__name__
        else:
            return str(annotation)
    else:
        args_str = ', '.join(get_annotation_name(arg) for arg in args)
        return f"{origin.__name__}[{args_str}]"

def enumerate_fields(fields: dict[str, Field]) -> str:
    """
    Enumerate the fields into a string.
    """
    parts = []
    for idx, (k, v) in enumerate(fields.items()):
        parts.append(f"{idx+1}. `{k}`")
        parts[-1] += f" ({get_annotation_name(v.annotation)})"
        parts[-1] += f": {v.json_schema_extra['desc']}" if v.json_schema_extra['desc'] != f'${{{k}}}' else ''

    return "\n".join(parts).strip()


def prepare_instructions(signature: Signature) -> str:
    """
    Convert the signature into an instructions string.
    """
    parts = []
    parts.append("Your input fields are:\n" + enumerate_fields(signature.input_fields))
    parts.append("Your output fields are:\n" + enumerate_fields(signature.output_fields))
    parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

    parts.append(format_fields({f: f"{{{f}}}" for f in signature.input_fields}))
    parts.append(format_fields({f: f"{{{f}}}" for f in signature.output_fields}))
    parts.append(format_fields({"completed": ""}))

    instructions = textwrap.dedent(signature.instructions)
    objective = ('\n' + ' ' * 8).join([''] + instructions.splitlines())
    parts.append(f"In adhering to this structure, your objective is: {objective}")

    # parts.append("You will receive some input fields in each interaction. " +
    #              "Respond only with the corresponding output fields, starting with the field " +
    #              ", then ".join(f"`{f}`" for f in signature.output_fields) +
    #              ", and then ending with the marker for `completed`.")

    return "\n\n".join(parts).strip()


def format_chat_turn(field_names: list[str], field_types: list[type], values: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Format the chat turn into a list of OAI-formatted messages.
    """
    # TODO: Reinstate validation after dealing with raw_demos in the system messages.
    # if not set(values).issuperset(set(field_names)):
    #     raise ValueError(f"Expected {field_names} but got {values.keys()}")

    message_contents: list[dict[str, str]] = []

    for field_name, field_type in zip(field_names, field_types):

        if field_type == Image:
            image = values[field_name]
            if not image:
                raise ValueError(f"Image not found for field {field_name}")

            image_base64 = encode_image(image)
            if not image_base64:
                raise ValueError(f"Failed to encode image for field {field_name}")

            if message_contents and message_contents[-1]["type"] == "text":
                message_contents[-1]["text"] += f"\n\n[[[ ### {field_name} ### ]]]\n"
            else:
                message_contents.append({"type": "text", "text": f"\n\n[[[ ### {field_name} ### ]]]\n"})

            message_contents.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            )
        else:
            message_contents.append({"type": "text", "text": format_fields({field_name: values[field_name]})})

    return message_contents
