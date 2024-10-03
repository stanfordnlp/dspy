import re
from typing import Any
from dsp.adapters.base_template import Field
from dspy.signatures.signature import Signature
from .base import Adapter
from .image_utils import encode_image, is_image

import ast
import json
import textwrap
from itertools import chain
from pydantic import TypeAdapter
from typing import get_origin, get_args

field_header_pattern = re.compile(r'\[\[ ## (\w+) ## \]\]')

class ChatAdapter(Adapter):
    """
    ChatAdapter is used to format and parse data for chat-based LLMs.
    """

    def format(self, signature: Signature, demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        # Extract demos where some of the output_fields are not filled in.
        incomplete_demos = [demo for demo in demos if not all(k in demo and demo[k] is not None for k in signature.fields)]
        complete_demos = [demo for demo in demos if demo not in incomplete_demos]
        incomplete_demos = [demo for demo in incomplete_demos \
                            if any(k in demo and demo[k] is not None for k in signature.input_fields) and \
                                any(k in demo and demo[k] is not None for k in signature.output_fields)]

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

def format_field(field_name, field_value):
    if is_image(field_value):
        image_data = encode_image(field_value)
        return [
            {"type": "text", "text": f"[[ ## {field_name} ## ]]\n"},
            {"type": "image_url", "image_url": {"url": image_data}}
        ]
    
    if isinstance(field_value, list):
        field_value = format_list(field_value)
    return [{"type": "text", "text": f"[[ ## {field_name} ## ]]\n{field_value}"}]

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
    fields_to_collapse = []      

    if role == "user":
        field_names = signature.input_fields.keys()
        if incomplete:
            fields_to_collapse.append({"type": "text", "text": "This is an example of the task, though some input or output fields are not supplied."})
    else:
        field_names = list(signature.output_fields.keys()) + ['completed']
        values = {**values, 'completed': ''}

    if not incomplete:
        if not set(values).issuperset(set(field_names)):
            raise ValueError(f"Expected {field_names} but got {values.keys()}")
    
    fields_to_collapse.extend([
        format_field(k, values.get(k, "Not supplied for this particular example."))
        for k in field_names
    ])

    if role == "user":
        output_fields = list(signature.output_fields.keys())
        if output_fields:
            fields_to_collapse.append({
                "type": "text",
                "text": (f"Respond with the corresponding output fields using the proper format of "
                         f"[[ ## <field_name> ## ]] followed by the field value. Start with the field "
                         f"{', then '.join(f'`{f}`' for f in output_fields)}, "
                         f"and then ending with the marker for `completed`.")
            })
        
    # flatmap the list if any items are lists otherwise keep the item
    flattened_list = list(chain.from_iterable(
        item if isinstance(item, list) else [item] for item in fields_to_collapse
    ))

    fields_to_collapse = flattened_list
    # Collapse all consecutive text messages into a single message.
    collapsed_messages = []
    for item in flattened_list:
        if item["type"] == "image_url" or (collapsed_messages and collapsed_messages[-1]["type"] != "text"):
            collapsed_messages.append(item)
        elif collapsed_messages and collapsed_messages[-1]["type"] == "text":
            collapsed_messages[-1]["text"] += "\n\n" + item["text"]
        else:
            collapsed_messages.append(item)

    # If all the messages are text, collapse them into a single message.
    if all(message["type"] == "text" for message in collapsed_messages):
        content = "\n\n".join(message["text"] for message in collapsed_messages)
    else:
        content = collapsed_messages

    return {"role": role, "content": content}


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
    parts = []
    for idx, (k, v) in enumerate(fields.items()):
        parts.append(f"{idx+1}. `{k}`")
        parts[-1] += f" ({get_annotation_name(v.annotation)})"
        parts[-1] += f": {v.json_schema_extra['desc']}" if v.json_schema_extra['desc'] != f'${{{k}}}' else ''

    return "\n".join(parts).strip()


def prepare_instructions(signature: Signature) -> str:
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