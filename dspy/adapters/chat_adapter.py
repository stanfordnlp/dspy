import re
import ast
import json
import textwrap

from pydantic import TypeAdapter
from .base import Adapter
from typing import get_origin, get_args

field_header_pattern = re.compile(r'\[\[ ## (\w+) ## \]\]')


class ChatAdapter(Adapter):
    def __init__(self):
        pass

    def format(self, signature, demos, inputs):
        messages = []

        # Extract demos where some of the output_fields are not filled in.
        incomplete_demos = [demo for demo in demos if not all(k in demo for k in signature.fields)]
        complete_demos = [demo for demo in demos if demo not in incomplete_demos]
        incomplete_demos = [demo for demo in incomplete_demos \
                            if any(k in demo for k in signature.input_fields) and \
                                any(k in demo for k in signature.output_fields)]

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
            if match: sections.append((match.group(1), []))
            else: sections[-1][1].append(line)

        sections = [(k, '\n'.join(v).strip()) for k, v in sections]

        fields = {}
        for k, v in sections:
            if (k not in fields) and (k in signature.output_fields):
                try:
                    fields[k] = parse_value(v, signature.output_fields[k].annotation) if _parse_values else v
                except Exception as e:
                    raise ValueError(f"Error parsing field {k}: {e}.\n\n\t\tOn attempting to parse the value\n```\n{v}\n```")

        if fields.keys() != signature.output_fields.keys():
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

def enumerate_fields(fields):
    parts = []
    for idx, (k, v) in enumerate(fields.items()):
        parts.append(f"{idx+1}. `{k}`")
        parts[-1] += f" ({get_annotation_name(v.annotation)})"
        parts[-1] += f": {v.json_schema_extra['desc']}" if v.json_schema_extra['desc'] != f'${{{k}}}' else ''

    return '\n'.join(parts).strip()

def prepare_instructions(signature):
    parts = []
    parts.append("Your input fields are:\n" + enumerate_fields(signature.input_fields))
    parts.append("Your output fields are:\n" + enumerate_fields(signature.output_fields))
    parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

    parts.append(format_fields({f : f"{{{f}}}" for f in signature.input_fields}))
    parts.append(format_fields({f : f"{{{f}}}" for f in signature.output_fields}))
    parts.append(format_fields({'completed' : ""}))

    instructions = textwrap.dedent(signature.instructions)
    objective = ('\n' + ' ' * 8).join([''] + instructions.splitlines())
    parts.append(f"In adhering to this structure, your objective is: {objective}")

    # parts.append("You will receive some input fields in each interaction. " +
    #              "Respond only with the corresponding output fields, starting with the field " +
    #              ", then ".join(f"`{f}`" for f in signature.output_fields) +
    #              ", and then ending with the marker for `completed`.")

    return '\n\n'.join(parts).strip()
