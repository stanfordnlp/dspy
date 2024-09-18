import re
from .base import Adapter
from .image_utils import encode_image

field_header_pattern = re.compile(r'\[\[\[ ### (\w+) ### \]\]\]')

class ChatAdapter(Adapter):
    def __init__(self):
        pass

    def format(self, signature, demos, inputs):
        messages = []

        # TODO: Extract `raw_demos` out of `demos`, i.e. demos where some of the output_fields are not filled in.
        # raw_demos = [demo for demo in demos if not all(k in demo for k in signature.output_fields)]
        # demos = [demo for demo in demos if demo not in raw_demos]

        messages.append({"role": "system", "content": prepare_instructions(signature)})
        # messages.append({"role": "system", "content": prepare_instructions(signature, raw_demos)})

        # TODO: Remove the raw_demos from demos.

        for demo in demos:
            output_fields_, demo_ = list(signature.output_fields.keys()) + ['completed'], {**demo, 'completed': ''}
            messages.append({"role": "user", "content": format_chat_turn(signature.input_fields.keys(), demo)})
            messages.append({"role": "assistant", "content": format_chat_turn(output_fields_, demo_)})
        
        messages.append({"role": "user", "content": format_chat_turn(signature.input_fields.keys(), inputs)})

        return messages
    
    def parse(self, signature, completion):
        sections = [(None, [])]

        for line in completion.splitlines():
            match = field_header_pattern.match(line.strip())
            if match: sections.append((match.group(1), []))
            else: sections[-1][1].append(line)

        sections = [(k, '\n'.join(v).strip()) for k, v in sections]

        fields = {}
        for k, v in sections:
            if (k not in fields) and (k in signature.output_fields): fields[k] = v

        if fields.keys() != signature.output_fields.keys():
            print("Expected", signature.output_fields.keys(), "but got", fields.keys(), "from", completion)
            raise ValueError(f"Expected {signature.output_fields.keys()} but got {fields.keys()}")

        return fields


def format_fields(fields):
    return '\n\n'.join([f"[[[ ### {k} ### ]]]\n{v}" for k, v in fields.items()]).strip()

def format_chat_turn(field_names, values):
    # TODO: Reinstate validation after dealing with raw_demos in the system messages.
    # if not set(values).issuperset(set(field_names)):
    #     raise ValueError(f"Expected {field_names} but got {values.keys()}")
    
    # text_content = format_fields({k: values[k] for k in field_names if 'image' not in k})
    
    request = []
    
    for k in field_names:
        if 'image' in k:
            image = values.get(k)
            if not image:
                raise ValueError(f"Image not found for field {k}")
            
            image_base64 = encode_image(image)
            if not image_base64:
                raise ValueError(f"Failed to encode image for field {k}")
            
            if request and request[-1]["type"] == "text":
                request[-1]["text"] += f"\n\n[[[ ### {k} ### ]]]\n"
            else:
                request.append({
                    "type": "text",
                    "text": f"\n\n[[[ ### {k} ### ]]]\n"
                })
            
            request.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            })
        else:
            request.append({
                "type": "text",
                "text": format_fields({k: values[k]})
            })
    
    return request

def enumerate_fields(fields):
    parts = []
    for idx, (k, v) in enumerate(fields.items()):
        parts.append(f"{idx+1}. `{k}`")
        parts[-1] += f" ({v.annotation.__name__})"
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

    objective = ('\n' + ' ' * 8).join([''] + signature.instructions.splitlines())
    parts.append(f"In adhering to this structure, your objective is: {objective}")

    parts.append("You will receive some input fields in each interaction. " +
                 "Respond only with the corresponding output fields, starting with the field " +
                 ", then ".join(f"`{f}`" for f in signature.output_fields) +
                 ", and then ending with the marker for `completed`.")

    return '\n\n'.join(parts).strip()
