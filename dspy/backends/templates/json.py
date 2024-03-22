
import json
from dspy.backends.templates.base import BaseTemplate
from dspy import Signature, Example


class JSONTemplate(BaseTemplate):
    def generate(self, signature: Signature, example: Example) -> str:

        prompt_spans = []

        # Start by getting the instructions
        prompt_spans.append(signature.instructions)

        # Generate the Guidelines
        prompt_spans.append(self._guidelines(signature, example))

        # Generate spans for all the demos
        for demo in example.demos:
            prompt_spans.append(self._example_span(signature, example))

        # Generate span for the active example
        prompt_spans.append(self._example_span(signature, example))

        return "\n\n--\n\n".join(prompt_spans)

    def extract(self, signature: Signature, example: Example, raw_pred: str) -> Example:

        example = example.copy()

        try:
            pred = json.loads(raw_pred)
            for k, v in pred.items():
                k = k.lower()
                if k in signature.fields:
                    example[k] = v
        except:
            pass

        return example

    @staticmethod
    def _guidelines(signature: Signature, example: Example) -> str:

        included_fields = []
        missing_fields = []
        for name, field in signature.fields.items():
            if name in example:
                included_fields.append((name, field))
            else:
                missing_fields.append((name, field))

        result = f"Provided the following:"
        for (name, field) in included_fields:
            result += f"\n{field.json_schema_extra['prefix']} {field.json_schema_extra['desc']}"

        result += f"\n\nPlease return the following fields:"
        for (name, field) in missing_fields:
            result += f"\n{field.json_schema_extra['prefix']} {field.json_schema_extra['desc']}"

        result += f"\n\nAccording to the following JSON schema:"

        schema = {"properties": {name: {"title": f"{name.capitalize()}", "type": "string"} for name, _ in signature.fields.items()},
                  "required": [name for name, _ in signature.fields.items()]}

        result += f"\n{json.dumps(schema)}"

        return result

    @staticmethod
    def _example_span(signature: Signature, example: Example) -> str:

        span = ""

        for name, field in signature.fields.items():
            if name in example:
                span += f"{field.json_schema_extra['prefix']} {example[name]}"

        return span




