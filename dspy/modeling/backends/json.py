import json
import logging
import typing as t
from json import JSONDecodeError

from dspy.primitives.example import Example
from dspy.primitives.prediction import Completions
from dspy.signatures.signature import Signature

from .text import TextBackend

logger = logging.getLogger(__name__)


def patch_example(example: Example, data: dict[str, t.Any]) -> Example:
    example = example.copy()
    for k, v in data.items():
        example[k.lower()] = v

    return example


class JSONBackend(TextBackend):
    def _guidelines(self, signature: Signature, example: Example) -> str:
        included_fields = []
        missing_fields = []
        for name, field in signature.fields.items():
            if name in example:
                included_fields.append((name, field))
            else:
                missing_fields.append((name, field))

        result = "Provided the following:"
        for _, field in included_fields:
            result += f"\n{field.json_schema_extra['prefix']} {field.json_schema_extra['desc']}"

        result += "\n\nPlease return the following fields:"
        for _, field in missing_fields:
            result += f"\n{field.json_schema_extra['prefix']} {field.json_schema_extra['desc']}"

        result += "\n\nAccording to the following JSON schema:"
        schema = {
            "properties": {
                name: {"title": f"{name.capitalize()}", "type": "string"} for name, _ in signature.fields.items()
            },
            "required": [name for name, _ in signature.fields.items()],
        }

        result += f"\n{json.dumps(schema)}"
        return result

    def _example_span(self, signature: Signature, example: Example) -> str:
        span = {}
        for name, _ in signature.fields.items():
            if name in example:
                span[name] = f"{example[name]}"

        return json.dumps(span)

    def _extract(self, signature: Signature, example: Example, text: str) -> Example:
        example = example.copy()
        try:
            pred = json.loads(text)
            for k, v in pred.items():
                k_lower = k.lower()
                if k_lower in signature.fields:
                    example[k_lower] = v
        except JSONDecodeError:
            logger.error(f"Invalid text provided for JSON extraction: {text}")

        return example

    def prepare_request(self, signature: Signature, example: Example, config: dict, **_kwargs) -> dict:
        prompt_spans = []

        # Start by getting the instructions
        prompt_spans.append(signature.instructions)

        # Generate the Guidelines
        prompt_spans.append(self._guidelines(signature, example))

        # Generate Spans for All the demos
        for demo in example.demos:
            prompt_spans.append(self._example_span(signature, demo))

        # Generate Span for the active example
        prompt_spans.append(self._example_span(signature, example))

        content = "\n\n---\n\n".join(prompt_spans)

        config.update({"messages": [{"role": "user", "content": content}]})

        return config

    def process_response(
        self,
        signature: Signature,
        example: Example,
        response: t.Any,
        input_kwargs: dict,
        **_kwargs,
    ) -> Completions:
        if len([c for c in response.choices if c["finish_reason"] == "length"]) > 0:
            logger.info("Some of the generations are being limited by 'max_tokens', you may want to raise this value.")

        messages = [c["message"] for c in response.choices if c["finish_reason"] != "length"]

        extracted = [self._extract(signature, example, message["content"]) for message in messages]
        return Completions(signature=signature, examples=extracted, input_kwargs=input_kwargs)
