
import typing as t

import regex
from pydantic import Field

from dspy.primitives.example import Example
from dspy.signatures.signature import Signature

from .base import BaseTemplate


def passages_to_text(passages: t.Iterable[str]) -> str:
    assert len(passages) > 0
    if len(passages) > 1:
        return "\n".join(
            [f"[{idx + 1}] <<{text}>>" for idx, text in enumerate(passages)],
        )
    else:
        return passages[0]


def format_answers(answers: t.Iterable[str]) -> str:
    assert len(answers) > 0
    return (answers[0]).strip()


def default_format_handler(x: str) -> str:
    assert type(x) == str
    return " ".join(x.split())


DEFAULT_FORMAT_HANDLERS = {
    "context": passages_to_text,
    "passages": passages_to_text,
    "answers": format_answers,

}

class TextTemplate(BaseTemplate):

    format_handlers: dict[str, t.Callable] = Field(default_factory=dict)

    def generate(self, signature: Signature, example: Example) -> str:

        # Set up Format Handlers
        format_handlers = DEFAULT_FORMAT_HANDLERS
        format_handlers.update(self.format_handlers)
        for name, field in signature.fields.items():
            format = field.json_schema_extra.get("format")
            if format:
                format_handlers[name] = format

        prompt_spans = []

        # Start by getting the instructions
        prompt_spans.append(signature.instructions)

        # Generate the Guidelines
        prompt_spans.append(self._guidelines(signature))

        # Generate Spans for Each Demo
        for demo in example.get("demos", []):
            prompt_spans.append(self._query(signature, demo, True, format_handlers))

        # Generate Empty Demo for Generation
        prompt_spans.append(self._query(signature, example, False, format_handlers))

        return "\n\n---\n\n".join([span.strip() for span in prompt_spans])

    def extract(self, signature: Signature, example: Example, raw_pred: str) -> Example:

        # We have to deepcopy, so that the values dont continously overwrite each other
        example = example.copy()

        full_text = self.generate(signature, example) + raw_pred

        if not full_text.endswith("\n\n---"):
            full_text = full_text + "\n\n---"

        # Generate Search Strings
        prefixes = (
            [
                field.json_schema_extra["prefix"]
                for _, field in signature.output_fields.items()
            ]
            + ["---"]
            + [
                    field.json_schema_extra["prefix"]
                    for _, field in signature.input_fields.items()
            ]
        )

        demos = example.get("demos", [])
        prefixes = [
            prefix.replace(" ", "\\s").replace("(", "\\(").replace(")", "\\)")
            for prefix in prefixes
        ]

        for idx, (name, field) in enumerate(signature.output_fields.items()):
            stop_prefixes = "|".join(prefixes[idx:])

            target_prefix = (
                field.json_schema_extra["prefix"]
                .replace(" ", "\\s")
                .replace("(", "\\(")
                .replace(")", "\\)")
            )

            search_string = f"(?s)\n\n{target_prefix}?(.+?)\n\n(?:{stop_prefixes})"
            matches = regex.findall(search_string, full_text)

            # Skip the first match as this is likely the guidelines
            non_generated_count = 1 + sum([name in demo for demo in demos])
            if non_generated_count >= len(matches):
                matches = []
            else:
                matches = matches[non_generated_count:]

            if matches == [] and len(signature.output_fields) == 0:
                example[name] = full_text
            elif matches != []:
                example[name] = matches[-1].strip()

        return example

    def _guidelines(self, signature: Signature) -> str:

        result = "Follow the following format.\n\n"

        field_strings = []
        for _, field in signature.fields.items():
            field_strings.append(
                f"{field.json_schema_extra['prefix']} {field.json_schema_extra['desc']}",
            )

        result = result + "\n\n".join(field_strings)

        return result

    def _query(self, signature: Signature, example: Example, is_demo: bool, format_handlers: dict[str, t.Callable]) -> str:

        if is_demo:
            for name in signature.output_fields:
                if name not in example:
                    raise Exception(f"Example missing necessary input field: {name}")

        result = []

        for name, field in signature.input_fields.items():
            format_handler = format_handlers.get(name, default_format_handler)

            result.append(
                f"{field.json_schema_extra['prefix']} {format_handler(example[name])}",
            )

        for name, field in signature.output_fields.items():
            format_handler = format_handlers.get(name, default_format_handler)

            if name not in example and not is_demo:
                result.append(f"{field.json_schema_extra['prefix']} ")
                break
            elif name in example:
                result.append(
                    f"{field.json_schema_extra['prefix']} {format_handler(example[name])}",
                )

        return "\n\n".join(result)

