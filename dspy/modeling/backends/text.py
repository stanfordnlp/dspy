import typing as t

import regex

from dspy.modeling.lm import BaseLM, LMOutput
from dspy.primitives.example import Example
from dspy.primitives.prediction import Completions
from dspy.signatures.signature import Signature, SignatureMeta

from .base import BaseBackend


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


class TextBackend(BaseBackend):
    """Behaves like prior versions of DSPy, using a general text completion prompt format and parsing predictions."""

    lm: BaseLM

    @staticmethod
    def _guidelines(signature: Signature, example: Example) -> str:
        result = "Follow the following format.\n\n"

        field_strings = []
        for _, field in signature.fields.items():
            field_strings.append(
                f"{field.json_schema_extra['prefix']} {field.json_schema_extra['desc']}",
            )

        return result + "\n\n".join(field_strings)

    def _query(
        self,
        signature: Signature,
        example: Example,
        is_demo: bool,
        format_handlers: dict[str, t.Callable],
    ) -> str:
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
                result.append(f"{field.json_schema_extra['prefix']} {format_handler(example[name])}")

        return "\n\n".join(result)

    def prepare_request(
        self,
        signature: Signature,
        example: Example,
        config: dict[str, t.Any],
    ) -> dict:
        # Set up Format Handlers
        format_handlers = DEFAULT_FORMAT_HANDLERS
        for name, field in signature.fields.items():
            fmt = field.json_schema_extra.get("format")
            if fmt:
                format_handlers[name] = fmt

        prompt_spans = []

        # Start by getting the instructions
        prompt_spans.append(signature.instructions)

        # Generate the Guidelines
        prompt_spans.append(self._guidelines(signature, example))

        # Generate Spans for Each Demo
        for demo in example.get("demos", []):
            prompt_spans.append(self._query(signature, demo, True, format_handlers))

        # Generate Empty Demo for Generation
        prompt_spans.append(self._query(signature, example, False, format_handlers))

        content = "\n\n---\n\n".join([span.strip() for span in prompt_spans])

        messages = {"messages": [{"role": "user", "content": content}]}

        config.update(**messages)

        return config

    def _extract(self, signature: Signature, example: Example, text: str) -> Example:
        # We have to deepcopy, so that the values don't continously overwrite each other
        example = example.copy()

        if not text.endswith("\n\n---"):
            text = text + "\n\n---"

        # Generate Search Strings
        prefixes = (
            [field.json_schema_extra["prefix"] for _, field in signature.output_fields.items()]
            + ["---"]
            + [field.json_schema_extra["prefix"] for _, field in signature.input_fields.items()]
        )

        demos = example.get("demos", [])
        prefixes = [prefix.replace(" ", "\\s").replace("(", "\\(").replace(")", "\\)") for prefix in prefixes]

        for idx, (name, field) in enumerate(signature.output_fields.items()):
            stop_prefixes = "|".join(prefixes[idx:])

            target_prefix = (
                field.json_schema_extra["prefix"].replace(" ", "\\s").replace("(", "\\(").replace(")", "\\)")
            )

            search_string = f"(?s)\n\n{target_prefix}?(.+?)\n\n(?:{stop_prefixes})"
            matches = regex.findall(search_string, text)

            non_generated_count = 1 + sum([name in demo for demo in demos])
            if non_generated_count >= len(matches):
                matches = []
            else:
                matches = matches[non_generated_count:]

            if matches == [] and len(signature.output_fields) == 0:
                example[name] = text
            elif matches != []:
                example[name] = matches[-1].strip()

        return example

    def process_response(
        self,
        signature: Signature,
        example: Example,
        output: LMOutput,
        input_kwargs: dict,
    ) -> Completions:
        # Get the full text
        prompt_text = "\n\n".join([message["content"] for message in input_kwargs["messages"]])

        # Extract examples
        extracted = [self._extract(signature, example, prompt_text + generation) for generation in output.generations]

        if type(signature) != SignatureMeta:
            raise AssertionError("Signature not provided appropriately.")

        return Completions(
            signature=signature,
            examples=extracted,
            input_kwargs=input_kwargs,
        )
