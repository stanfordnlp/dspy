import logging
import typing as t

from pydantic import Field

from dspy.primitives import Completions, Example
from dspy.signatures.signature import Signature, SignatureMeta

from .text import DEFAULT_FORMAT_HANDLERS, TextBackend, default_format_handler

logger = logging.getLogger(__name__)


class ChatBackend(TextBackend):
    system_prompt: t.Optional[str] = Field(default=None)

    def _query(
        self,
        signature: Signature,
        example: Example,
        is_demo: bool,
        format_handlers: t.Dict[str, t.Callable],
    ) -> t.Tuple[str, str]:
        if is_demo:
            for name in signature.output_fields:
                if name not in example:
                    raise Exception(f"Example missing necessary output field: {name}")

        input = []
        for name, field in signature.input_fields.items():
            format_handler = format_handlers.get(name, default_format_handler)

            input.append(
                f"{field.json_schema_extra['prefix']} {format_handler(example[name])}",
            )

        output = []
        for name, field in signature.output_fields.items():
            format_handler = format_handlers.get(name, default_format_handler)

            if name not in example and not is_demo:
                output.append(f"{field.json_schema_extra['prefix']} ")
                break

            if name in example:
                output.append(f"{field.json_schema_extra['prefix']} {format_handler(example[name])}")

        return "\n\n".join(input), "\n\n".join(output)

    def prepare_request(self, signature: Signature, example: Example, config: dict, **_kwargs) -> dict:
        options = {**self.STANDARD_PARAMS, **self.params, **config}

        # Set up format handlers
        format_handlers = DEFAULT_FORMAT_HANDLERS
        for name, field in signature.fields.items():
            fmt = field.json_schema_extra.get("format")
            if fmt:
                format_handlers[name] = fmt

        messages = []

        # Append the system prompt
        system_prompt = ""
        if self.system_prompt:
            system_prompt += self.system_prompt.strip() + "\n\n"
        system_prompt += signature.instructions.strip() + "\n\n"
        system_prompt += self._guidelines(signature, example).strip()
        messages.append({"role": "system", "content": system_prompt})

        # Append each demo
        for demo in example.get("demos", []):
            inputs, outputs = self._query(signature, demo, True, format_handlers)
            messages.append({"role": "user", "content": inputs.strip()})
            messages.append({"role": "assistant", "content": outputs.strip()})

        # Append empty demo (actual input) to coerce generation
        inputs, _ = self._query(signature, example, False, format_handlers)
        messages.append({"role": "user", "content": inputs.strip()})

        options["messages"] = messages

        return options

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

        generated_messages = [c["message"] for c in response.choices if c["finish_reason"] != "length"]
        messages = input_kwargs["messages"]

        # Reconstruct the full text
        prompt_text = ""
        for idx, message in enumerate(messages):
            if message["role"] == "system":
                prompt_text += message["content"] + "\n\n---\n\n"
            elif message["role"] == "user":
                prompt_text += message["content"] + "\n\n"
                if idx < len(messages) - 1:
                    prompt_text += messages[idx + 1]["content"] + "\n\n---\n\n"

        # Extract examples
        extracted = [
            self._extract(signature, example, prompt_text + message["content"]) for message in generated_messages
        ]

        if type(signature) != SignatureMeta:
            raise AssertionError("Signature not provided appropriately.")

        return Completions(
            signature=signature,
            examples=extracted,
            input_kwargs=input_kwargs,
        )
