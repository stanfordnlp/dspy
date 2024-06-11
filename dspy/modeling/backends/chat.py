import logging
import typing as t

from pydantic import Field

from dspy.primitives import Completions, Example
from dspy.signatures.signature import Signature, SignatureMeta

from .text import DEFAULT_FORMAT_HANDLERS, DEFAULT_PARAMS, TextBackend, default_format_handler

logger = logging.getLogger(__name__)


class ChatBackend(TextBackend):
    system_prompt: t.Optional[str] = Field(default=None)

    def _build_example(
        self,
        signature: Signature,
        example: Example,
        is_demo: bool,
    ) -> t.Tuple[str, str]:
        inputs = []
        for field, info in signature.input_fields.items():
            if is_demo and field not in example:
                continue

            format_handler = (
                info.json_schema_extra.get("format") or DEFAULT_FORMAT_HANDLERS.get(field) or default_format_handler
            )
            inputs.append(f"{info.json_schema_extra['prefix']} {format_handler(example[field])}")

        if not is_demo:
            return "\n\n".join(inputs), ""

        outputs = []
        for field, info in signature.output_fields.items():
            if field not in example:
                continue

            format_handler = (
                info.json_schema_extra.get("format") or DEFAULT_FORMAT_HANDLERS.get(field) or default_format_handler
            )
            outputs.append(f"{info.json_schema_extra['prefix']} {format_handler(example[field])}")

        return "\n\n".join(inputs), "\n\n".join(outputs)

    def prepare_request(self, signature: Signature, example: Example, config: dict, **_kwargs) -> dict:
        options = {**DEFAULT_PARAMS, **self.params, **config}

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
            inputs, outputs = self._build_example(signature, demo, True)
            messages.append({"role": "user", "content": inputs.strip()})
            messages.append({"role": "assistant", "content": outputs.strip()})

        # Append actual input to coerce generation
        inputs, _ = self._build_example(signature, example, False)
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
