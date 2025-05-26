from dspy.adapters.chat_adapter import ChatAdapter
from typing import Any, Dict, NamedTuple, Optional, Type
from dspy.signatures.signature import Signature
from dspy.adapters.utils import (
    format_field_value,
    get_annotation_name,
    get_field_description_string,
    parse_value,
    translate_field_type,
)
from dspy.adapters.types import BaseType
import itertools


def format_field_value(value) -> list[dict]:
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    elif isinstance(value, list):
        formatted_list = [format_field_value(v) for v in value]
        flattened = list(itertools.chain.from_iterable(formatted_list))
        return flattened
    elif isinstance(value, BaseType) or hasattr(
        value, "format"
    ):  # Check if Custom Type
        return value.format()  # WARN: assumes a list. Dangerous.
    else:
        return value


class PassThroughChatAdapter(ChatAdapter):
    def format(
        self,
        signature: Type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        inputs_copy = dict(inputs)

        # If the signature and inputs have conversation history, we need to format the conversation history and
        # remove the history field from the signature.
        history_field_name = self._get_history_field_name(signature)
        if history_field_name:
            # In order to format the conversation history, we need to remove the history field from the signature.
            signature_without_history = signature.delete(history_field_name)
            conversation_history = self.format_conversation_history(
                signature_without_history,
                history_field_name,
                inputs_copy,
            )

        messages = []
        system_message = (
            f"{self.format_field_description(signature)}\n"
            f"{self.format_field_structure(signature)}\n"
            f"{self.format_task_description(signature)}"
        )
        messages.append({"role": "system", "content": system_message})
        messages.extend(self.format_demos(signature, demos))
        if history_field_name:
            # Conversation history and current input
            content_parts = self.format_user_message_content(
                signature_without_history, inputs_copy, main_request=True
            )
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": content_parts})
        else:
            # Only current input
            content_parts = self.format_user_message_content(
                signature, inputs_copy, main_request=True
            )
            messages.append({"role": "user", "content": content_parts})

        return messages

    def format_user_message_content(
        self,
        signature: Type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> list[dict[str, Any]]:
        messages = [{"type": "text", "text": prefix}]
        for k, v in signature.input_fields.items():
            messages.append(
                {
                    "type": "text",
                    "text": f"[[ ## {k} ## ]]\n",
                }
            )

            if k in inputs:
                value = inputs.get(k)
                normalized_value = format_field_value(value)
                messages.extend(normalized_value)

        if main_request:
            output_requirements = self.user_message_output_requirements(signature)
            if output_requirements is not None:
                messages.append({"type": "text", "text": output_requirements})

        messages.append({"type": "text", "text": suffix})
        return messages
