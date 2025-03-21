import textwrap
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Type

from dspy.adapters.types import History
from dspy.adapters.types.image import try_expand_image_tags
from dspy.adapters.utils import format_field_value, get_field_description_string
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback, with_callbacks

if TYPE_CHECKING:
    from dspy.clients.lm import LM


class Adapter(ABC):
    def __init__(self, callbacks: Optional[list[BaseCallback]] = None):
        self.callbacks = callbacks or []

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # Decorate format() and parse() method with with_callbacks
        cls.format = with_callbacks(cls.format)
        cls.parse = with_callbacks(cls.parse)

    def __call__(
        self,
        lm: "LM",
        lm_kwargs: dict[str, Any],
        signature: Type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        inputs = self.format(signature, demos, inputs)

        outputs = lm(messages=inputs, **lm_kwargs)
        values = []

        for output in outputs:
            output_logprobs = None

            if isinstance(output, dict):
                output, output_logprobs = output["text"], output["logprobs"]

            value = self.parse(signature, output)

            if output_logprobs is not None:
                value["logprobs"] = output_logprobs

            values.append(value)

        return values

    def format(
        self,
        signature: Type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        inputs_copy = dict(inputs)
        messages = []
        system_message = (
            f"{self.format_field_description(signature)}\n"
            f"{self.format_field_structure(signature)}\n"
            f"{self.format_objective(signature)}"
        )
        messages.append({"role": "system", "content": system_message})
        messages.extend(self.format_demos(signature, demos))
        messages.extend(self.format_conversation_history(signature, inputs_copy))
        messages.append({"role": "user", "content": self.format_user_message(signature, inputs_copy)})

        messages = try_expand_image_tags(messages)
        return messages

    def format_field_description(self, signature: Type[Signature]) -> str:
        return (
            f"Your input fields are:\n{get_field_description_string(signature.input_fields)}\n"
            f"Your output fields are:\n{get_field_description_string(signature.output_fields)}"
        )

    def format_field_structure(self, signature: Type[Signature]) -> str:
        raise NotImplementedError

    def format_objective(self, signature: Type[Signature]) -> str:
        instructions = textwrap.dedent(signature.instructions)
        objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
        return f"In adhering to this structure, your objective is: {objective}"

    def format_user_message(
        self,
        signature: Type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        include_output_format: bool = True,
    ) -> str:
        messages = [prefix]

        for k, v in signature.input_fields.items():
            value = inputs[k]
            formatted_field_value = format_field_value(field_info=v, value=value)
            messages.append(f"[[ ## {k} ## ]]\n{formatted_field_value}")

        if include_output_format:
            output_format = self.get_output_format_in_user_message(signature)
            if output_format is not None:
                messages.append(output_format)

        messages.append(suffix)
        return "\n\n".join(messages).strip()

    def get_output_format_in_user_message(self, signature: Type[Signature]) -> str:
        return None

    def format_assistant_message(
        self,
        signature: Type[Signature],
        outputs: dict[str, Any],
        missing_field_message: str = None,
    ) -> str:
        raise NotImplementedError

    def format_demos(self, signature: Type[Signature], demos: list[dict[str, Any]]) -> str:
        complete_demos = []
        incomplete_demos = []

        for demo in demos:
            # Check if all fields are present and not None
            is_complete = all(k in demo and demo[k] is not None for k in signature.fields)

            # Check if demo has at least one input and one output field
            has_input = any(k in demo for k in signature.input_fields)
            has_output = any(k in demo for k in signature.output_fields)

            if is_complete:
                complete_demos.append(demo)
            elif has_input and has_output:
                # We only keep incomplete demos that have at least one input and one output field
                incomplete_demos.append(demo)

        messages = []

        incomplete_demo_prefix = "This is an example of the task, though some input or output fields are not supplied."
        for demo in incomplete_demos:
            messages.append(
                {"role": "user", "content": self.format_user_message(signature, demo, prefix=incomplete_demo_prefix)}
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": self.format_assistant_message(
                        signature, demo, missing_field_message="Not supplied for this particular example. "
                    ),
                }
            )

        for demo in complete_demos:
            messages.append({"role": "user", "content": self.format_user_message(signature, demo)})
            messages.append(
                {
                    "role": "assistant",
                    "content": self.format_assistant_message(
                        signature, demo, missing_field_message="Not supplied for this conversation history message. "
                    ),
                }
            )

        return messages

    def format_conversation_history(self, signature: Type[Signature], inputs: dict[str, Any]) -> list[dict[str, Any]]:
        history_field_name = None
        for name, field in signature.input_fields.items():
            if field.annotation == History:
                history_field_name = name
                break

        if history_field_name is None:
            return []

        # In order to format the conversation history, we need to remove the history field from the signature.
        signature_without_history = signature.delete(history_field_name)
        conversation_history = inputs[history_field_name].messages if history_field_name in inputs else None

        if conversation_history is None:
            return []

        messages = []
        for message in conversation_history:
            messages.append(
                {
                    "role": "user",
                    "content": self.format_user_message(
                        signature_without_history, message, include_output_format=False
                    ),
                }
            )
            messages.append(
                {"role": "assistant", "content": self.format_assistant_message(signature_without_history, message)}
            )

        # Remove the history field from the inputs
        del inputs[history_field_name]

        return messages

    @abstractmethod
    def parse(self, signature: Type[Signature], completion: str) -> dict[str, Any]:
        raise NotImplementedError
