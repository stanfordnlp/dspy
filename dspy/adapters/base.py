from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Type

from dspy.adapters.types import History
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

    def __call__(self, lm: "LM", lm_kwargs: dict[str, Any], signature: Type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]:
        inputs_ = self.format(signature, demos, inputs)
        inputs_ = dict(prompt=inputs_) if isinstance(inputs_, str) else dict(messages=inputs_)

        outputs = lm(**inputs_, **lm_kwargs)
        values = []

        for output in outputs:
            output_logprobs = None

            if isinstance(output, dict):
                output, output_logprobs = output["text"], output["logprobs"]

            value = self.parse(signature, output)

            if set(value.keys()) != set(signature.output_fields.keys()):
                raise ValueError(
                    "Parsed output fields do not match signature output fields. "
                    f"Expected: {set(signature.output_fields.keys())}, Got: {set(value.keys())}"
                )

            if output_logprobs is not None:
                value["logprobs"] = output_logprobs

            values.append(value)

        return values


    @abstractmethod
    def format(self, signature: Type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def parse(self, signature: Type[Signature], completion: str) -> dict[str, Any]:
        raise NotImplementedError
    
    def format_fields(self, signature: Type[Signature], values: dict[str, Any], role: str) -> str:
        raise NotImplementedError
    
    def format_finetune_data(self, signature: Type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, list[Any]]:
        raise NotImplementedError

    def format_turn(self, signature: Type[Signature], values, role: str, incomplete: bool = False, is_conversation_history: bool = False) -> dict[str, Any]:
        raise NotImplementedError

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
                self.format_turn(signature_without_history, message, role="user", is_conversation_history=True)
            )
            messages.append(
                self.format_turn(signature_without_history, message, role="assistant", is_conversation_history=True)
            )

        inputs_copy = dict(inputs)
        del inputs_copy[history_field_name]

        messages.append(self.format_turn(signature_without_history, inputs_copy, role="user"))
        return messages
