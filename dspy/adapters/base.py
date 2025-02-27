from abc import ABC, abstractmethod

from litellm import ContextWindowExceededError

from dspy.adapters.types import History
from dspy.utils.callback import with_callbacks


class Adapter(ABC):
    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # Decorate format() and parse() method with with_callbacks
        cls.format = with_callbacks(cls.format)
        cls.parse = with_callbacks(cls.parse)

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        inputs_ = self.format(signature, demos, inputs)
        inputs_ = dict(prompt=inputs_) if isinstance(inputs_, str) else dict(messages=inputs_)

        outputs = lm(**inputs_, **lm_kwargs)
        values = []

        try:
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

        except Exception as e:
            if isinstance(e, ContextWindowExceededError):
                # On context window exceeded error, we don't want to retry with a different adapter.
                raise e
            from dspy.adapters.json_adapter import JSONAdapter

            if not isinstance(self, JSONAdapter):
                return JSONAdapter()(lm, lm_kwargs, signature, demos, inputs)
            raise e

    @abstractmethod
    def format(self, signature, demos, inputs):
        raise NotImplementedError

    @abstractmethod
    def parse(self, signature, completion):
        raise NotImplementedError

    def format_finetune_data(self, signature, demos, inputs, outputs):
        raise NotImplementedError

    def format_turn(self, signature, values, role, incomplete=False, is_conversation_history=False):
        pass

    def format_conversation_history(self, signature, inputs):
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
