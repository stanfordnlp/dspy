from abc import ABC, abstractmethod

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
                if isinstance(output, dict):
                    output_text = output["text"]
                else:
                    output_text = output

                if output_text:
                    # Output text, e.g., response["choices"][0]["message"]["content"] can be None when tool calls are
                    # used.
                    value = self.parse(signature, output_text)
                    if not set(value.keys()) == set(signature.output_fields.keys()):
                        raise ValueError(f"Expected {signature.output_fields.keys()} but got {value.keys()}")
                else:
                    value = {}

                if isinstance(output, dict) and "logprobs" in output:
                    value["logprobs"] = output["logprobs"]

                if isinstance(output, dict) and "tool_calls" in output:
                    value["tool_calls"] = output["tool_calls"]

                values.append(value)

            return values

        except Exception as e:
            from .json_adapter import JSONAdapter
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
