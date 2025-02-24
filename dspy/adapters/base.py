from abc import ABC, abstractmethod

from litellm import ContextWindowExceededError

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

                assert set(value.keys()) == set(signature.output_fields.keys()), \
                    f"Expected {signature.output_fields.keys()} but got {value.keys()}"
                
                if output_logprobs is not None:
                    value["logprobs"] = output_logprobs
                
                values.append(value)

            return values

        except Exception as e:
            if isinstance(e, ContextWindowExceededError):
                # On context window exceeded error, we don't want to retry with a different adapter.
                raise e
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
