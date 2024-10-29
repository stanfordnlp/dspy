import abc
from dspy.utils.callback import with_callbacks


class Adapter:
    """
    Base class for all Adapters.
    Adapters are used to format and parse data for different types of LLMs.
    """

    @abc.abstractmethod
    def format(self, signature, demos, inputs):
        """
        Format the input data for the LLM.
        """

    @abc.abstractmethod
    def parse(self, signature, completion):
        """
        Parse the output data from the LLM.
        """

    def __call__(self, lm, lm_kwargs, signature, demos, inputs, _parse_values=True):
        inputs = self.format(signature, demos, inputs)
        inputs = dict(prompt=inputs) if isinstance(inputs, str) else dict(messages=inputs)


class Adapter:
    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # Decorate format() and parse() method with with_callbacks
        cls.format = with_callbacks(cls.format)
        cls.parse = with_callbacks(cls.parse)

    def __call__(self, lm, lm_kwargs, signature, demos, inputs, _parse_values=True):
        inputs_ = self.format(signature, demos, inputs)
        inputs_ = dict(prompt=inputs_) if isinstance(inputs_, str) else dict(messages=inputs_)

        outputs = lm(**inputs_, **lm_kwargs)
        values = []

        
        try:
            for output in outputs:
                value = self.parse(signature, output, _parse_values=_parse_values)
                assert set(value.keys()) == set(signature.output_fields.keys()), f"Expected {signature.output_fields.keys()} but got {value.keys()}"
                values.append(value)
            return values

        except Exception as e:
            from .json_adapter import JsonAdapter
            if _parse_values and not isinstance(self, JsonAdapter):
                return JsonAdapter()(lm, lm_kwargs, signature, demos, inputs, _parse_values=_parse_values)
            raise e

