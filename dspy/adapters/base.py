import abc
from dspy.utils.logging import logger


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

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        inputs = self.format(signature, demos, inputs)
        inputs = dict(prompt=inputs) if isinstance(inputs, str) else dict(messages=inputs)

        outputs = lm(**inputs, **lm_kwargs)
        values = []

        for output in outputs:
            try:
                value = self.parse(signature, output)
            except Exception as e:
                logger.exception("Failed to parse LLM output:\nInput: %s\nOutput: %s", inputs, output)
                raise e
            assert set(value.keys()) == set(
                signature.output_fields.keys()
            ), f"Expected {signature.output_fields.keys()} but got {value.keys()}"
            values.append(value)

        return values
