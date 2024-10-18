from abc import ABC, abstractmethod
from copy import deepcopy


def update_output_with_keys_from_input(signature, inputs, output):
    for key in signature.output_fields:
        if key not in output and key in inputs:
            if inputs[key] is not None:
                output[key] = inputs[key]
    return output


def extend_generation(func):
    """
    Generate output values which conform to the given signature - repeatedly calling
    the language model until a complete value is generated or the maximum number of
    calls is reached.

    Args:
        func: The function to decorate - should be the __call__ method of the Adapter class
    """

    def wrapper(self, lm, lm_kwargs, signature, demos, inputs, _parse_values=True, max_extensions=0):
        incomplete_generations = [inputs]
        inner_lm_kwargs = deepcopy(lm_kwargs)
        values = []
        for i in range(max_extensions + 1):
            still_incomplete_generations = []
            for input_ in incomplete_generations:
                parsed_generations = func(
                    self, lm, inner_lm_kwargs, signature, demos, input_, _parse_values=_parse_values
                )
                for value in parsed_generations:
                    value = update_output_with_keys_from_input(signature, input_, value)
                    # if no output fields are generated, and this is the first generation, raise an error
                    # as the model is not generating any correctly formatted outputs
                    if len(value) == 0 and i == 0:
                        raise ValueError("No output values generated - check the input data and signature.")
                    if set(value.keys()) != set(signature.output_fields.keys()):
                        still_incomplete_generations.append({**input_, **value})
                    else:
                        values.append(value)

            if len(still_incomplete_generations) == 0:
                return values
            # Only generate one completion at a time when extending generations
            inner_lm_kwargs["n"] = 1
            incomplete_generations = still_incomplete_generations

        raise ValueError(f"Failed to generate the signature's complete output after {max_extensions} retries.")

    return wrapper


class Adapter(ABC):
    @abstractmethod
    def format(self, signature, demos, inputs):
        ...

    @abstractmethod
    def parse(self, signature, completion, _parse_values=True):
        ...

    @extend_generation
    def __call__(self, lm, lm_kwargs, signature, demos, inputs, _parse_values=True, **_):
        """
        Generate output values which conform to the given signature - repeatedly calling
        the language model until a complete value is generated or the maximum number of
        extensions is reached.

        Args:
            lm: The language model to use for generation.
            lm_kwargs (dict): Keyword arguments for the language model.
            signature: The expected signature of the output.
            demos (list): Demonstration examples.
            inputs (dict): Input data for generation.
            _parse_values (bool): Whether to parse the generated values.
            extension (int): Current recursion depth.

        Returns:
            list: A list of complete output values.

        Raises:
            ValueError: If unable to generate complete values within max_extensions.
        """
        inputs = self.format(signature, demos, inputs)
        inputs = dict(prompt=inputs) if isinstance(inputs, str) else dict(messages=inputs)

        outputs = lm(**inputs, **lm_kwargs)
        values = []

        for output in outputs:
            value = self.parse(signature, output, _parse_values=_parse_values)
            values.append(value)

        return values
