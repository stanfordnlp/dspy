from abc import ABC, abstractmethod


# TODO(PR Reviewers): Should this be an abstract class? It looks like __call__
# expects .format and .parse to be implemented by the subclasses. We
# additionally use .format_turn for fine-tuning. Relatedly, did we want to
# rename this method?
class Adapter(ABC):
    
    def __init__(self):
        pass

    def __call__(self, lm, lm_kwargs, signature, demos, inputs, _parse_values=True):
        inputs = self.format(signature, demos, inputs)
        inputs = dict(prompt=inputs) if isinstance(inputs, str) else dict(messages=inputs)

        outputs = lm(**inputs, **lm_kwargs)
        values = []

        for output in outputs:
            value = self.parse(signature, output, _parse_values=_parse_values)
            assert set(value.keys()) == set(signature.output_fields.keys()), f"Expected {signature.output_fields.keys()} but got {value.keys()}"
            values.append(value)
        
        return values

    @abstractmethod
    def format(self, signature, demos, inputs):
       raise NotImplementedError

    @abstractmethod
    def format_finetune_data(self, signature, demos, inputs, outputs):
        raise NotImplementedError

    @abstractmethod
    def parse(self, signature, completion, _parse_values):
       raise NotImplementedError
