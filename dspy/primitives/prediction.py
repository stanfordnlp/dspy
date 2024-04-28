import typing as t

from pydantic import BaseModel, ConfigDict

from dsp.utils import normalize_text
from dspy.primitives.example import Example
from dspy.signatures.signature import Signature, SignatureMeta

default_normalize = lambda s: normalize_text(s) or None


class Completions(BaseModel):
    # For some reason Signature is leading to an error here
    model_config = ConfigDict(arbitrary_types_allowed=True)

    signature: SignatureMeta
    examples: list[Example]
    input_kwargs: dict[str, t.Any]
    data: dict[str, list[t.Any]]

    def __init__(self, signature: Signature, examples: list[Example], input_kwargs: dict, **kwargs):
        data = {}
        for example in examples:
            for k, v in example.items():
                if k in data:
                    data[k].append(v)
                else:
                    data[k] = [v]

        super().__init__(signature=signature, examples=examples, input_kwargs=input_kwargs, data=data, **kwargs)

    def has_complete_example(self) -> bool:
        for example in self.examples:
            complete = True
            for field in self.signature.output_fields:
                if field not in example:
                    complete = False
                    break

            if complete:
                return True

        return False

    def add_example(self, example: Example, position: int = -1):
        assert type(example) == Example
        self.examples.insert(position, example)

    def extend_examples(self, examples: list[Example]):
        for idx, example in enumerate(examples):
            assert type(example) == Example, f"idx: {idx}, example: {example}"
        self.examples.extend(examples)

    def get_farthest_example(self) -> Example:
        # This may likely can be cleaned up to return the first complete example
        if len(self.examples) == 0:
            raise Exception("No examples available")

        max_example = self.examples[0]
        for example in self.examples:
            if len(example) > len(max_example):
                max_example = example

        return max_example

    def filter(self, field: str, value: str):
        i = 0
        while i < len(self.examples):
            if field not in self.examples[i] or normalize_text(self.examples[i][field]) != value:
                del self.examples[i]
            else:
                i += 1

    def remove_incomplete(self):
        i = 0
        while i < len(self.examples):
            if self.is_complete_example(self.examples[i]):
                i += 1
            else:
                del self.examples[i]

        # Reset Data object for correct number of completions
        data = {}
        for example in self.examples:
            for k, v in example.items():
                if k not in data:
                    data[k] = [v]
                else:
                    data[k].append(v)

        self.data = data

    def is_complete_example(self, example: Example) -> bool:
        complete = True
        for field in self.signature.output_fields:
            if field not in example:
                complete = False
                break

        return complete

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def __getitem__(self, key) -> Example:
        if isinstance(key, str):
            return self.__getattr__(key)

        return self.examples[key]

    def __getattr__(self, name):
        if name in self.data:
            return self.data[name][0]

        return self.__getattribute__(name)

    def items(self):
        return self.data.items()

    def __repr__(self):
        items_repr = ",\n    ".join(f"{k}={repr(v)}" for k, v in self.data.items())
        return f"Completions(\n    {items_repr}\n)"

    def __str__(self):
        return self.__repr__()


class Prediction(Example):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        del self._demos
        del self._input_keys

        self._completions = None

    @classmethod
    def from_completions(cls, completions: Completions):
        obj = cls()
        obj._completions = completions
        obj._store = {k: v[0] for k, v in obj._completions.items()}

        return obj

    def __repr__(self):
        store_repr = ",\n    ".join(f"{k}={repr(v)}" for k, v in self._store.items())

        if self._completions is None or len(self._completions) == 1:
            return f"Prediction(\n    {store_repr}\n)"

        num_completions = len(self._completions)
        return f"Prediction(\n    {store_repr},\n    completions=Completions(...)\n) ({num_completions-1} completions omitted)"

    def __str__(self):
        return self.__repr__()

    @property
    def completions(self):
        return self._completions
