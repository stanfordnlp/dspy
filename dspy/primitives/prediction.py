import typing as t
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, Field
from dspy.primitives.example import Example
from dspy.signatures.signature import SignatureMeta, Signature


class Completion(BaseModel):
    # At some point in the future we should make the Example class pydantic
    # in the meantime this is needed to allow Completion/Completions to be Pydantic
    model_config = ConfigDict(arbitrary_types_allowed=True)
    example: Example
    complete: bool

    def __len__(self) -> int:
        return len(self.example.keys())

    def __getattr__(self, name):
        if name in self:
            return self.__getattribute__(name)

        if name in self.example:
            return self.example[name]

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def items(self):
        return self.example.items()


def convert_to_completion(signature: Signature, example: Example) -> Completion:
    complete = True
    for field in signature.output_fields:
        if field not in example:
            complete = False

    return Completion(example=example, complete=complete)


def get_completion_data(completions: list[Completion]) -> dict[str, list[t.Any]]:
    data = {}
    for completion in completions:
        if completion.complete:
            for k, v in completion.example.items():
                if k in data:
                    data[k].append(v)
                else:
                    data[k] = [v]

    return data


class Completions(BaseModel):
    # For some reason Signature is leading to an error here
    model_config = ConfigDict(arbitrary_types_allowed=True)

    signature: SignatureMeta
    completions: list[Completion]
    prompt: str
    kwargs: dict[str, t.Any]
    data: dict[str, list[t.Any]]

    def filter_complete(self) -> t.Self:
        cls = self.model_copy()

        i = 0
        while i < len(cls.completions):
            if cls.completions[i].complete:
                i += 1
            else:
                del cls.completions[i]

        return cls

    def __len__(self) -> int:
        i = 0
        for completion in self.completions:
            if completion.complete:
                i += 1

        return i

    def __getitem__(self, key) -> Completion:
        return self.completions[key]

    def __getattr__(self, name):
        if name in self.data:
            return self.data[name][0]

        if name in self:
            return self.__getattribute__(name)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

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
    def from_list_or_dict(cls, list_or_dict, signature=None):
        obj = cls()
        obj._completions = Completions(list_or_dict, signature=signature)
        obj._store = {k: v[0] for k, v in obj._completions.items()}

        return obj

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
