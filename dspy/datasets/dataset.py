import random
import typing as t
from functools import cached_property

from pydantic import BaseModel, ConfigDict, Field

import dspy
from dspy import Example


class Dataset(BaseModel):
    # Example is not yet Pydantic Valid, as such we need to allow arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    do_shuffle: bool = Field(default=True)
    data: dict[str, list[Example]] = Field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @cached_property
    def split_names(self) -> list[str]:
        return list(self.data.keys())

    def split_existing_split(self, source: str, target: str, percentage: float) -> None:
        if source not in self.data:
            raise ValueError(f"{source} is not available in Dataset.")

        if target not in self.data:
            raise ValueError(f"{target} is not available in Dataset.")

        source_examples = self.data[source]

        source_len = len(source_examples)
        count = int(source_len * percentage)
        target_idx = random.sample(list(range(source_len)), k=count)
        new_split = []

        for i in sorted(target_idx, key=lambda x: -x):
            new_split.append(source_examples[i])
            del source_examples[i]

        self.data[source] = source_examples
        self.data[target] = new_split

    def sample_split(
        self,
        name: str,
        seed: int = 0,
        n: t.Optional[int] = None,
        shuffle: t.Optional[bool] = None,
    ) -> list[Example]:
        # TODO: This function is not currently deterministic
        # This can likely cause issues, as we are generating copies of data
        if name not in self.data:
            raise ValueError(f"{name} split not found in Dataset. Splits available: {self.split_names}")

        if shuffle is None:
            shuffle = self.do_shuffle

        examples = self.data[name]

        if shuffle:
            return self._shuffle(dataset=examples, seed=seed, sample_size=n)

        return examples

    def __getattr__(self, name: str):
        if name in self.data:
            return self.data[name]

        if hasattr(self, name):
            return self.__getattribute__(name)

        raise AttributeError(f"{name} not available in Dataset")

    @staticmethod
    def _shuffle(dataset: list[Example], seed: int, sample_size: t.Optional[int] = None) -> list[Example]:
        if sample_size is None:
            sample_size = len(dataset)

        elif sample_size > len(dataset):
            dspy.logger.warning(
                "Sample size is larger than provided dataset, returning all available samples in dataset.",
            )
            sample_size = len(dataset)

        random.seed(seed)
        return random.sample(dataset, k=sample_size)
