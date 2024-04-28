import random
import typing as t
from functools import cached_property
from abc import abstractmethod, ABC

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

import dspy
from dspy import Example


class Dataset(BaseModel, ABC):
    # Example is not yet Pydantic Valid
    model_config = ConfigDict(arbitrary_types_allowed=True)

    do_shuffle: bool = Field(default=True)
    data: dict[str, list[Example]] = Field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def get_split(self, name: str, seed: int = 0, n: t.Optional[int] = None, shuffle: t.Optional[bool] = None) -> list[Example]:
        if name not in self.data:
            raise ValueError(f"{name} split not found in Dataset. Splits available: {self.splits.keys()}")

        data = self.data[name]


    def __getattr__(self, name: str):
        if hasattr(self, name):
            return self.__getattribute__(name)

            return self.data[name]
        if name in self.data:

        raise AttributeError(f"{name} not found in Dataset")

    @cached_property
    def train(self) -> list[Example]:
        if self.train_examples is None:
            raise ValueError("Train data is not available")

        if self.do_shuffle:
            return self._shuffle(
                dataset=self.train_examples, seed=self.train_seed, sample_size=len(self.train_examples)
            )

        return self.train_examples

    @cached_property
    def dev(self) -> list[Example]:
        if self.dev_examples is None:
            raise ValueError("Dev data is not available")

        if self.do_shuffle:
            return self._shuffle(dataset=self.dev_examples, seed=self.dev_seed, sample_size=len(self.dev_examples))

        return self.dev_examples

    @cached_property
    def test(self) -> list[Example]:
        if self.test_examples is None:
            raise ValueError("Test data is not available")

        if self.do_shuffle:
            return self._shuffle(dataset=self.test_examples, seed=self.test_seed, sample_size=len(self.test_examples))

        return self.test_examples

    @staticmethod
    def _shuffle(dataset: list[Example], seed: int, sample_size: int) -> list[Example]:
        if sample_size > len(dataset):
            dspy.logger.warning(
                "Sample size is larger than provided dataset, returning all available samples in dataset.",
            )
            sample_size = len(dataset)

        random.seed(seed)
        return random.sample(dataset, k=sample_size)
