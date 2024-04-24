import random
import typing as t
from functools import cached_property
from abc import abstractmethod, ABC

from pydantic import BaseModel, ConfigDict, Field

import dspy
from dspy import Example


class Dataset(BaseModel, ABC):
    # Example is not yet Pydantic Valid
    model_config = ConfigDict(arbitrary_types_allowed=True)

    train_seed: int = Field(default=0)
    dev_seed: int = Field(default=0)
    test_seed: int = Field(default=0)
    do_shuffle: bool = Field(default=True)
    train_examples: t.Optional[list[Example]] = Field(default=None)
    dev_examples: t.Optional[list[Example]] = Field(default=None)
    test_examples: t.Optional[list[Example]] = Field(default=None)

    @classmethod
    def load(
        cls,
        train: t.Optional[list[Example]] = None,
        dev: t.Optional[list[Example]] = None,
        test: t.Optional[list[Example]] = None,
    ):
        if train:
            self.train_examples = train

        if dev:
            self.dev_examples = dev

        if test:
            self.test_examples = test

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def reset_seeds(
        self,
        train_seed: t.Optional[int] = None,
        dev_seed: t.Optional[int] = None,
        test_seed: t.Optional[int] = None,
    ) -> None:
        self.train_seed = train_seed if train_seed is not None else self.train_seed
        self.dev_seed = dev_seed if dev_seed is not None else self.dev_seed
        self.test_seed = test_seed if test_seed is not None else self.test_seed

        self.train_examples = None
        self.dev_examples = None
        self.test_examples = None

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
