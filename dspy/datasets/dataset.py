import random
import typing as t
from functools import cached_property

from pydantic import BaseModel, ConfigDict, Field

import dspy
from dspy import Example


class Dataset(BaseModel):
    # Example is not yet Pydantic Valid
    model_config = ConfigDict(arbitrary_types_allowed=True)

    train_seed: int = Field(default=0)
    dev_seed: int = Field(default=0)
    test_seed: int = Field(default=0)
    do_shuffle: bool = Field(default=True)
    _train: t.Optional[list[Example]] = Field(default=None)
    _dev: t.Optional[list[Example]] = Field(default=None)
    _test: t.Optional[list[Example]] = Field(default=None)

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

        self._train = None
        self._dev = None
        self._test = None

    @cached_property
    def train(self) -> list[Example]:
        if self._train is None:
            raise ValueError("Train data is not available")

        if self.do_shuffle:
            return self._shuffle(dataset=self._train, seed=self.train_seed, sample_size=len(self._train))

        return self._train

    @cached_property
    def dev(self) -> list[Example]:
        if self._dev is None:
            raise ValueError("Dev data is not available")

        if self.do_shuffle:
            return self._shuffle(dataset=self._dev, seed=self.dev_seed, sample_size=len(self._dev))

        return self._dev

    @cached_property
    def test(self) -> list[Example]:
        if self._test is None:
            raise ValueError("Test data is not available")

        if self.do_shuffle:
            return self._shuffle(dataset=self._test, seed=self.test_seed, sample_size=len(self._test))

        return self._test

    @staticmethod
    def _shuffle(dataset: list[Example], seed: int, sample_size: int) -> list[Example]:
        if sample_size > len(dataset):
            dspy.logger.warning(
                "Sample size is larger than provided dataset, returning all available samples in dataset.",
            )
            sample_size = len(dataset)

        random.seed(seed)
        return random.sample(dataset, k=sample_size)
