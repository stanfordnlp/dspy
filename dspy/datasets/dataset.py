from __future__ import annotations

import random
import uuid
from collections.abc import Iterable
from typing import Any

from dspy import Example
from dspy.dsp.utils import dotdict


class Dataset:
    """Base class for all DSPy datasets.

    The ``Dataset`` class provides a standard interface for loading, shuffling, and splitting data into
    train, dev, and test sets. Subclasses should populate the ``_train``, ``_dev``, and ``_test`` attributes
    with raw data (iterables of dicts), and this base class handles shuffling, sampling, and wrapping
    each record as a :class:`dspy.Example`.

    Shuffling is seeded for reproducibility. You can control the random seeds and split sizes at
    construction time or later via :meth:`reset_seeds`.

    Args:
        train_seed (int): Random seed used for shuffling the training set. Defaults to 0.
        train_size (int, optional): Maximum number of training examples to return. If None, all
            available training examples are returned.
        eval_seed (int): Random seed used for shuffling both the dev and test sets. Defaults to 0.
        dev_size (int, optional): Maximum number of dev examples to return. If None, all available
            dev examples are returned.
        test_size (int, optional): Maximum number of test examples to return. If None, all available
            test examples are returned.
        input_keys (list[str], optional): Field names to mark as inputs on every example via
            :meth:`Example.with_inputs`. If None, no input keys are set.

    Examples:
        ```python
        import dspy

        class MyDataset(dspy.Dataset):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._train = [
                    {"question": "What is 2+2?", "answer": "4"},
                    {"question": "What is 3+3?", "answer": "6"},
                ]
                self._dev = [
                    {"question": "What is 4+4?", "answer": "8"},
                ]
                self._test = []

        dataset = MyDataset(train_size=1, input_keys=["question"])
        print(len(dataset.train))  # 1
        print(dataset.train[0].question)
        ```
    """

    def __init__(
        self,
        train_seed: int = 0,
        train_size: int | None = None,
        eval_seed: int = 0,
        dev_size: int | None = None,
        test_size: int | None = None,
        input_keys: list[str] | None = None,
    ) -> None:
        self.train_size = train_size
        self.train_seed = train_seed
        self.dev_size = dev_size
        self.dev_seed = eval_seed
        self.test_size = test_size
        self.test_seed = eval_seed
        self.input_keys = input_keys or []

        self.do_shuffle = True

        self.name = self.__class__.__name__

    def reset_seeds(
        self,
        train_seed: int | None = None,
        train_size: int | None = None,
        eval_seed: int | None = None,
        dev_size: int | None = None,
        test_size: int | None = None,
    ) -> None:
        """Reset the random seeds and split sizes, clearing any cached splits.

        After calling this method, the next access to :attr:`train`, :attr:`dev`, or :attr:`test`
        will re-shuffle and re-sample the data with the updated seeds and sizes.

        Args:
            train_seed (int, optional): New random seed for the training split. If None, keeps the
                current seed.
            train_size (int, optional): New maximum number of training examples. If None, keeps the
                current size.
            eval_seed (int, optional): New random seed for both the dev and test splits. If None,
                keeps the current seed.
            dev_size (int, optional): New maximum number of dev examples. If None, keeps the current
                size.
            test_size (int, optional): New maximum number of test examples. If None, keeps the
                current size.
        """
        self.train_size = train_size or self.train_size
        self.train_seed = train_seed or self.train_seed
        self.dev_size = dev_size or self.dev_size
        self.dev_seed = eval_seed or self.dev_seed
        self.test_size = test_size or self.test_size
        self.test_seed = eval_seed or self.test_seed

        if hasattr(self, "_train_"):
            del self._train_

        if hasattr(self, "_dev_"):
            del self._dev_

        if hasattr(self, "_test_"):
            del self._test_

    @property
    def train(self) -> list[Example]:
        """Return the training split as a list of :class:`dspy.Example` objects.

        The result is shuffled and sampled on first access using :attr:`train_seed` and
        :attr:`train_size`, then cached. Call :meth:`reset_seeds` to regenerate.

        Returns:
            list[Example]: The training examples.
        """
        if not hasattr(self, "_train_"):
            self._train_ = self._shuffle_and_sample("train", self._train, self.train_size, self.train_seed)

        return self._train_

    @property
    def dev(self) -> list[Example]:
        """Return the development (validation) split as a list of :class:`dspy.Example` objects.

        The result is shuffled and sampled on first access using :attr:`dev_seed` and
        :attr:`dev_size`, then cached. Call :meth:`reset_seeds` to regenerate.

        Returns:
            list[Example]: The dev examples.
        """
        if not hasattr(self, "_dev_"):
            self._dev_ = self._shuffle_and_sample("dev", self._dev, self.dev_size, self.dev_seed)

        return self._dev_

    @property
    def test(self) -> list[Example]:
        """Return the test split as a list of :class:`dspy.Example` objects.

        The result is shuffled and sampled on first access using :attr:`test_seed` and
        :attr:`test_size`, then cached. Call :meth:`reset_seeds` to regenerate.

        Returns:
            list[Example]: The test examples.
        """
        if not hasattr(self, "_test_"):
            self._test_ = self._shuffle_and_sample("test", self._test, self.test_size, self.test_seed)

        return self._test_

    def _shuffle_and_sample(
        self, split: str, data: Iterable[dict[str, Any]], size: int | None, seed: int = 0
    ) -> list[Example]:
        data_list = list(data)

        # Shuffle the data irrespective of the requested size.
        base_rng = random.Random(seed)

        if self.do_shuffle:
            base_rng.shuffle(data_list)

        data_list = data_list[:size]
        output: list[Example] = []

        for example in data_list:
            example_obj = Example(**example, dspy_uuid=str(uuid.uuid4()), dspy_split=split)
            if self.input_keys:
                example_obj = example_obj.with_inputs(*self.input_keys)
            output.append(example_obj)
        # TODO: NOTE: Ideally we use these uuids for dedup internally, for demos and internal train/val splits.
        # Now, some tasks (like convQA and Colors) have overlapping examples. Here, we should allow the user to give us
        # a uuid field that would respect this in some way. This means that we need a more refined concept that
        # uuid (each example is unique) and more like a group_uuid.

        return output

    @classmethod
    def prepare_by_seed(
        cls,
        train_seeds: list[int] | None = None,
        train_size: int = 16,
        dev_size: int = 1000,
        divide_eval_per_seed: bool = True,
        eval_seed: int = 2023,
        **kwargs: Any,
    ) -> Any:
        train_seeds = train_seeds or [1, 2, 3, 4, 5]
        data_args = dotdict(train_size=train_size, eval_seed=eval_seed, dev_size=dev_size, test_size=0, **kwargs)
        dataset = cls(**data_args)

        eval_set = dataset.dev
        eval_sets: list[list[Example]] = []
        train_sets: list[list[Example]] = []

        examples_per_seed = dev_size // len(train_seeds) if divide_eval_per_seed else dev_size
        eval_offset = 0

        for train_seed in train_seeds:
            data_args.train_seed = train_seed
            dataset.reset_seeds(**data_args)

            eval_sets.append(eval_set[eval_offset : eval_offset + examples_per_seed])
            train_sets.append(dataset.train)

            assert len(eval_sets[-1]) == examples_per_seed, len(eval_sets[-1])
            assert len(train_sets[-1]) == train_size, len(train_sets[-1])

            if divide_eval_per_seed:
                eval_offset += examples_per_seed

        return dotdict(train_sets=train_sets, eval_sets=eval_sets)
