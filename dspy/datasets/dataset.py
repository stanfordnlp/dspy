from __future__ import annotations

import random
import uuid
from collections.abc import Iterable
from typing import Any

from dspy import Example
from dspy.dsp.utils import dotdict


class Dataset:
    """Base class for DSPy datasets.

    Provides a standard interface for loading, shuffling, and splitting datasets
    into train, dev, and test sets. Subclasses should populate ``_train``, ``_dev``,
    and ``_test`` iterables of dictionaries representing examples.

    The class handles shuffling (controlled by seed) and optional size-limiting
    for each split, converting raw dicts into :class:`dspy.Example` objects.

    Args:
        train_seed: Random seed for shuffling the training split. Defaults to 0.
        train_size: Maximum number of training examples to use. ``None`` means use all.
        eval_seed: Random seed for shuffling the dev and test splits. Defaults to 0.
        dev_size: Maximum number of dev examples to use. ``None`` means use all.
        test_size: Maximum number of test examples to use. ``None`` means use all.
        input_keys: List of keys to mark as input fields on each :class:`dspy.Example`.

    Examples:
        Subclass ``Dataset`` and set ``_train``, ``_dev``, ``_test``::

            class MyDataset(Dataset):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self._train = [{"question": "What is 1+1?", "answer": "2"}]
                    self._dev = [{"question": "What is 2+2?", "answer": "4"}]
                    self._test = []

            dataset = MyDataset(train_size=1, input_keys=["question"])
            print(dataset.train[0].question)
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
        """Reset the seeds and sizes for the dataset splits, clearing cached data.

        After calling this method, the next access to ``train``, ``dev``, or ``test``
        will reshuffle and resample with the updated parameters.

        Args:
            train_seed: New random seed for the training split.
            train_size: New maximum size for the training split.
            eval_seed: New random seed for the dev and test splits.
            dev_size: New maximum size for the dev split.
            test_size: New maximum size for the test split.
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
        """Return the training split as a list of :class:`dspy.Example` objects."""
        if not hasattr(self, "_train_"):
            self._train_ = self._shuffle_and_sample("train", self._train, self.train_size, self.train_seed)

        return self._train_

    @property
    def dev(self) -> list[Example]:
        """Return the dev (validation) split as a list of :class:`dspy.Example` objects."""
        if not hasattr(self, "_dev_"):
            self._dev_ = self._shuffle_and_sample("dev", self._dev, self.dev_size, self.dev_seed)

        return self._dev_

    @property
    def test(self) -> list[Example]:
        """Return the test split as a list of :class:`dspy.Example` objects."""
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
        """Prepare multiple train/eval splits using different random seeds.

        Creates one training set per seed and optionally divides the evaluation set
        across seeds for cross-validation-style evaluation.

        Args:
            train_seeds: List of random seeds for generating training splits.
                Defaults to ``[1, 2, 3, 4, 5]``.
            train_size: Number of training examples per seed. Defaults to 16.
            dev_size: Total number of dev examples. Defaults to 1000.
            divide_eval_per_seed: If ``True``, divides the dev set evenly across seeds.
                If ``False``, each seed gets the full dev set. Defaults to ``True``.
            eval_seed: Random seed for the dev split. Defaults to 2023.
            **kwargs: Additional keyword arguments passed to the dataset constructor.

        Returns:
            A ``dotdict`` with ``train_sets`` (list of training sets) and
            ``eval_sets`` (list of evaluation sets), one per seed.
        """
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
