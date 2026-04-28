from __future__ import annotations

import random
import uuid
from collections.abc import Iterable
from typing import Any

from dspy import Example
from dspy.dsp.utils import dotdict


class Dataset:
    """Base class for DSPy datasets with deterministic train/dev/test splits.

    The base class is responsible for shuffling and sampling the underlying
    splits with a fixed seed so that downstream evaluations are reproducible.
    Subclasses are expected to populate the raw splits (``self._train``,
    ``self._dev``, ``self._test``) with iterables of dictionaries; the public
    ``train``/``dev``/``test`` properties then turn those records into
    ``dspy.Example`` instances on first access and cache the result.

    Args:
        train_seed: Seed used to shuffle the training split. Defaults to ``0``.
        train_size: Optional cap on the number of training examples returned
            after shuffling. ``None`` keeps the full split.
        eval_seed: Seed used to shuffle both the dev and test splits. Defaults
            to ``0``.
        dev_size: Optional cap on the number of dev examples returned after
            shuffling. ``None`` keeps the full split.
        test_size: Optional cap on the number of test examples returned after
            shuffling. ``None`` keeps the full split.
        input_keys: Field names that should be marked as inputs on each
            generated ``dspy.Example`` (via ``with_inputs``). Defaults to an
            empty list.

    Attributes:
        do_shuffle: When ``True`` (the default), splits are shuffled with the
            configured seeds before sampling. Set to ``False`` in subclasses
            that need to preserve the source ordering.
        name: The subclass name; useful for logging and identifying the
            dataset in mixed pipelines.

    Examples:
        Subclasses populate ``self._train``/``self._dev``/``self._test``::

            class MyDataset(Dataset):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self._train = [{"question": "1+1?", "answer": "2"}]
                    self._dev = [{"question": "2+2?", "answer": "4"}]

            data = MyDataset(input_keys=["question"])
            example = data.train[0]  # dspy.Example with question marked as input
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
        """Update the shuffling seeds and split sizes, then invalidate caches.

        Any argument left as ``None`` keeps the existing value. Cached splits
        (``_train_``/``_dev_``/``_test_``) are removed so that the next access
        of ``train``/``dev``/``test`` re-shuffles and re-samples with the new
        configuration.

        Args:
            train_seed: New seed for the training split, or ``None`` to keep
                the current value.
            train_size: New cap on the training split, or ``None`` to keep the
                current value.
            eval_seed: New seed for both dev and test splits, or ``None`` to
                keep the current values.
            dev_size: New cap on the dev split, or ``None`` to keep the
                current value.
            test_size: New cap on the test split, or ``None`` to keep the
                current value.
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
        """Lazily build and cache the shuffled, sampled training split.

        Returns:
            A list of ``dspy.Example`` instances drawn from ``self._train``,
            shuffled with ``self.train_seed`` and capped at ``self.train_size``.
        """
        if not hasattr(self, "_train_"):
            self._train_ = self._shuffle_and_sample("train", self._train, self.train_size, self.train_seed)

        return self._train_

    @property
    def dev(self) -> list[Example]:
        """Lazily build and cache the shuffled, sampled dev split.

        Returns:
            A list of ``dspy.Example`` instances drawn from ``self._dev``,
            shuffled with ``self.dev_seed`` and capped at ``self.dev_size``.
        """
        if not hasattr(self, "_dev_"):
            self._dev_ = self._shuffle_and_sample("dev", self._dev, self.dev_size, self.dev_seed)

        return self._dev_

    @property
    def test(self) -> list[Example]:
        """Lazily build and cache the shuffled, sampled test split.

        Returns:
            A list of ``dspy.Example`` instances drawn from ``self._test``,
            shuffled with ``self.test_seed`` and capped at ``self.test_size``.
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
        """Build aligned training and evaluation sets for a list of seeds.

        Useful for sweeps where you want to evaluate the same model across
        several training seeds while keeping evaluation data consistent (or
        partitioned) across runs.

        Args:
            train_seeds: Seeds used to draw distinct training sets. Defaults
                to ``[1, 2, 3, 4, 5]``.
            train_size: Number of examples in each training set. Defaults to
                ``16``.
            dev_size: Total number of dev examples to draw before optional
                per-seed partitioning. Defaults to ``1000``.
            divide_eval_per_seed: If ``True``, the dev set is partitioned into
                disjoint slices of ``dev_size // len(train_seeds)`` examples,
                one per seed. If ``False``, every seed reuses the full dev
                slice. Defaults to ``True``.
            eval_seed: Seed used to shuffle the dev split. Defaults to
                ``2023``.
            **kwargs: Additional keyword arguments forwarded to the dataset
                constructor (for example, dataset-specific options).

        Returns:
            A ``dotdict`` with two fields:

            - ``train_sets``: list of training-example lists, one per seed.
            - ``eval_sets``: list of evaluation-example lists aligned with
              ``train_sets``.
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
