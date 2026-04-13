import random
from collections.abc import Mapping
from typing import TYPE_CHECKING

import dspy
from dspy.datasets.dataset import Dataset

if TYPE_CHECKING:
    import pandas as pd


class DataLoader(Dataset):
    """Load datasets from various sources and convert them to DSPy Examples.

    The DataLoader provides a unified interface for loading data from HuggingFace,
    CSV files, Pandas DataFrames, JSON files, Parquet files, and retrieval modules.
    Each loader converts rows into ``dspy.Example`` objects with optional input key
    annotations for use in DSPy programs.

    Example::

        loader = DataLoader()
        data = loader.from_huggingface("dataset_name", split="train")
        train, test = loader.train_test_split(data, train_size=0.8)
    """

    def __init__(self):
        pass

    def from_huggingface(
        self,
        dataset_name: str,
        *args,
        input_keys: tuple[str] = (),
        fields: tuple[str] | None = None,
        **kwargs,
    ) -> Mapping[str, list[dspy.Example]] | list[dspy.Example]:
        """Load a dataset from HuggingFace Hub and convert to DSPy Examples.

        Wraps the ``datasets.load_dataset`` function and converts each row into a
        ``dspy.Example``. When multiple splits are returned, a mapping of split
        names to example lists is returned; otherwise a flat list is returned.

        Args:
            dataset_name: The name of the dataset on HuggingFace Hub
                (e.g. ``"rajpurkar/squad"``).
            *args: Additional positional arguments forwarded to
                ``datasets.load_dataset``.
            input_keys: Tuple of field names to mark as inputs via
                ``Example.with_inputs()``.
            fields: Optional tuple of field names to select from each row. If
                ``None``, all fields are included.
            **kwargs: Additional keyword arguments forwarded to
                ``datasets.load_dataset`` (e.g. ``split="train"``).

        Returns:
            A dict mapping split names to lists of Examples when multiple splits
            are loaded, or a flat list of Examples for a single split.

        Raises:
            ValueError: If ``fields`` is not a tuple or ``input_keys`` is not a
                tuple.
        """
        if fields and not isinstance(fields, tuple):
            raise ValueError("Invalid fields provided. Please provide a tuple of fields.")

        if not isinstance(input_keys, tuple):
            raise ValueError("Invalid input keys provided. Please provide a tuple of input keys.")

        from datasets import load_dataset

        dataset = load_dataset(dataset_name, *args, **kwargs)

        if isinstance(dataset, list) and isinstance(kwargs["split"], list):
            dataset = {split_name: dataset[idx] for idx, split_name in enumerate(kwargs["split"])}

        try:
            returned_split = {}
            for split_name in dataset.keys():
                if fields:
                    returned_split[split_name] = [
                        dspy.Example({field: row[field] for field in fields}).with_inputs(*input_keys)
                        for row in dataset[split_name]
                    ]
                else:
                    returned_split[split_name] = [
                        dspy.Example({field: row[field] for field in row.keys()}).with_inputs(*input_keys)
                        for row in dataset[split_name]
                    ]

            return returned_split
        except AttributeError:
            if fields:
                return [
                    dspy.Example({field: row[field] for field in fields}).with_inputs(*input_keys) for row in dataset
                ]
            else:
                return [
                    dspy.Example({field: row[field] for field in row.keys()}).with_inputs(*input_keys)
                    for row in dataset
                ]

    def from_csv(
        self,
        file_path: str,
        fields: list[str] | None = None,
        input_keys: tuple[str] = (),
    ) -> list[dspy.Example]:
        """Load examples from a CSV file.

        Args:
            file_path: Path to the CSV file.
            fields: Optional list of column names to include. If ``None``, all
                columns are used.
            input_keys: Tuple of field names to mark as inputs.

        Returns:
            A list of ``dspy.Example`` objects, one per row.
        """
        from datasets import load_dataset

        dataset = load_dataset("csv", data_files=file_path)["train"]

        if not fields:
            fields = list(dataset.features)

        return [dspy.Example({field: row[field] for field in fields}).with_inputs(*input_keys) for row in dataset]

    def from_pandas(
        self,
        df: "pd.DataFrame",
        fields: list[str] | None = None,
        input_keys: tuple[str] = (),
    ) -> list[dspy.Example]:
        """Load examples from a Pandas DataFrame.

        Args:
            df: The Pandas DataFrame to convert.
            fields: Optional list of column names to include. If ``None``, all
                columns are used.
            input_keys: Tuple of field names to mark as inputs.

        Returns:
            A list of ``dspy.Example`` objects, one per row.
        """
        if fields is None:
            fields = list(df.columns)

        return [
            dspy.Example({field: row[field] for field in fields}).with_inputs(*input_keys) for _, row in df.iterrows()
        ]

    def from_json(
        self,
        file_path: str,
        fields: list[str] | None = None,
        input_keys: tuple[str] = (),
    ) -> list[dspy.Example]:
        """Load examples from a JSON or JSON Lines file.

        Args:
            file_path: Path to the JSON file.
            fields: Optional list of field names to include. If ``None``, all
                fields are used.
            input_keys: Tuple of field names to mark as inputs.

        Returns:
            A list of ``dspy.Example`` objects, one per row.
        """
        from datasets import load_dataset

        dataset = load_dataset("json", data_files=file_path)["train"]

        if not fields:
            fields = list(dataset.features)

        return [dspy.Example({field: row[field] for field in fields}).with_inputs(*input_keys) for row in dataset]

    def from_parquet(
        self,
        file_path: str,
        fields: list[str] | None = None,
        input_keys: tuple[str] = (),
    ) -> list[dspy.Example]:
        """Load examples from a Parquet file.

        Args:
            file_path: Path to the Parquet file.
            fields: Optional list of field names to include. If ``None``, all
                fields are used.
            input_keys: Tuple of field names to mark as inputs.

        Returns:
            A list of ``dspy.Example`` objects, one per row.
        """
        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files=file_path)["train"]

        if not fields:
            fields = list(dataset.features)

        return [dspy.Example({field: row[field] for field in fields}).with_inputs(*input_keys) for row in dataset]

    def from_rm(self, num_samples: int, fields: list[str], input_keys: list[str]) -> list[dspy.Example]:
        """Load examples from the configured retrieval module.

        Samples objects from the retrieval model (``dspy.settings.rm``) and
        converts them to ``dspy.Example`` objects.

        Args:
            num_samples: Number of samples to retrieve.
            fields: List of field names to extract from each retrieved object.
            input_keys: List of field names to mark as inputs.

        Returns:
            A list of ``dspy.Example`` objects from the retrieval module.

        Raises:
            ValueError: If no retrieval module is configured or the module does
                not support ``get_objects``.
        """
        try:
            rm = dspy.settings.rm
            try:
                return [
                    dspy.Example({field: row[field] for field in fields}).with_inputs(*input_keys)
                    for row in rm.get_objects(num_samples=num_samples, fields=fields)
                ]
            except AttributeError:
                raise ValueError(
                    "Retrieval module does not support `get_objects`. Please use a different retrieval module."
                )
        except AttributeError:
            raise ValueError(
                "Retrieval module not found. Please set a retrieval module using `dspy.configure`."
            )

    def sample(
        self,
        dataset: list[dspy.Example],
        n: int,
        *args,
        **kwargs,
    ) -> list[dspy.Example]:
        """Randomly sample examples from a dataset.

        Args:
            dataset: A list of ``dspy.Example`` objects to sample from.
            n: Number of samples to draw.
            *args: Additional positional arguments forwarded to
                ``random.sample``.
            **kwargs: Additional keyword arguments forwarded to
                ``random.sample``.

        Returns:
            A list of ``n`` randomly selected examples.

        Raises:
            ValueError: If ``dataset`` is not a list.
        """
        if not isinstance(dataset, list):
            raise ValueError(
                f"Invalid dataset provided of type {type(dataset)}. Please provide a list of `dspy.Example`s."
            )

        return random.sample(dataset, n, *args, **kwargs)

    def train_test_split(
        self,
        dataset: list[dspy.Example],
        train_size: int | float = 0.75,
        test_size: int | float | None = None,
        random_state: int | None = None,
    ) -> Mapping[str, list[dspy.Example]]:
        """Split a dataset into train and test subsets.

        Randomly shuffles the dataset and divides it into training and testing
        splits based on the specified sizes.

        Args:
            dataset: A list of ``dspy.Example`` objects to split.
            train_size: Size of the training split. If a float between 0 and 1,
                it represents the proportion of the dataset. If an int, it
                represents the absolute number of samples. Defaults to 0.75.
            test_size: Size of the test split. Accepts the same types as
                ``train_size``. If ``None``, the remaining samples after the
                train split are used.
            random_state: Optional seed for reproducible shuffling.

        Returns:
            A dict with ``"train"`` and ``"test"`` keys, each mapping to a list
            of ``dspy.Example`` objects.

        Raises:
            ValueError: If ``train_size`` or ``test_size`` are invalid, or if
                their sum exceeds the dataset length.
        """
        if random_state is not None:
            random.seed(random_state)

        dataset_shuffled = dataset.copy()
        random.shuffle(dataset_shuffled)

        if train_size is not None and isinstance(train_size, float) and (0 < train_size < 1):
            train_end = int(len(dataset_shuffled) * train_size)
        elif train_size is not None and isinstance(train_size, int):
            train_end = train_size
        else:
            raise ValueError(
                "Invalid `train_size`. Please provide a float between 0 and 1 to represent the proportion of the "
                "dataset to include in the train split or an int to represent the absolute number of samples to "
                f"include in the train split. Received `train_size`: {train_size}."
            )

        if test_size is not None:
            if isinstance(test_size, float) and (0 < test_size < 1):
                test_end = int(len(dataset_shuffled) * test_size)
            elif isinstance(test_size, int):
                test_end = test_size
            else:
                raise ValueError(
                    "Invalid `test_size`. Please provide a float between 0 and 1 to represent the proportion of the "
                    "dataset to include in the test split or an int to represent the absolute number of samples to "
                    f"include in the test split. Received `test_size`: {test_size}."
                )
            if train_end + test_end > len(dataset_shuffled):
                raise ValueError(
                    "`train_size` + `test_size` cannot exceed the total number of samples. Received "
                    f"`train_size`: {train_end}, `test_size`: {test_end}, and `dataset_size`: {len(dataset_shuffled)}."
                )
        else:
            test_end = len(dataset_shuffled) - train_end

        train_dataset = dataset_shuffled[:train_end]
        test_dataset = dataset_shuffled[train_end : train_end + test_end]

        return {"train": train_dataset, "test": test_dataset}
