import random
from collections.abc import Mapping
from typing import TYPE_CHECKING

import dspy
from dspy.datasets.dataset import Dataset

if TYPE_CHECKING:
    import pandas as pd


class DataLoader(Dataset):
    """Utility for loading datasets from various sources into DSPy Examples.

    DataLoader provides methods to load data from Hugging Face Hub, CSV, JSON,
    Parquet files, Pandas DataFrames, and retrieval modules, converting each row
    into a `dspy.Example` with the specified input keys.

    Examples:
        ```python
        dl = dspy.DataLoader()
        dataset = dl.from_csv("data.csv", input_keys=("question",))
        trainset, testset = dl.train_test_split(dataset, train_size=0.8).values()
        ```
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
        """Load a dataset from Hugging Face Hub.

        Wraps `datasets.load_dataset` and converts each row into a
        `dspy.Example`. When the dataset has multiple splits the return value
        is a dict mapping split names to lists of examples; when a single split
        is requested a flat list is returned.

        Args:
            dataset_name: Name or path of the Hugging Face dataset
                (e.g. `"squad"`).
            *args: Positional arguments forwarded to `datasets.load_dataset`.
            input_keys: Fields to mark as inputs via
                `Example.with_inputs()`.
            fields: Subset of columns to keep. If `None`, all columns are
                included.
            **kwargs: Keyword arguments forwarded to `datasets.load_dataset`
                (e.g. `split="train"`).

        Returns:
            A dict of `{split_name: [dspy.Example, ...]}` when multiple
            splits are loaded, or a flat `[dspy.Example, ...]` for a single
            split.

        Raises:
            ValueError: If `fields` is not a tuple or `input_keys` is not
                a tuple.

        Examples:
            ```python
            dl = dspy.DataLoader()
            dataset = dl.from_huggingface(
                "squad",
                split="train[:100]",
                input_keys=("question", "context"),
            )
            ```
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
        """Load a dataset from a CSV file.

        Args:
            file_path: Path to the CSV file.
            fields: Columns to include. If `None`, all columns are included.
            input_keys: Fields to mark as inputs via
                `Example.with_inputs()`.

        Returns:
            A list of `dspy.Example` instances.

        Examples:
            ```python
            dl = dspy.DataLoader()
            dataset = dl.from_csv("data.csv", input_keys=("question",))
            ```
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
        """Load a dataset from a Pandas DataFrame.

        Args:
            df: The source DataFrame.
            fields: Columns to include. If `None`, all columns are included.
            input_keys: Fields to mark as inputs via
                `Example.with_inputs()`.

        Returns:
            A list of `dspy.Example` instances.

        Examples:
            ```python
            import pandas as pd
            dl = dspy.DataLoader()
            df = pd.DataFrame({"question": ["What is AI?"], "answer": ["..."]})
            dataset = dl.from_pandas(df, input_keys=("question",))
            ```
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
        """Load a dataset from a JSON or JSONL file.

        Args:
            file_path: Path to the JSON or JSONL file.
            fields: Columns to include. If `None`, all columns are included.
            input_keys: Fields to mark as inputs via
                `Example.with_inputs()`.

        Returns:
            A list of `dspy.Example` instances.

        Examples:
            ```python
            dl = dspy.DataLoader()
            dataset = dl.from_json("data.jsonl", input_keys=("question",))
            ```
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
        """Load a dataset from a Parquet file.

        Args:
            file_path: Path to the Parquet file.
            fields: Columns to include. If `None`, all columns are included.
            input_keys: Fields to mark as inputs via
                `Example.with_inputs()`.

        Returns:
            A list of `dspy.Example` instances.

        Examples:
            ```python
            dl = dspy.DataLoader()
            dataset = dl.from_parquet("data.parquet", input_keys=("question",))
            ```
        """
        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files=file_path)["train"]

        if not fields:
            fields = list(dataset.features)

        return [dspy.Example({field: row[field] for field in fields}).with_inputs(*input_keys) for row in dataset]

    def from_rm(self, num_samples: int, fields: list[str], input_keys: list[str]) -> list[dspy.Example]:
        """Load examples from the configured retrieval module.

        Fetches objects from the retrieval module set via `dspy.configure`
        and converts them into `dspy.Example` instances.

        Args:
            num_samples: Number of samples to retrieve.
            fields: Fields to include from each retrieved object.
            input_keys: Fields to mark as inputs via
                `Example.with_inputs()`.

        Returns:
            A list of `dspy.Example` instances.

        Raises:
            ValueError: If no retrieval module is configured or it does not
                support `get_objects`.
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
            raise ValueError("Retrieval module not found. Please set a retrieval module using `dspy.configure`.")

    def sample(
        self,
        dataset: list[dspy.Example],
        n: int,
        *args,
        **kwargs,
    ) -> list[dspy.Example]:
        """Return a random sample of examples from a dataset.

        Args:
            dataset: A list of `dspy.Example` instances to sample from.
            n: Number of examples to sample.
            *args: Additional arguments forwarded to `random.sample`.
            **kwargs: Additional keyword arguments forwarded to
                `random.sample`.

        Returns:
            A list of `n` randomly selected `dspy.Example` instances.

        Raises:
            ValueError: If `dataset` is not a list.

        Examples:
            ```python
            dl = dspy.DataLoader()
            subset = dl.sample(dataset, n=10)
            ```
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
        """Split a dataset into train and test sets.

        The dataset is shuffled before splitting. Sizes can be specified as
        floats (proportions) or ints (absolute counts).

        Args:
            dataset: A list of `dspy.Example` instances to split.
            train_size: If float, the proportion of the dataset for the train
                split (between 0 and 1). If int, the absolute number of train
                samples.
            test_size: If float, the proportion for the test split. If int,
                the absolute number of test samples. If `None`, defaults to
                the remainder after the train split.
            random_state: Seed for the random number generator used for
                shuffling. If `None`, results are non-deterministic.

        Returns:
            A dict with `"train"` and `"test"` keys mapping to lists of
            `dspy.Example` instances.

        Raises:
            ValueError: If `train_size` or `test_size` are invalid, or if
                their sum exceeds the dataset size.

        Examples:
            ```python
            dl = dspy.DataLoader()
            splits = dl.train_test_split(dataset, train_size=0.8, random_state=42)
            trainset, testset = splits["train"], splits["test"]
            ```
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
