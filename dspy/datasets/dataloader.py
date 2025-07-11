import random
from collections.abc import Mapping
from typing import TYPE_CHECKING

import dspy
from dspy.datasets.dataset import Dataset

if TYPE_CHECKING:
    import pandas as pd


class DataLoader(Dataset):
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
        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files=file_path)["train"]

        if not fields:
            fields = list(dataset.features)

        return [dspy.Example({field: row[field] for field in fields}).with_inputs(*input_keys) for row in dataset]

    def from_rm(self, num_samples: int, fields: list[str], input_keys: list[str]) -> list[dspy.Example]:
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
                "Retrieval module not found. Please set a retrieval module using `dspy.settings.configure`."
            )

    def sample(
        self,
        dataset: list[dspy.Example],
        n: int,
        *args,
        **kwargs,
    ) -> list[dspy.Example]:
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
