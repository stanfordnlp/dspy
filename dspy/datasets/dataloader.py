import random
from collections.abc import Mapping
from typing import List, Tuple, Union, Optional

import datasets
from datasets import load_dataset

import dspy
from dspy.datasets.dataset import Dataset


class DataLoader:
    @staticmethod
    def from_dataset(
        dataset: datasets.Dataset
        | datasets.DatasetDict
        | datasets.IterableDataset
        | datasets.IterableDatasetDict
        | dict,
        input_keys: Tuple[str],
        fields: Optional[Tuple[str]],
    ) -> Dataset:
        # If dataset is a DatasetDict, there is a key for each split
        # if only a dataset is returned, assume it is a single train split
        if not isinstance(dataset, datasets.DatasetDict):
            dataset = {"train": dataset}

        try:
            examples = {"train": [], "test": [], "dev": []}
            for split, rows in dataset.items():
                # If we provide fields only take those fields
                # otherwise, assume we are using all fields
                if fields:
                    if split in examples:
                        examples[split] = [
                            dspy.Example(**{field: row[field] for field in fields}).with_inputs(*input_keys)
                            for row in rows
                        ]
                else:
                    if split in examples:
                        examples[split] = [
                            dspy.Example(**{field: row[field] for field in row}).with_inputs(*input_keys)
                            for row in rows
                        ]

            return Dataset.load(**examples)

        except AttributeError:
            raise NotImplementedError()

    def from_huggingface(
        self,
        dataset_name: str,
        input_keys: Tuple[str],
        fields: Optional[Tuple[str]] = None,
        split: Optional[list[str]] = None,
        **kwargs,
    ) -> Dataset:
        if split is None:
            split = ["train", "test"]

        dataset = load_dataset(dataset_name, **kwargs)
        return DataLoader.from_dataset(dataset=dataset, input_keys=input_keys, fields=fields, split=split)

    def from_csv(
        self,
        file_path: str,
        input_keys: Tuple[str],
        fields: Optional[Tuple[str]],
        split: Optional[list[str]] = None,
        **kwargs,
    ) -> Dataset:
        if split is None:
            split = ["train"]

        dataset = load_dataset("csv", data_fields=file_path, **kwargs)
        return DataLoader.from_dataset(dataset=dataset, input_keys=input_keys, fields=fields, split=split)

    def from_json(
        self,
        file_path: str,
        input_keys: Tuple[str],
        fields: Optional[Tuple[str]],
        split: Optional[list[str]] = None,
        **kwargs,
    ) -> Dataset:
        if split is None:
            split = ["train"]

        dataset = load_dataset("json", data_files=file_path, **kwargs)
        return DataLoader.from_dataset(dataset=dataset, input_keys=input_keys, fields=fields, split=split)

    def from_parquet(
        self,
        file_path: str,
        input_keys: Tuple[str],
        fields: Optional[Tuple[str]] = None,
        split: Optional[list[str]] = None,
        **kwargs,
    ) -> Dataset:
        if split is None:
            split = ["train"]

        dataset = load_dataset("parquet", data_files=file_path, **kwargs)
        return DataLoader.from_dataset(dataset=dataset, input_keys=input_keys, fields=fields, split=split)

    # def sample(
    #     self,
    #     dataset: List[dspy.Example],
    #     n: int,
    #     *args,
    #     **kwargs,
    # ) -> List[dspy.Example]:
    #     raise NotImplementedError()
    #     # if not isinstance(dataset, list):
    #     #     raise ValueError(f"Invalid dataset provided of type {type(dataset)}. Please provide a list of examples.")
    #     #
    #     # return random.sample(dataset, n, *args, **kwargs)
    #
    # def train_test_split(
    #     self,
    #     dataset: List[dspy.Example],
    #     train_size: Union[int, float] = 0.75,
    #     test_size: Union[int, float] = None,
    #     random_state: int = None,
    # ) -> Mapping[str, List[dspy.Example]]:
    #     raise NotImplementedError()
    #     # if random_state is not None:
    #     #     random.seed(random_state)
    #     #
    #     # dataset_shuffled = dataset.copy()
    #     # random.shuffle(dataset_shuffled)
    #     #
    #     # if train_size is not None and isinstance(train_size, float) and (0 < train_size < 1):
    #     #     train_end = int(len(dataset_shuffled) * train_size)
    #     # elif train_size is not None and isinstance(train_size, int):
    #     #     train_end = train_size
    #     # else:
    #     #     raise ValueError("Invalid train_size. Please provide a float between 0 and 1 or an int.")
    #     #
    #     # if test_size is not None:
    #     #     if isinstance(test_size, float) and (0 < test_size < 1):
    #     #         test_end = int(len(dataset_shuffled) * test_size)
    #     #     elif isinstance(test_size, int):
    #     #         test_end = test_size
    #     #     else:
    #     #         raise ValueError("Invalid test_size. Please provide a float between 0 and 1 or an int.")
    #     #     if train_end + test_end > len(dataset_shuffled):
    #     #         raise ValueError("train_size + test_size cannot exceed the total number of samples.")
    #     # else:
    #     #     test_end = len(dataset_shuffled) - train_end
    #     #
    #     # train_dataset = dataset_shuffled[:train_end]
    #     # test_dataset = dataset_shuffled[train_end : train_end + test_end]
    #     #
    #     # return {"train": train_dataset, "test": test_dataset}
