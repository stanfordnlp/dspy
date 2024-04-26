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
        dataset: Union[
            datasets.Dataset,
            datasets.DatasetDict,
            datasets.IterableDataset,
            datasets.IterableDatasetDict,
            dict,
            list,
        ],
        input_keys: Tuple[str],
        fields: Optional[Tuple[str]],
        split: list[str],
        dev_percentage: float = 0.6,
    ) -> Dataset:
        # If dataset is a DatasetDict, there is a key for each split
        # if only a dataset is returned, assume it is a single train split
        split = [s.split("[")[0] if "[" in s else s for s in split]
        if isinstance(dataset, list):
            dataset = {split_name: dataset[idx] for idx, split_name in enumerate(split)}
        elif isinstance(dataset, (datasets.Dataset, datasets.IterableDataset)):
            # If it is a dataset of iterable dataset, assume it is the train split
            dataset = {"train": dataset}

        # If Fields is None, assume we are keeping all features
        if fields is None:
            fields = tuple(dataset[split[0]].features)

        examples = {"train": [], "test": [], "dev": []}
        for split, rows in dataset.items():
            if split == "train" and dev_percentage > 0.0:
                train_size = int((1 - dev_percentage) * len(rows))

                examples["train"] = [
                    dspy.Example(**{field: rows[row_idx][field] for field in fields}).with_inputs(*input_keys)
                    for row_idx in range(0, train_size)
                ]
                examples["dev"] = [
                    dspy.Example(**{field: rows[row_idx][field] for field in fields}).with_inputs(*input_keys)
                    for row_idx in range(train_size, len(rows))
                ]
            else:
                examples["test"] = [
                    dspy.Example(**{field: row[field] for field in fields}).with_inputs(*input_keys) for row in rows
                ]

        return Dataset.load(**examples)

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

        dataset = load_dataset(dataset_name, split=split, **kwargs)
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
            split = ["train", "test"]

        dataset = load_dataset("csv", split=split, data_fields=file_path, **kwargs)
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
            split = ["train", "test"]

        dataset = load_dataset("json", split=split, data_files=file_path, **kwargs)
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
            split = ["train", "test"]

        dataset = load_dataset("parquet", split=split, data_files=file_path, **kwargs)
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
