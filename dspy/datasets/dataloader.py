import random
from collections.abc import Mapping
from typing import Tuple, Union, Optional

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
        **kwargs,
    ) -> Dataset:
        # If dataset is a DatasetDict, there is a key for each split
        # if only a dataset is returned, assume it is a single train split
        if isinstance(dataset, list):
            if "split" not in kwargs:
                raise ValueError("Dataset provided is a list, but 'split' not provided in kwargs")

            split = kwargs["split"]
            dataset = {split_name: dataset[idx] for idx, split_name in enumerate(split)}
        elif isinstance(dataset, (datasets.Dataset, datasets.IterableDataset)):
            # If it is a dataset of iterable dataset, assume it is the train split
            dataset = {"default": dataset}

        # If Fields is None, assume we are keeping all features
        if fields is None:
            fields = tuple(list(dataset.values())[0].features)

        examples = {}
        for split, rows in dataset.items():
            examples[split] = [
                dspy.Example(**{field: row[field] for field in fields}).with_inputs(*input_keys) for row in rows
            ]

        return Dataset(data=examples)

    def from_huggingface(
        self,
        dataset_name: str,
        input_keys: Tuple[str],
        fields: Optional[Tuple[str]] = None,
        **kwargs,
    ) -> Dataset:
        dataset = load_dataset(dataset_name, **kwargs)
        return DataLoader.from_dataset(dataset=dataset, input_keys=input_keys, fields=fields, **kwargs)

    def from_csv(
        self,
        file_path: str,
        input_keys: Tuple[str],
        fields: Optional[Tuple[str]],
        **kwargs,
    ) -> Dataset:
        dataset = load_dataset("csv", data_fields=file_path, **kwargs)
        return DataLoader.from_dataset(dataset=dataset, input_keys=input_keys, fields=fields, **kwargs)

    def from_json(
        self,
        file_path: str,
        input_keys: Tuple[str],
        fields: Optional[Tuple[str]],
        **kwargs,
    ) -> Dataset:
        dataset = load_dataset("json", data_files=file_path, **kwargs)
        return DataLoader.from_dataset(dataset=dataset, input_keys=input_keys, fields=fields, **kwargs)

    def from_parquet(
        self,
        file_path: str,
        input_keys: Tuple[str],
        fields: Optional[Tuple[str]] = None,
        **kwargs,
    ) -> Dataset:
        dataset = load_dataset("parquet", data_files=file_path, **kwargs)
        return DataLoader.from_dataset(dataset=dataset, input_keys=input_keys, fields=fields, **kwargs)
