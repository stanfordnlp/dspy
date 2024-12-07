import random
from collections.abc import Mapping
from typing import List, Tuple, Union, Optional

import pandas as pd
from datasets import load_dataset

import dspy
from dspy.datasets.dataset import Dataset

class DataLoader(Dataset):
    def __init__(self):
        pass

    def _load_dataset(self, dataset_name: str, *args, fields: Optional[Tuple[str]] = None, input_keys: Tuple[str] = (), **kwargs) -> Union[Mapping[str, List[dspy.Example]], List[dspy.Example]]:
        if fields and not isinstance(fields, tuple):
            raise ValueError("Invalid fields provided. Please provide a tuple of fields.")

        if not isinstance(input_keys, tuple):
            raise ValueError("Invalid input keys provided. Please provide a tuple of input keys.")

        dataset = load_dataset(dataset_name, *args, **kwargs)

        if isinstance(dataset, list) and isinstance(kwargs.get("split"), list):
            dataset = {split_name: dataset[idx] for idx, split_name in enumerate(kwargs["split"])}

        return self._process_dataset(dataset, fields, input_keys)

    def _process_dataset(self, dataset, fields: Optional[Tuple[str]], input_keys: Tuple[str]) -> Union[Mapping[str, List[dspy.Example]], List[dspy.Example]]:
        try:
            returned_split = {}
            for split_name in dataset.keys():
                returned_split[split_name] = self._create_examples(dataset[split_name], fields, input_keys)
            return returned_split
        except AttributeError:
            return self._create_examples(dataset, fields, input_keys)

    def _create_examples(self, dataset, fields: Optional[Tuple[str]], input_keys: Tuple[str]) -> List[dspy.Example]:
        if fields:
            return [dspy.Example({field: row[field] for field in fields}).with_inputs(*input_keys) for row in dataset]
        else:
            return [dspy.Example({field: row[field] for field in row.keys()}).with_inputs(*input_keys) for row in dataset]

    def from_huggingface(self, dataset_name: str, *args, input_keys: Tuple[str] = (), fields: Optional[Tuple[str]] = None, **kwargs) -> Union[Mapping[str, List[dspy.Example]], List[dspy.Example]]:
        return self._load_dataset(dataset_name, *args, fields=fields, input_keys=input_keys, **kwargs)

    def from_csv(self, file_path: str, fields: Optional[List[str]] = None, input_keys: Tuple[str] = ()) -> List[dspy.Example]:
        dataset = load_dataset("csv", data_files=file_path)["train"]
        if not fields:
            fields = list(dataset.features)
        return self._create_examples(dataset, fields, input_keys)

    def from_pandas(self, df: pd.DataFrame, fields: Optional[List[str]] = None, input_keys: Tuple[str] = ()) -> List[dspy.Example]:
        if fields is None:
            fields = list(df.columns)
        return self._create_examples(df.to_dict(orient="records"), fields, input_keys)

    def from_json(self, file_path: str, fields: Optional[List[str]] = None, input_keys: Tuple[str] = ()) -> List[dspy.Example]:
        dataset = load_dataset("json", data_files=file_path)["train"]
        if not fields:
            fields = list(dataset.features)
        return self._create_examples(dataset, fields, input_keys)

    def from_parquet(self, file_path: str, fields: Optional[List[str]] = None, input_keys: Tuple[str] = ()) -> List[dspy.Example]:
        dataset = load_dataset("parquet", data_files=file_path)["train"]
        if not fields:
            fields = list(dataset.features)
        return self._create_examples(dataset, fields, input_keys)

    def from_rm(self, num_samples: int, fields: List[str], input_keys: List[str]) -> List[dspy.Example]:
        try:
            rm = dspy.settings.rm
            try:
                return self._create_examples(rm.get_objects(num_samples=num_samples, fields=fields), fields, input_keys)
            except AttributeError:
                raise ValueError("Retrieval module does not support `get_objects`. Please use a different retrieval module.")
        except AttributeError:
            raise ValueError("Retrieval module not found. Please set a retrieval module using `dspy.settings.configure`.")

    def sample(self, dataset: List[dspy.Example], n: int, *args, **kwargs) -> List[dspy.Example]:
        if not isinstance(dataset, list):
            raise ValueError(f"Invalid dataset provided of type {type(dataset)}. Please provide a list of `dspy.Example`s.")
        return random.sample(dataset, n, *args, **kwargs)

    def train_test_split(self, dataset: List[dspy.Example], train_size: Union[int, float] = 0.75, test_size: Optional[Union[int, float]] = None, random_state: Optional[int] = None) -> Mapping[str, List[dspy.Example]]:
        if random_state is not None:
            random.seed(random_state)

        dataset_shuffled = dataset.copy()
        random.shuffle(dataset_shuffled)

        train_end = self._calculate_split_size(train_size, len(dataset_shuffled))
        test_end = self._calculate_split_size(test_size, len(dataset_shuffled), train_end)

        train_dataset = dataset_shuffled[:train_end]
        test_dataset = dataset_shuffled[train_end:train_end + test_end]

        return {"train": train_dataset, "test": test_dataset}

    def _calculate_split_size(self, size: Optional[Union[int, float]], total_size: int, train_end: int = 0) -> int:
        if size is not None:
            if isinstance(size, float) and (0 < size < 1):
                return int(total_size * size)
            elif isinstance(size, int):
                return size
            else:
                raise ValueError(f"Invalid size. Please provide a float between 0 and 1 or an int. Received `size`: {size}.")
        return total_size - train_end
