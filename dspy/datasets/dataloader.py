import random
from collections.abc import Mapping
from typing import List, Tuple, Union

from datasets import load_dataset

import dspy
from dspy.datasets.dataset import Dataset


class DataLoader(Dataset):
    def __init__(self,):
        pass

    def from_huggingface(
        self,
        dataset_name: str,
        *args,
        input_keys: Tuple[str] = (),
        fields: Tuple[str] = None,
        **kwargs,
    ) -> Union[Mapping[str, List[dspy.Example]], List[dspy.Example]]:
        if fields and not isinstance(fields, tuple):
            raise ValueError("Invalid fields provided. Please provide a tuple of fields.")

        if not isinstance(input_keys, tuple):
            raise ValueError("Invalid input keys provided. Please provide a tuple of input keys.")

        dataset = load_dataset(dataset_name, *args, **kwargs)
        
        if isinstance(dataset, list) and isinstance(kwargs["split"], list):
            dataset = {split_name:dataset[idx] for idx, split_name in enumerate(kwargs["split"])}

        try:
            returned_split = {}
            for split_name in dataset.keys():
                if fields:
                    returned_split[split_name] = [dspy.Example({field:row[field] for field in fields}).with_inputs(*input_keys) for row in dataset[split_name]]
                else:
                    returned_split[split_name] = [dspy.Example({field:row[field] for field in row.keys()}).with_inputs(*input_keys) for row in dataset[split_name]]

            return returned_split
        except AttributeError:
            if fields:
                return [dspy.Example({field:row[field] for field in fields}).with_inputs(*input_keys) for row in dataset]
            else:
                return [dspy.Example({field:row[field] for field in row.keys()}).with_inputs(*input_keys) for row in dataset]

    def from_csv(self, file_path:str, fields: List[str] = None, input_keys: Tuple[str] = ()) -> List[dspy.Example]:
        dataset = load_dataset("csv", data_files=file_path)["train"]
        
        if not fields:
            fields = list(dataset.features)
        
        return [dspy.Example({field:row[field] for field in fields}).with_inputs(*input_keys) for row in dataset]

    def from_json(self, file_path:str, fields: List[str] = None, input_keys: Tuple[str] = ()) -> List[dspy.Example]:
        dataset = load_dataset("json", data_files=file_path)["train"]
        
        if not fields:
            fields = list(dataset.features)
        
        return [dspy.Example({field:row[field] for field in fields}).with_inputs(*input_keys) for row in dataset]

    def from_parquet(self, file_path: str, fields: List[str] = None, input_keys: Tuple[str] = ()) -> List[dspy.Example]:
        dataset = load_dataset("parquet", data_files=file_path)["train"]

        if not fields:
            fields = list(dataset.features)

        return [dspy.Example({field: row[field] for field in fields}).with_inputs(input_keys) for row in dataset]

    def sample(
        self,
        dataset: List[dspy.Example],
        n: int,
        *args,
        **kwargs,
    ) -> List[dspy.Example]:
        if not isinstance(dataset, list):
            raise ValueError(f"Invalid dataset provided of type {type(dataset)}. Please provide a list of examples.")
        
        return random.sample(dataset, n, *args, **kwargs)

    def train_test_split(
        self,
        dataset: List[dspy.Example],
        train_size: Union[int, float] = 0.75,
        test_size: Union[int, float] = None,
        random_state: int = None,
    ) -> Mapping[str, List[dspy.Example]]:
        if random_state is not None:
            random.seed(random_state)

        dataset_shuffled = dataset.copy()
        random.shuffle(dataset_shuffled)

        if train_size is not None and isinstance(train_size, float) and (0 < train_size < 1):
            train_end = int(len(dataset_shuffled) * train_size)
        elif train_size is not None and isinstance(train_size, int):
            train_end = train_size
        else:
            raise ValueError("Invalid train_size. Please provide a float between 0 and 1 or an int.")

        if test_size is not None:
            if isinstance(test_size, float) and (0 < test_size < 1):
                test_end = int(len(dataset_shuffled) * test_size)
            elif isinstance(test_size, int):
                test_end = test_size
            else:
                raise ValueError("Invalid test_size. Please provide a float between 0 and 1 or an int.")
            if train_end + test_end > len(dataset_shuffled):
                raise ValueError("train_size + test_size cannot exceed the total number of samples.")
        else:
            test_end = len(dataset_shuffled) - train_end

        train_dataset = dataset_shuffled[:train_end]
        test_dataset = dataset_shuffled[train_end:train_end + test_end]

        return {'train': train_dataset, 'test': test_dataset}
