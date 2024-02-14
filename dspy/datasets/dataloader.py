from dspy.datasets import Dataset

from typing import Union, List
from datasets import load_dataset, ReadInstruction

class DataLoader(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _process_dataset(
        self, 
        dataset: Dataset,
        fields: List[str] = None
    ):
        train_split_size = self.train_size if self.train_size else 0
        dev_split_size = self.dev_size if self.dev_size else 0
        test_split_size = self.test_size if self.test_size else 0

        if isinstance(train_split_size, float):
            train_split_size = int(len(dataset) * train_split_size)
        
        if train_split_size:
            tmp_dataset = dataset.train_test_split(test_size=(dev_split_size+test_split_size))
            train_dataset = tmp_dataset["train"]
            dataset = tmp_dataset["test"]

        if isinstance(dev_split_size, float):
            dev_split_size = int(len(dataset) * dev_split_size)

        if isinstance(test_split_size, float):
            test_split_size = int(len(dataset) * test_split_size)

        if dev_split_size or test_split_size:
            tmp_dataset = dataset.train_test_split(test_size=dev_split_size)
            dev_dataset = tmp_dataset["train"]
            test_dataset = tmp_dataset["test"]

        if train_split_size:
            self._train = [{field:row[field] for field in fields} for row in train_dataset]
        
        if dev_split_size:
            self._dev = [{field:row[field] for field in fields} for row in dev_dataset]
        
        if test_split_size:
            self._test = [{field:row[field] for field in fields} for row in test_dataset]

    def from_huggingface(
        self,
        dataset_name: str,
        fields: List[str] = None,
        splits: Union[str, List[str]] = None,
        revision: str = None,
    ):
        dataset = None
        if splits:
            if isinstance(splits, str):
                splits = [splits]
            
            try:
                ri = ReadInstruction(splits[0])
                for split in splits[1:]:
                    ri += ReadInstruction(split)
                dataset = load_dataset(dataset_name, split=ri, revision=revision)
            except:
                raise ValueError("Invalid split name provided. Please provide a valid split name or list of split names.")
        else:
            dataset = load_dataset(dataset_name, revision=revision)
            if len(dataset.keys())==1:
                split_name = next(iter(dataset.keys()))
                dataset = dataset[split_name]

            else:
                raise ValueError("No splits provided and dataset has more than one split. At this moment multiple splits will be concatenated into one single split.")
        
        if not fields:
            fields = list(dataset.features)    

        self._process_dataset(dataset, fields)

    def from_csv(self, file_path:str, fields: List[str] = None):
        dataset = load_dataset("csv", data_files=file_path)["train"]
        
        if not fields:
            fields = list(dataset.features)
        
        self._process_dataset(dataset, fields)