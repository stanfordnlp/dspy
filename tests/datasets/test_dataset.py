import tempfile
import uuid

import pytest

from dspy import Example
from dspy.datasets.dataset import Dataset

dummy_data = """content,question,answer
"This is content 1","What is this?","This is answer 1"
"This is content 2","What is that?","This is answer 2"
"""


class CSVDataset(Dataset):
    def __init__(self, file_path, input_keys=None, **kwargs) -> None:
        import pandas as pd
        super().__init__(input_keys=input_keys, **kwargs)
        df = pd.read_csv(file_path)
        data = df.to_dict(orient="records")
        self._train = [
            Example(**record, dspy_uuid=str(uuid.uuid4()), dspy_split="train").with_inputs(*input_keys)
            for record in data[:1]
        ]
        self._dev = [
            Example(**record, dspy_uuid=str(uuid.uuid4()), dspy_split="dev").with_inputs(*input_keys)
            for record in data[1:2]
        ]


@pytest.fixture
def csv_file():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv") as tmp_file:
        tmp_file.write(dummy_data)
        tmp_file.flush()
        yield tmp_file.name


@pytest.mark.extra
def test_input_keys(csv_file):
    dataset = CSVDataset(csv_file, input_keys=["content", "question"])
    assert dataset.train is not None

    for example in dataset.train:
        inputs = example.inputs()
        assert inputs is not None
        assert "content" in inputs
        assert "question" in inputs
        assert set(example._input_keys) == {"content", "question"}
