import tempfile
import unittest
import uuid

import pandas as pd

from dspy import Example
from dspy.datasets.dataset import Dataset

dummy_data = """content,question,answer
"This is content 1","What is this?","This is answer 1"
"This is content 2","What is that?","This is answer 2"
"""


class CSVDataset(Dataset):
    def __init__(self, file_path, input_keys=None, *args, **kwargs) -> None:
        super().__init__(input_keys=input_keys, *args, **kwargs)
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


class TestCSVDataset(unittest.TestCase):
    def test_input_keys(self):
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv") as tmp_file:
            tmp_file.write(dummy_data)
            tmp_file.flush()
            dataset = CSVDataset(tmp_file.name, input_keys=["content", "question"])
            self.assertIsNotNone(dataset.train)

            for example in dataset.train:
                inputs = example.inputs()
                self.assertIsNotNone(inputs)
                self.assertIn("content", inputs)
                self.assertIn("question", inputs)
                self.assertEqual(set(example._input_keys), {"content", "question"})


if __name__ == "__main__":
    unittest.main()
