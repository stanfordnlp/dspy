import json
import os

from dspy.clients.utils_finetune import save_data


def test_save_data_writes_jsonl_to_finetune_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("DSPY_FINETUNEDIR", str(tmp_path))
    data = [
        {"messages": [{"role": "user", "content": "What is 1 + 1?"}]},
        {"messages": [{"role": "assistant", "content": "2"}]},
    ]

    file_path = save_data(data)

    assert os.path.isabs(file_path)
    assert os.path.dirname(file_path) == os.path.abspath(str(tmp_path))
    assert file_path.endswith(".jsonl")

    with open(file_path, encoding="utf-8") as f:
        assert [json.loads(line) for line in f] == data
