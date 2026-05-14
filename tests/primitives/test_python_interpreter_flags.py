import pytest

from dspy.primitives.python_interpreter import PythonInterpreter


def test_enable_read_paths_reject_comma(tmp_path):
    path = tmp_path / "sales,2025.csv"

    with pytest.raises(ValueError, match=r"enable_read_paths.*comma"):
        PythonInterpreter(enable_read_paths=[path])


def test_enable_write_paths_reject_comma(tmp_path):
    path = tmp_path / "report,final.txt"

    with pytest.raises(ValueError, match=r"enable_write_paths.*comma"):
        PythonInterpreter(enable_write_paths=[path])
