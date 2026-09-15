"""Regression test for #8489: directories with `.` in name raise ValueError when save_program=True."""
import pytest

import dspy


def test_save_program_accepts_directory_with_dot_in_name(tmp_path):
    """A directory like `dspy.2/` is a valid filesystem path and must work."""
    cot = dspy.Predict("question -> answer")
    target = tmp_path / "dspy.2"
    cot.save(str(target), save_program=True)
    assert (target / "program.pkl").exists()
    assert (target / "metadata.json").exists()


def test_save_program_accepts_versioned_directory(tmp_path):
    """Common pattern: `model.v1/`, `model.exp/`, etc."""
    cot = dspy.Predict("question -> answer")
    target = tmp_path / "model.v1"
    cot.save(str(target), save_program=True)
    assert (target / "program.pkl").exists()


def test_save_program_still_rejects_json_path(tmp_path):
    """Guard: passing a .json file with save_program=True should still error."""
    cot = dspy.Predict("question -> answer")
    with pytest.raises(ValueError, match="must point to a directory"):
        cot.save(str(tmp_path / "module.json"), save_program=True)


def test_save_program_still_rejects_pkl_path(tmp_path):
    """Guard: passing a .pkl file with save_program=True should still error."""
    cot = dspy.Predict("question -> answer")
    with pytest.raises(ValueError, match="must point to a directory"):
        cot.save(str(tmp_path / "module.pkl"), save_program=True)
