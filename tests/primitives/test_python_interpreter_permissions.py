import os

import pytest

from dspy.primitives.python_interpreter import PythonInterpreter


def _allow_read_paths(interpreter: PythonInterpreter) -> list[str]:
    allow_read_arg = next(arg for arg in interpreter.deno_command if arg.startswith("--allow-read="))
    return allow_read_arg.removeprefix("--allow-read=").split(",")


def test_deno_read_permissions_include_resolved_symlink_paths(tmp_path, monkeypatch):
    real_dir = tmp_path / "real"
    link_dir = tmp_path / "link"
    real_dir.mkdir()
    try:
        os.symlink(real_dir, link_dir, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink unavailable: {exc}")

    monkeypatch.setattr(PythonInterpreter, "_get_deno_dir", staticmethod(lambda: str(link_dir)))
    interpreter = PythonInterpreter(enable_read_paths=[str(link_dir)])

    allow_read_paths = _allow_read_paths(interpreter)
    assert str(link_dir) in allow_read_paths
    assert str(real_dir.resolve()) in allow_read_paths
