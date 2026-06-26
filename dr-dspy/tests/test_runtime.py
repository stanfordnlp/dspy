from __future__ import annotations

import os

from dr_dspy.runtime import load_env_file


def test_load_env_file_sets_missing_values(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("DATABASE_URL=postgresql:///unit\n")

    loaded = load_env_file(env_file)

    assert loaded == env_file
    assert os.environ["DATABASE_URL"] == "postgresql:///unit"


def test_load_env_file_preserves_existing_values(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql:///existing")
    env_file = tmp_path / ".env"
    env_file.write_text("DATABASE_URL=postgresql:///unit\n")

    loaded = load_env_file(env_file)

    assert loaded == env_file
    assert os.environ["DATABASE_URL"] == "postgresql:///existing"
