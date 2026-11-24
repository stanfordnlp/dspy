from pathlib import Path
from unittest.mock import patch

from dspy.utils.caching import default_cache_dir


def test_default_cache_dir_prefers_explicit_env(monkeypatch):
    monkeypatch.setenv("DSPY_CACHEDIR", "/custom/cache")
    assert default_cache_dir() == "/custom/cache"


def test_default_cache_dir_ignores_empty_env(monkeypatch):
    monkeypatch.setenv("DSPY_CACHEDIR", "")
    assert default_cache_dir() == str(Path.home() / ".dspy_cache")


def test_default_cache_dir_uses_home(monkeypatch):
    monkeypatch.delenv("DSPY_CACHEDIR", raising=False)
    assert default_cache_dir() == str(Path.home() / ".dspy_cache")


def test_default_cache_dir_falls_back_when_home_is_unresolvable(monkeypatch):
    """`Path.home()` raises where no home can be determined, e.g. a WebAssembly guest."""
    monkeypatch.delenv("DSPY_CACHEDIR", raising=False)
    with patch.object(Path, "home", side_effect=RuntimeError("Could not determine home directory.")):
        assert default_cache_dir() == ".dspy_cache"
