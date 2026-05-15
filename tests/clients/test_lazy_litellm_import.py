import importlib.util
import sys

import pytest


def _hide_litellm(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "litellm" or name.startswith("litellm."):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.delitem(sys.modules, "litellm", raising=False)

    from dspy.clients._litellm import get_litellm

    get_litellm.cache_clear()


def test_import_dspy_does_not_import_litellm(monkeypatch):
    monkeypatch.delitem(sys.modules, "litellm", raising=False)

    import dspy

    _ = dspy.LM
    _ = dspy.Embedder
    _ = dspy.streamify

    assert "litellm" not in sys.modules


def test_lm_litellm_use_raises_helpful_error_without_litellm(monkeypatch):
    import dspy

    _hide_litellm(monkeypatch)

    with pytest.raises(ImportError) as exc_info:
        _ = dspy.LM("openai/gpt-4o-mini").supports_function_calling

    msg = str(exc_info.value)
    assert "[litellm]" in msg
    assert "dspy.LM" in msg


def test_embedder_litellm_use_raises_helpful_error_without_litellm(monkeypatch):
    import dspy

    _hide_litellm(monkeypatch)

    with pytest.raises(ImportError) as exc_info:
        dspy.Embedder("openai/text-embedding-3-small")(["hello"])

    msg = str(exc_info.value)
    assert "[litellm]" in msg
    assert "dspy.Embedder" in msg
