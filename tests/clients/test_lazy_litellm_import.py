import importlib.util
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def test_concurrent_lm_first_use_materializes_litellm_once():
    import dspy
    from dspy.clients._litellm import get_litellm

    original_litellm = sys.modules.pop("litellm", None)
    get_litellm.cache_clear()
    try:
        threads = 8
        barrier = threading.Barrier(threads)

        def supports_function_calling(_):
            barrier.wait()
            lm = dspy.LM("openai/gpt-4o-mini", cache=False, num_retries=0)
            return lm.supports_function_calling

        with ThreadPoolExecutor(max_workers=threads) as executor:
            results = list(executor.map(supports_function_calling, range(threads)))

        assert len(results) == threads
        assert all(isinstance(result, bool) for result in results)
    finally:
        get_litellm.cache_clear()
        if original_litellm is not None:
            sys.modules["litellm"] = original_litellm


def test_concurrent_litellm_first_use_serializes_materialization(monkeypatch):
    import dspy.clients._litellm as litellm_client

    class RaceSensitiveLiteLLM(types.ModuleType):
        def __init__(self):
            super().__init__("litellm")
            self._lock = threading.Lock()
            self._materializing = False
            self._materialized = False

        @property
        def completion(self):
            with self._lock:
                if self._materializing and not self._materialized:
                    raise AttributeError("module 'litellm' has no attribute 'completion'")
                if self._materialized:
                    return object()
                self._materializing = True

            time.sleep(0.02)

            with self._lock:
                self._materialized = True
                self._materializing = False
            return object()

    fake_litellm = RaceSensitiveLiteLLM()

    def require_litellm(*args, **kwargs):
        return fake_litellm

    litellm_client.get_litellm.cache_clear()
    litellm_client._configure_litellm_defaults.cache_clear()
    monkeypatch.setattr(litellm_client, "require", require_litellm)

    try:
        workers = 8
        barrier = threading.Barrier(workers)

        def worker():
            barrier.wait()
            return litellm_client.get_litellm(feature="dspy.LM")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker) for _ in range(workers)]
            results = [future.result() for future in as_completed(futures)]

        assert results == [fake_litellm] * workers
    finally:
        litellm_client.get_litellm.cache_clear()
        litellm_client._configure_litellm_defaults.cache_clear()
