import copy
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.test_utils.server import litellm_test_server, read_litellm_test_server_request_logs  # noqa: F401

SKIP_DEFAULT_FLAGS = ["reliability", "extra", "llm_call", "deno"]


def _close_cache(cache: Any) -> None:
    disk_cache = getattr(cache, "disk_cache", None)
    if hasattr(disk_cache, "close"):
        disk_cache.close()


@pytest.fixture(autouse=True)
def clear_settings(tmp_path: Path) -> Iterator[None]:
    """Ensure each test gets fresh DSPy settings and an isolated cache."""
    import dspy

    original_cache = dspy.cache
    dspy.configure_cache(disk_cache_dir=tmp_path / ".dspy_cache")
    try:
        yield
    finally:
        from dspy.dsp.utils.settings import DEFAULT_CONFIG

        try:
            dspy.configure(**copy.deepcopy(DEFAULT_CONFIG), inherit_config=False)
        finally:
            try:
                _close_cache(dspy.cache)
            finally:
                dspy.cache = original_cache


@pytest.fixture
def anyio_backend():
    return "asyncio"


# Taken from: https://gist.github.com/justinmklam/b2aca28cb3a6896678e2e2927c6b6a38
def pytest_addoption(parser):
    for flag in SKIP_DEFAULT_FLAGS:
        parser.addoption(
            f"--{flag}",
            action="store_true",
            default=False,
            help=f"run {flag} tests",
        )


def pytest_configure(config):
    for flag in SKIP_DEFAULT_FLAGS:
        config.addinivalue_line("markers", flag)


def pytest_collection_modifyitems(config, items):
    for flag in SKIP_DEFAULT_FLAGS:
        if config.getoption(f"--{flag}"):
            continue

        skip_mark = pytest.mark.skip(reason=f"need --{flag} option to run")
        for item in items:
            if flag in item.keywords:
                item.add_marker(skip_mark)


@pytest.fixture
def lm_for_test():
    model = os.environ.get("LM_FOR_TEST", None)
    if model is None:
        pytest.skip("LM_FOR_TEST is not set in the environment variables")
    return model
