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


# Booting a Deno/Pyodide interpreter costs ~2.5s, which dominates the deno-marked
# test suite. Tests that only exercise execute() semantics share one interpreter
# per pytest process and restore its global namespace between tests. Tests that
# configure Deno *process-level* permissions (enable_env_vars/enable_read_paths/
# enable_write_paths/enable_network_access/deno_command/sync_files) or exercise
# session lifecycle (shutdown, process death, protocol failure) must keep
# creating their own instances.
_POOL_SETUP_CODE = """
def _pool_reset():
    g = globals()
    for name in [n for n in g if n not in _POOL_SAVED]:
        del g[name]
    g.update(_POOL_SAVED)

_POOL_SAVED = dict(globals())
_POOL_SAVED["_POOL_SAVED"] = _POOL_SAVED
"""


@pytest.fixture(scope="session")
def _interpreter_pool() -> Iterator[dict[str, Any]]:
    holder: dict[str, Any] = {"interpreter": None}
    yield holder
    if holder["interpreter"] is not None:
        holder["interpreter"].shutdown()


@pytest.fixture
def pooled_interpreter(_interpreter_pool: dict[str, Any]):
    """A shared PythonInterpreter with per-test namespace restoration.

    Per-test tools and output_fields may be set by mutating ``.tools`` /
    ``.output_fields`` and clearing ``_tools_registered`` (the same protocol
    RLM uses for caller-owned interpreters); teardown restores both along
    with the sandbox's global namespace.
    """
    from dspy.primitives.code_interpreter import CodeInterpreterError
    from dspy.primitives.python_interpreter import PythonInterpreter

    interpreter = _interpreter_pool["interpreter"]
    if interpreter is None:
        interpreter = PythonInterpreter()
        interpreter.execute(_POOL_SETUP_CODE)
        _interpreter_pool["interpreter"] = interpreter

    yield interpreter

    try:
        interpreter.tools.clear()
        interpreter.output_fields = None
        interpreter._tools_registered = False
        interpreter.execute("_pool_reset()")
    except CodeInterpreterError:
        # The test that just ran killed the session. Surface that loudly on
        # this test and boot a fresh interpreter for the next consumer.
        _interpreter_pool["interpreter"] = None
        interpreter.shutdown()
        raise


@pytest.fixture
def configure_pooled_interpreter(pooled_interpreter):
    """Configure the pooled interpreter with per-test tools/output_fields."""

    def configure(tools: dict[str, Any] | None = None, output_fields: list[dict[str, Any]] | None = None):
        if tools:
            pooled_interpreter.tools.update(tools)
        if output_fields is not None:
            pooled_interpreter.output_fields = output_fields
        pooled_interpreter._tools_registered = False
        return pooled_interpreter

    return configure
