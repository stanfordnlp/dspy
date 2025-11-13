import copy
import os

import pytest

from tests.test_utils.server import litellm_test_server, read_litellm_test_server_request_logs  # noqa: F401

SKIP_DEFAULT_FLAGS = ["reliability", "extra", "llm_call"]


@pytest.fixture(autouse=True)
def clear_settings():
    """Ensures that the settings are cleared after each test."""

    yield

    import dspy
    from dspy.dsp.utils.settings import DEFAULT_CONFIG

    dspy.configure(**copy.deepcopy(DEFAULT_CONFIG), inherit_config=False)


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
            return

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
