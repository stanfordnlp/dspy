import importlib
import os
import shutil
import tempfile

import pytest

import dspy
from tests.test_utils.server import litellm_test_server, read_litellm_test_server_request_logs


@pytest.fixture()
def temporary_blank_cache_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as cache_dir_path:
        monkeypatch.setenv("DSPY_CACHEDIR", cache_dir_path)
        importlib.reload(dspy.clients)
        yield cache_dir_path


@pytest.fixture()
def temporary_populated_cache_dir(monkeypatch):
    """
    A DSPy cache directory populated with a response for the request with text "Example query"
    to the model "openai/dspy-test-model".
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    populated_cache_path = os.path.join(module_dir, "example_cache")

    with tempfile.TemporaryDirectory() as cache_dir_path:
        shutil.copytree(populated_cache_path, cache_dir_path, dirs_exist_ok=True)
        monkeypatch.setenv("DSPY_CACHEDIR", cache_dir_path)
        importlib.reload(dspy.clients)
        yield cache_dir_path


def test_lm_calls_are_cached_across_lm_instances(litellm_test_server, temporary_blank_cache_dir):
    api_base, server_log_file_path = litellm_test_server

    # Call 2 LM instances with the same model & text and verify that only one API request is sent
    # to the LiteLLM server
    lm1 = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
    )
    lm1("Example query")
    lm2 = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
    )
    lm2("Example query")
    request_logs = read_litellm_test_server_request_logs(server_log_file_path)
    assert len(request_logs) == 1

    # Call one of the LMs with new text and verify that a new API request is sent to the
    # LiteLLM server
    lm1("New query")
    request_logs = read_litellm_test_server_request_logs(server_log_file_path)
    assert len(request_logs) == 2

    # Create a new LM instance with a different model and query it twice with the original text.
    # Verify that one new API request is sent to the LiteLLM server
    lm3 = dspy.LM(
        model="openai/dspy-test-model-2",
        api_base=api_base,
        api_key="fakekey",
    )
    lm3("Example query")
    lm3("Example query")
    request_logs = read_litellm_test_server_request_logs(server_log_file_path)
    assert len(request_logs) == 3


def test_lm_calls_are_cached_across_interpreter_sessions(litellm_test_server, temporary_populated_cache_dir):
    """
    Verifies that LM calls are cached across interpreter sessions. Pytest test cases effectively
    simulate separate interpreter sessions.
    """
    api_base, server_log_file_path = litellm_test_server

    lm1 = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
    )
    lm1("Example query")

    request_logs = read_litellm_test_server_request_logs(server_log_file_path)
    assert len(request_logs) == 0
