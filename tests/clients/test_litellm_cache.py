import importlib
import shutil
import tempfile
from unittest.mock import patch

import pytest

import dspy
from tests.test_utils.server import read_litellm_test_server_request_logs


@pytest.fixture()
def temporary_blank_cache_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as cache_dir_path:
        monkeypatch.setenv("DSPY_CACHEDIR", cache_dir_path)
        importlib.reload(dspy.clients)
        dspy.configure_cache(enable_memory_cache=True, enable_disk_cache=False, enable_litellm_cache=True)
        yield cache_dir_path
        dspy.configure_cache(enable_memory_cache=True, enable_disk_cache=True, enable_litellm_cache=False)


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


def test_lm_calls_are_cached_in_memory_when_expected(litellm_test_server, temporary_blank_cache_dir):
    api_base, server_log_file_path = litellm_test_server

    lm1 = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
    )
    lm1("Example query")
    # Remove the disk cache, after which the LM must rely on in-memory caching
    shutil.rmtree(temporary_blank_cache_dir)
    lm1("Example query2")
    lm1("Example query2")
    lm1("Example query2")
    lm1("Example query2")

    request_logs = read_litellm_test_server_request_logs(server_log_file_path)
    assert len(request_logs) == 2


def test_lm_calls_skip_in_memory_cache_if_key_not_computable():
    with patch("litellm.completion") as mock_litellm_completion:

        class NonJsonSerializable:
            pass

        lm = dspy.LM(
            model="fakemodel/fakemodel",
            non_json_serializable=NonJsonSerializable(),
        )
        lm("Example query")
        lm("Example query")

        assert mock_litellm_completion.call_count == 2


def test_lms_called_expected_number_of_times_for_cache_key_generation_failures():
    with pytest.raises(RuntimeError), patch("litellm.completion") as mock_completion:
        mock_completion.side_effect = RuntimeError("Mocked exception")
        lm = dspy.LM(
            model="openai/dspy-test-model",
            api_base="fakebase",
            api_key="fakekey",
        )
        lm("Do not retry")

    assert mock_completion.call_count == 1
