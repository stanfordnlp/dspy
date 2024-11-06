import json
import os
import socket
import subprocess
import sys
import tempfile
import time

import pytest

from tests.test_utils.server.litellm_server import LITELLM_TEST_SERVER_LOG_FILE_PATH_ENV_VAR


@pytest.fixture()
def mock_litellm_server():
    with tempfile.TemporaryDirectory() as server_log_dir_path:
        # Create a server log file used to store request logs
        server_log_file_path = os.path.join(server_log_dir_path, "request_logs.jsonl")
        open(server_log_file_path, "a").close()

        port = _get_random_port()
        host = "127.0.0.1"
        print(f"Starting LiteLLM proxy server on port {port}")

        # Start the CLI as a subprocess
        process = subprocess.Popen(
            ["litellm", "--host", host, "--port", str(port), "--config", _get_litellm_config_path()],
            env={LITELLM_TEST_SERVER_LOG_FILE_PATH_ENV_VAR: server_log_file_path, **os.environ.copy()},
            text=True,
        )

        # Wait for the CLI to be ready by checking the port
        try:
            _wait_for_port(host=host, port=port)
        except TimeoutError as e:
            process.terminate()
            raise e

        server_url = f"http://{host}:{port}"
        yield server_url, server_log_file_path

        # Terminate the CLI process after tests complete
        process.kill()
        process.wait()


def read_server_request_logs(server_log_file_path):
    data = []
    with open(server_log_file_path, "r") as f:
        for line in f:
            # Parse each line as a JSON object
            data.append(json.loads(line))

    return data


def _get_litellm_config_path():
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(module_dir, "litellm_server_config.yaml")


def _get_random_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_port(host, port, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.connect((host, port))
                return True
            except ConnectionRefusedError:
                time.sleep(0.5)  # Wait briefly before trying again
    raise TimeoutError(f"Server on port {port} did not become ready within {timeout} seconds.")
