import socket
import subprocess
import time

import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_litellm_server():
    port = _get_random_port()
    print(f"Starting CLI on port {port}")

    # Start the CLI as a subprocess
    process = subprocess.Popen(
        ["python", "-m", "your_cli_module", "--host", "127.0.0.1", "--port", str(port), "--debug"]
    )

    # Wait for the CLI to be ready by checking the port
    try:
        _wait_for_port(port)
    except TimeoutError as e:
        process.terminate()
        raise e

    # Yield the port for tests to use
    yield port

    # Terminate the CLI process after tests complete
    process.terminate()
    process.wait()


def _get_random_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_port(port, host, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.connect((host, port))
                return True
            except ConnectionRefusedError:
                time.sleep(0.1)  # Wait briefly before trying again
    raise TimeoutError(f"Server on port {port} did not become ready within {timeout} seconds.")
