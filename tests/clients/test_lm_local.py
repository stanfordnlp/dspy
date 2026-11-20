from unittest import mock
from unittest.mock import patch

from dspy.clients.lm_local import LocalProvider, wait_for_server


@patch("dspy.clients.lm_local.threading.Thread")
@patch("dspy.clients.lm_local.subprocess.Popen")
@patch("dspy.clients.lm_local.get_free_port")
@patch("dspy.clients.lm_local.wait_for_server")
def test_command_with_spaces_in_path(mock_wait, mock_port, mock_popen, mock_thread):
    mock_port.return_value = 8000
    mock_process = mock.Mock()
    mock_process.pid = 12345
    mock_process.stdout.readline.return_value = ""
    mock_process.poll.return_value = 0
    mock_popen.return_value = mock_process

    lm = mock.Mock(spec=[])
    lm.model = "/path/to/my models/llama"
    lm.launch_kwargs = {}
    lm.kwargs = {}

    with mock.patch.dict("sys.modules", {"sglang": mock.Mock(), "sglang.utils": mock.Mock()}):
        LocalProvider.launch(lm, launch_kwargs={})

        assert mock_popen.called
        call_args = mock_popen.call_args
        command = call_args[0][0]

        assert isinstance(command, list)
        assert "--model-path" in command
        model_index = command.index("--model-path")
        assert command[model_index + 1] == "/path/to/my models/llama"


@patch("dspy.clients.lm_local.threading.Thread")
@patch("dspy.clients.lm_local.subprocess.Popen")
@patch("dspy.clients.lm_local.get_free_port")
@patch("dspy.clients.lm_local.wait_for_server")
def test_command_construction_prevents_injection(mock_wait, mock_port, mock_popen, mock_thread):
    mock_port.return_value = 8000
    mock_process = mock.Mock()
    mock_process.pid = 12345
    mock_process.stdout.readline.return_value = ""
    mock_process.poll.return_value = 0
    mock_popen.return_value = mock_process

    lm = mock.Mock(spec=[])
    lm.model = "model --trust-remote-code"
    lm.launch_kwargs = {}
    lm.kwargs = {}

    with mock.patch.dict("sys.modules", {"sglang": mock.Mock(), "sglang.utils": mock.Mock()}):
        LocalProvider.launch(lm, launch_kwargs={})

        assert mock_popen.called
        call_args = mock_popen.call_args
        command = call_args[0][0]

        assert isinstance(command, list)
        assert "--model-path" in command
        model_index = command.index("--model-path")
        assert command[model_index + 1] == "model --trust-remote-code"


@patch("dspy.clients.lm_local.threading.Thread")
@patch("dspy.clients.lm_local.subprocess.Popen")
@patch("dspy.clients.lm_local.get_free_port")
@patch("dspy.clients.lm_local.wait_for_server")
def test_command_is_list_not_string(mock_wait, mock_port, mock_popen, mock_thread):
    mock_port.return_value = 8000
    mock_process = mock.Mock()
    mock_process.pid = 12345
    mock_process.stdout.readline.return_value = ""
    mock_process.poll.return_value = 0
    mock_popen.return_value = mock_process

    lm = mock.Mock(spec=[])
    lm.model = "meta-llama/Llama-2-7b"
    lm.launch_kwargs = {}
    lm.kwargs = {}

    with mock.patch.dict("sys.modules", {"sglang": mock.Mock(), "sglang.utils": mock.Mock()}):
        LocalProvider.launch(lm, launch_kwargs={})

        assert mock_popen.called
        call_args = mock_popen.call_args
        command = call_args[0][0]

        assert isinstance(command, list)
        assert command[0] == "python"
        assert command[1] == "-m"
        assert command[2] == "sglang.launch_server"
        assert "--model-path" in command
        assert "--port" in command
        assert "--host" in command


def _launch_with_mocks(mock_port, mock_popen, launch_kwargs):
    mock_port.return_value = 8000
    mock_process = mock.Mock()
    mock_process.pid = 12345
    mock_process.stdout.readline.return_value = ""
    mock_process.poll.return_value = 0
    mock_popen.return_value = mock_process

    lm = mock.Mock(spec=[])
    lm.model = "openai/meta-llama/Llama-2-7b"
    lm.launch_kwargs = {}
    lm.kwargs = {}

    with mock.patch.dict("sys.modules", {"sglang": mock.Mock(), "sglang.utils": mock.Mock()}):
        LocalProvider.launch(lm, launch_kwargs=launch_kwargs)

    return lm, mock_popen.call_args[0][0]


@patch("dspy.clients.lm_local.threading.Thread")
@patch("dspy.clients.lm_local.subprocess.Popen")
@patch("dspy.clients.lm_local.get_free_port")
@patch("dspy.clients.lm_local.wait_for_server")
def test_launch_binds_to_loopback_by_default(mock_wait, mock_port, mock_popen, mock_thread):
    lm, command = _launch_with_mocks(mock_port, mock_popen, launch_kwargs={})

    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert "0.0.0.0" not in command
    assert "--api-key" not in command
    assert lm.kwargs["api_base"] == "http://127.0.0.1:8000/v1"
    assert lm.kwargs["api_key"] == "local"
    assert mock_wait.call_args.args[0] == "http://127.0.0.1:8000"
    assert mock_wait.call_args.kwargs["api_key"] is None


@patch("dspy.clients.lm_local.threading.Thread")
@patch("dspy.clients.lm_local.subprocess.Popen")
@patch("dspy.clients.lm_local.get_free_port")
@patch("dspy.clients.lm_local.wait_for_server")
def test_launch_uses_host_from_launch_kwargs(mock_wait, mock_port, mock_popen, mock_thread):
    lm, command = _launch_with_mocks(mock_port, mock_popen, launch_kwargs={"host": "0.0.0.0"})

    assert command[command.index("--host") + 1] == "0.0.0.0"
    # A wildcard bind is still reached over loopback from this machine.
    assert lm.kwargs["api_base"] == "http://127.0.0.1:8000/v1"
    assert mock_wait.call_args.args[0] == "http://127.0.0.1:8000"


@patch("dspy.clients.lm_local.threading.Thread")
@patch("dspy.clients.lm_local.subprocess.Popen")
@patch("dspy.clients.lm_local.get_free_port")
@patch("dspy.clients.lm_local.wait_for_server")
def test_launch_uses_specific_host_for_client(mock_wait, mock_port, mock_popen, mock_thread):
    lm, command = _launch_with_mocks(mock_port, mock_popen, launch_kwargs={"host": "10.0.0.5"})

    assert command[command.index("--host") + 1] == "10.0.0.5"
    assert lm.kwargs["api_base"] == "http://10.0.0.5:8000/v1"


@patch("dspy.clients.lm_local.threading.Thread")
@patch("dspy.clients.lm_local.subprocess.Popen")
@patch("dspy.clients.lm_local.get_free_port")
@patch("dspy.clients.lm_local.wait_for_server")
def test_launch_passes_api_key_to_server_and_client(mock_wait, mock_port, mock_popen, mock_thread):
    lm, command = _launch_with_mocks(mock_port, mock_popen, launch_kwargs={"api_key": "secret-key"})

    assert command[command.index("--api-key") + 1] == "secret-key"
    assert lm.kwargs["api_key"] == "secret-key"
    assert mock_wait.call_args.kwargs["api_key"] == "secret-key"


@patch("dspy.clients.lm_local.time.sleep")
@patch("dspy.clients.lm_local.requests.get")
def test_wait_for_server_sends_api_key(mock_get, mock_sleep):
    mock_get.return_value = mock.Mock(status_code=200)

    wait_for_server("http://127.0.0.1:8000", timeout=1, api_key="secret-key")

    assert mock_get.call_args.args[0] == "http://127.0.0.1:8000/v1/models"
    assert mock_get.call_args.kwargs["headers"] == {"Authorization": "Bearer secret-key"}


@patch("dspy.clients.lm_local.threading.Thread")
@patch("dspy.clients.lm_local.subprocess.Popen")
@patch("dspy.clients.lm_local.get_free_port")
@patch("dspy.clients.lm_local.wait_for_server")
def test_launch_ipv6_wildcard_uses_ipv6_loopback(mock_wait, mock_port, mock_popen, mock_thread):
    lm, command = _launch_with_mocks(mock_port, mock_popen, launch_kwargs={"host": "::"})

    assert command[command.index("--host") + 1] == "::"
    assert lm.kwargs["api_base"] == "http://[::1]:8000/v1"
    assert mock_wait.call_args.args[0] == "http://[::1]:8000"


@patch("dspy.clients.lm_local.threading.Thread")
@patch("dspy.clients.lm_local.subprocess.Popen")
@patch("dspy.clients.lm_local.get_free_port")
@patch("dspy.clients.lm_local.wait_for_server")
def test_launch_brackets_specific_ipv6_host(mock_wait, mock_port, mock_popen, mock_thread):
    for host, client_host in (("::1", "[::1]"), ("fd00::5", "[fd00::5]")):
        lm, command = _launch_with_mocks(mock_port, mock_popen, launch_kwargs={"host": host})

        assert command[command.index("--host") + 1] == host
        assert lm.kwargs["api_base"] == f"http://{client_host}:8000/v1"
        assert mock_wait.call_args.args[0] == f"http://{client_host}:8000"


def test_dump_state_excludes_launch_api_key():
    import dspy

    lm = dspy.LM("openai/local-model", launch_kwargs={"api_key": "secret-key", "host": "127.0.0.1"})

    state = lm.dump_state()

    assert state["launch_kwargs"] == {"host": "127.0.0.1"}
    assert "secret-key" not in str(state)
    # Saving must not change the live launch configuration.
    assert lm.launch_kwargs == {"api_key": "secret-key", "host": "127.0.0.1"}
