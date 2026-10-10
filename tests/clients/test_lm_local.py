import threading
import time
from unittest import mock
from unittest.mock import patch

from dspy.clients.lm_local import LocalProvider, get_free_port, wait_for_server

# Launch tests patch threading.Thread. Keep the real class for the hang watchdog.
_RealThread = threading.Thread


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


def _outcome_within(seconds, fn):
    outcome = {}

    def target():
        try:
            fn()
            outcome["result"] = "returned"
        except TimeoutError:
            outcome["result"] = "timeout"
        except Exception as exc:
            outcome["result"] = f"raised {type(exc).__name__}"

    thread = _RealThread(target=target, daemon=True)
    started = time.monotonic()
    thread.start()
    thread.join(seconds)
    outcome["alive"] = thread.is_alive()
    outcome["elapsed"] = time.monotonic() - started
    return outcome


def test_wait_for_server_times_out_when_nothing_is_listening():
    port = get_free_port()
    outcome = _outcome_within(6, lambda: wait_for_server(f"http://127.0.0.1:{port}", timeout=1))

    assert not outcome["alive"]
    assert outcome["result"] == "timeout"
    assert outcome["elapsed"] < 5


@patch("dspy.clients.lm_local.threading.Thread")
@patch("dspy.clients.lm_local.subprocess.Popen")
def test_launch_raises_when_server_process_has_exited(mock_popen, mock_thread):
    process = mock.Mock()
    process.pid = 12345
    process.poll.return_value = 1
    process.returncode = 1
    process.stdout.readline.return_value = ""
    mock_popen.return_value = process

    lm = mock.Mock(spec=[])
    lm.model = "openai/some-local-model"
    lm.launch_kwargs = {}
    lm.kwargs = {}

    def launch():
        with mock.patch.dict("sys.modules", {"sglang": mock.Mock()}):
            LocalProvider.launch(lm, launch_kwargs={"timeout": 30})

    outcome = _outcome_within(6, launch)

    assert not outcome["alive"]
    assert outcome["result"] == "timeout"
    assert outcome["elapsed"] < 5
    process.kill.assert_not_called()


@patch("dspy.clients.lm_local.threading.Thread")
@patch("dspy.clients.lm_local.subprocess.Popen")
def test_launch_kills_server_when_port_never_accepts(mock_popen, mock_thread):
    process = mock.Mock()
    process.pid = 12345
    process.poll.return_value = None
    process.stdout.readline.return_value = ""
    mock_popen.return_value = process

    lm = mock.Mock(spec=[])
    lm.model = "openai/some-local-model"
    lm.launch_kwargs = {}
    lm.kwargs = {}

    def launch():
        with mock.patch.dict("sys.modules", {"sglang": mock.Mock()}):
            LocalProvider.launch(lm, launch_kwargs={"timeout": 1})

    outcome = _outcome_within(6, launch)

    assert not outcome["alive"]
    assert outcome["result"] == "timeout"
    assert outcome["elapsed"] < 5
    process.kill.assert_called_once()
