"""Tests for LocalProvider to ensure secure subprocess command construction."""

from unittest import mock

from dspy.clients.lm_local import LocalProvider


class TestLocalProviderCommandConstruction:
    """Test that LocalProvider correctly constructs subprocess commands."""

    @mock.patch("dspy.clients.lm_local.threading.Thread")
    @mock.patch("dspy.clients.lm_local.subprocess.Popen")
    @mock.patch("dspy.clients.lm_local.get_free_port")
    @mock.patch("dspy.clients.lm_local.wait_for_server")
    def test_command_with_spaces_in_path(self, mock_wait, mock_port, mock_popen, mock_thread):
        """Test that model paths with spaces are handled correctly."""
        # Setup
        mock_port.return_value = 8000
        mock_process = mock.Mock()
        mock_process.pid = 12345
        mock_process.stdout.readline.return_value = ""
        mock_process.poll.return_value = 0
        mock_popen.return_value = mock_process

        # Create a mock LM object with a model path containing spaces
        lm = mock.Mock(spec=[])  # spec=[] means no attributes by default
        lm.model = "/path/to/my models/llama"
        lm.launch_kwargs = {}
        lm.kwargs = {}

        # Mock sglang import
        with mock.patch.dict("sys.modules", {"sglang": mock.Mock(), "sglang.utils": mock.Mock()}):
            LocalProvider.launch(lm, launch_kwargs={})

            # Verify Popen was called with a list
            assert mock_popen.called, "Popen should have been called"
            call_args = mock_popen.call_args
            command = call_args[0][0]

            # Command should be a list
            assert isinstance(command, list), "Command should be a list to handle paths with spaces"

            # The model path should be a separate argument
            assert "--model-path" in command
            model_index = command.index("--model-path")
            # The next element should be the full model path with spaces intact
            assert command[model_index + 1] == "/path/to/my models/llama"

    @mock.patch("dspy.clients.lm_local.threading.Thread")
    @mock.patch("dspy.clients.lm_local.subprocess.Popen")
    @mock.patch("dspy.clients.lm_local.get_free_port")
    @mock.patch("dspy.clients.lm_local.wait_for_server")
    def test_command_construction_prevents_injection(self, mock_wait, mock_port, mock_popen, mock_thread):
        """Test that command construction prevents argument injection."""
        # Setup
        mock_port.return_value = 8000
        mock_process = mock.Mock()
        mock_process.pid = 12345
        mock_process.stdout.readline.return_value = ""
        mock_process.poll.return_value = 0
        mock_popen.return_value = mock_process

        # Create a mock LM object with a potentially malicious model path
        lm = mock.Mock(spec=[])  # spec=[] means no attributes by default
        lm.model = "model --trust-remote-code"
        lm.launch_kwargs = {}
        lm.kwargs = {}

        # Mock sglang import to bypass ImportError
        with mock.patch.dict("sys.modules", {"sglang": mock.Mock(), "sglang.utils": mock.Mock()}):
            LocalProvider.launch(lm, launch_kwargs={})

            # Verify Popen was called with a list, not a string
            assert mock_popen.called, "Popen should have been called"
            call_args = mock_popen.call_args
            command = call_args[0][0]

            # Command should be a list
            assert isinstance(command, list), "Command should be a list to prevent injection"

            # The model path should be a separate argument, not split incorrectly
            assert "--model-path" in command
            model_index = command.index("--model-path")
            # The next element should be the full model path, including any spaces or special chars
            assert command[model_index + 1] == "model --trust-remote-code"

    @mock.patch("dspy.clients.lm_local.threading.Thread")
    @mock.patch("dspy.clients.lm_local.subprocess.Popen")
    @mock.patch("dspy.clients.lm_local.get_free_port")
    @mock.patch("dspy.clients.lm_local.wait_for_server")
    def test_command_is_list_not_string(self, mock_wait, mock_port, mock_popen, mock_thread):
        """Test that the command is passed as a list to Popen."""
        # Setup
        mock_port.return_value = 8000
        mock_process = mock.Mock()
        mock_process.pid = 12345
        mock_process.stdout.readline.return_value = ""
        mock_process.poll.return_value = 0
        mock_popen.return_value = mock_process

        lm = mock.Mock(spec=[])  # spec=[] means no attributes by default
        lm.model = "meta-llama/Llama-2-7b"
        lm.launch_kwargs = {}
        lm.kwargs = {}

        # Mock sglang import
        with mock.patch.dict("sys.modules", {"sglang": mock.Mock(), "sglang.utils": mock.Mock()}):
            LocalProvider.launch(lm, launch_kwargs={})

            # Verify Popen was called
            assert mock_popen.called, "Popen should have been called"
            call_args = mock_popen.call_args
            command = call_args[0][0]

            # Command should be a list
            assert isinstance(command, list), "Command should be a list"

            # Verify the structure of the command
            assert command[0] == "python"
            assert command[1] == "-m"
            assert command[2] == "sglang.launch_server"
            assert "--model-path" in command
            assert "--port" in command
            assert "--host" in command
