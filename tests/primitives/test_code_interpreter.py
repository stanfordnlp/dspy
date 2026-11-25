import io
import json

import pytest

import dspy
from dspy.primitives.code_interpreter import (
    DEFAULT_MAX_MESSAGE_BYTES,
    CodeExecutionError,
    CodeInterpreterError,
    _MessageSizeLimitError,
    _read_bounded_line,
    _validate_max_message_bytes,
)
from dspy.primitives.python_interpreter import PythonInterpreter


def test_code_interpreter_error_is_dspy_error():
    error = CodeInterpreterError("boom")
    assert isinstance(error, dspy.DSPyError)
    assert isinstance(error, RuntimeError)
    assert str(error) == "boom"
    assert error.args == ("boom",)

    try:
        raise error
    except RuntimeError as caught:
        assert caught is error


def test_code_execution_error_is_dspy_error():
    error = CodeExecutionError("boom")
    assert isinstance(error, dspy.DSPyError)
    assert isinstance(error, CodeInterpreterError)


@pytest.mark.parametrize("bad_value", [0, -1, -1024, True, False])
def test_max_message_bytes_validation_rejects_non_positive_and_bool(bad_value):
    with pytest.raises(ValueError, match="max_message_bytes"):
        _validate_max_message_bytes(bad_value)
    with pytest.raises(ValueError, match="max_message_bytes"):
        PythonInterpreter(deno_command=["deno"], max_message_bytes=bad_value)


def test_read_bounded_line_enforces_utf8_bytes_and_never_reads_unbounded():
    assert _read_bounded_line(io.StringIO("a" * 10 + "\n"), 10) == "a" * 10 + "\n"
    with pytest.raises(_MessageSizeLimitError, match="10 bytes"):
        _read_bounded_line(io.StringIO("a" * 11 + "\n"), 10)

    # 25 four-byte UTF-8 characters == 100 bytes; 26 == 104 bytes.
    assert _read_bounded_line(io.StringIO("😀" * 25 + "\n"), 100) == "😀" * 25 + "\n"
    with pytest.raises(_MessageSizeLimitError, match="100 bytes"):
        _read_bounded_line(io.StringIO("😀" * 26 + "\n"), 100)

    # Exact chunk-boundary line followed by newline succeeds; one byte over fails.
    chunk_size = 65536
    assert len(_read_bounded_line(io.StringIO("x" * chunk_size + "\n"), chunk_size)) == chunk_size + 1
    with pytest.raises(_MessageSizeLimitError):
        _read_bounded_line(io.StringIO("x" * (chunk_size + 1) + "\n"), chunk_size)

    # Unbounded stream with no newline stops reading immediately after exceeding max_message_bytes.
    class _InfiniteNoNewlineStream:
        def __init__(self):
            self.chars_read = 0

        def readline(self, limit: int = -1) -> str:
            assert limit > 0, "bounded reader must pass a positive limit to readline"
            self.chars_read += limit
            return "a" * limit

    stream = _InfiniteNoNewlineStream()
    with pytest.raises(_MessageSizeLimitError, match="128 bytes"):
        _read_bounded_line(stream, 128)
    assert stream.chars_read <= 129

    # None disables the bound.
    assert _read_bounded_line(io.StringIO("x" * 200 + "\n"), None) == "x" * 200 + "\n"


def test_python_interpreter_oversized_response_terminates_process_and_ends_session():
    class _FakeProcess:
        def __init__(self, stdout_stream):
            self.stdin = io.StringIO()
            self.stdout = stdout_stream
            self.stderr = io.StringIO()
            self.terminated = False
            self.killed = False
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.killed = True
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

    default_interp = PythonInterpreter(deno_command=["deno"])
    assert default_interp.max_message_bytes == DEFAULT_MAX_MESSAGE_BYTES

    payload = json.dumps({"jsonrpc": "2.0", "result": {"output": "😀" * 100}, "id": "req"}) + "\n"
    interp = PythonInterpreter(deno_command=["deno"], max_message_bytes=128)
    proc = _FakeProcess(io.StringIO(payload))
    interp.deno_process = proc

    with pytest.raises(CodeInterpreterError, match="max_message_bytes \\(128 bytes\\)") as exc_info:
        interp._read_response_line("during execution")
    assert type(exc_info.value) is CodeInterpreterError
    assert proc.terminated
    assert interp._session_ended
    with pytest.raises(CodeInterpreterError, match="session has ended"):
        interp.execute("1 + 1")
