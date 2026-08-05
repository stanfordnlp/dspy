import json

import pytest

import dspy
from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError, FinalOutput
from dspy.utils.callback import ACTIVE_CALL_ID, BaseCallback, with_callbacks


class RecordingCallback(BaseCallback):
    def __init__(self, name=None, observed=None):
        self.events = []
        self.name = name
        self.observed = observed

    def _record(self, event):
        self.events.append(event)
        if self.observed is not None:
            self.observed.append((self.name, event["handler"]))

    def _start(self, handler, call_id, instance, inputs):
        self._record({
            "handler": handler,
            "call_id": call_id,
            "parent_call_id": ACTIVE_CALL_ID.get(),
            "instance": instance,
            "inputs": inputs,
        })

    def _end(self, handler, call_id, outputs, exception):
        self._record({
            "handler": handler,
            "call_id": call_id,
            "parent_call_id": ACTIVE_CALL_ID.get(),
            "outputs": outputs,
            "exception": exception,
        })

    def on_interpreter_execute_start(self, call_id, instance, inputs):
        self._start("execute_start", call_id, instance, inputs)

    def on_interpreter_execute_end(self, call_id, outputs, exception):
        self._end("execute_end", call_id, outputs, exception)

    def on_interpreter_tool_call_start(self, call_id, instance, inputs):
        self._start("tool_start", call_id, instance, inputs)

    def on_interpreter_tool_call_end(self, call_id, outputs, exception):
        self._end("tool_end", call_id, outputs, exception)

    def on_interpreter_startup_start(self, call_id, instance, inputs):
        self._start("startup_start", call_id, instance, inputs)

    def on_interpreter_startup_end(self, call_id, outputs, exception):
        self._end("startup_end", call_id, outputs, exception)

    def on_interpreter_shutdown_start(self, call_id, instance, inputs):
        self._start("shutdown_start", call_id, instance, inputs)

    def on_interpreter_shutdown_end(self, call_id, outputs, exception):
        self._end("shutdown_end", call_id, outputs, exception)


def events_for(callback, handler):
    return [event for event in callback.events if event["handler"] == handler]


def test_custom_interpreter_protocol_methods_use_interpreter_callbacks():
    class CustomInterpreter:
        def __init__(self):
            self.tools = {}

        @with_callbacks
        def start(self):
            pass

        @with_callbacks
        def execute(self, code, variables=None):
            return code

        def shutdown(self):
            pass

    callback = RecordingCallback()
    interpreter = CustomInterpreter()

    with dspy.context(callbacks=[callback]):
        interpreter.start()
        assert interpreter.execute("result", {"x": 1}) == "result"

    startup_start, startup_end, execute_start, execute_end = callback.events
    assert startup_start["handler"] == "startup_start"
    assert startup_start["instance"] is interpreter
    assert startup_start["inputs"] == {}
    assert startup_end["handler"] == "startup_end"
    assert startup_end["call_id"] == startup_start["call_id"]
    assert startup_end["outputs"] is None
    assert startup_end["exception"] is None
    assert execute_start["handler"] == "execute_start"
    assert execute_start["instance"] is interpreter
    assert execute_start["inputs"] == {"code": "result", "variables": {"x": 1}}
    assert execute_end["handler"] == "execute_end"
    assert execute_end["call_id"] == execute_start["call_id"]
    assert execute_end["outputs"] == "result"
    assert execute_end["exception"] is None


def test_invoke_tool_reports_raw_result_and_propagates_exception():
    callback = RecordingCallback()
    raw_result = object()
    expected_error = RuntimeError("tool failed")

    def fail():
        raise expected_error

    interpreter = dspy.PythonInterpreter(
        tools={"succeed": lambda: raw_result, "fail": fail},
        callbacks=[callback],
    )

    assert interpreter.invoke_tool("succeed", {}) is raw_result
    with pytest.raises(RuntimeError) as exc_info:
        interpreter.invoke_tool("fail", {})

    starts = events_for(callback, "tool_start")
    ends = events_for(callback, "tool_end")
    assert starts[0]["inputs"] == {"tool_name": "succeed", "kwargs": {}}
    assert ends[0]["call_id"] == starts[0]["call_id"]
    assert ends[0]["outputs"] is raw_result
    assert ends[0]["exception"] is None
    assert ends[1]["outputs"] is None
    assert ends[1]["exception"] is expected_error
    assert exc_info.value is expected_error


def test_handle_tool_call_reports_host_exception_before_jsonrpc_conversion(monkeypatch):
    callback = RecordingCallback()
    expected_error = ValueError("invalid tool input")

    def fail():
        raise expected_error

    interpreter = dspy.PythonInterpreter(tools={"fail": fail}, callbacks=[callback])
    writes = []
    monkeypatch.setattr(interpreter, "_write_message", lambda message, context: writes.append((message, context)))

    interpreter._handle_tool_call({"id": 7, "params": {"name": "fail", "kwargs": {}}})

    end = events_for(callback, "tool_end")[0]
    response = json.loads(writes[0][0])
    assert end["outputs"] is None
    assert end["exception"] is expected_error
    assert response["id"] == 7
    assert response["error"]["data"]["type"] == "ValueError"
    assert response["error"]["message"] == "invalid tool input"


def test_unknown_tool_reports_code_interpreter_error():
    callback = RecordingCallback()
    interpreter = dspy.PythonInterpreter(callbacks=[callback])

    with pytest.raises(CodeInterpreterError, match="Unknown tool: missing") as exc_info:
        interpreter.invoke_tool("missing", {})

    end = events_for(callback, "tool_end")[0]
    assert end["outputs"] is None
    assert end["exception"] is exc_info.value


def test_shutdown_callbacks_fire_for_each_idempotent_call():
    callback = RecordingCallback()
    interpreter = dspy.PythonInterpreter(callbacks=[callback])

    interpreter.shutdown()
    interpreter.shutdown()

    starts = events_for(callback, "shutdown_start")
    ends = events_for(callback, "shutdown_end")
    assert len(starts) == len(ends) == 2
    assert [event["call_id"] for event in starts] == [event["call_id"] for event in ends]
    assert all(event["outputs"] is None and event["exception"] is None for event in ends)


@pytest.mark.deno
def test_real_interpreter_callbacks_preserve_lazy_startup_tool_nesting_and_final_output():
    observed = []
    global_callback = RecordingCallback("global", observed)
    instance_callback = RecordingCallback("instance", observed)
    with dspy.context(callbacks=[global_callback]):
        with dspy.PythonInterpreter(
            tools={"echo": lambda value: value}, callbacks=[instance_callback]
        ) as interpreter:
            assert interpreter.execute("echo(value=payload)", variables={"payload": "first"}) == "first"
            final_output = interpreter.execute("SUBMIT('done')")

    assert final_output == FinalOutput({"output": "done"})

    expected_handlers = [
        "execute_start",
        "startup_start",
        "startup_end",
        "tool_start",
        "tool_end",
        "execute_end",
        "execute_start",
        "execute_end",
        "shutdown_start",
        "shutdown_end",
    ]
    assert [event["handler"] for event in global_callback.events] == expected_handlers
    assert [event["handler"] for event in instance_callback.events] == expected_handlers
    assert observed == [
        (source, handler)
        for handler in expected_handlers
        for source in ("global", "instance")
    ]
    first_execute, startup_start, startup_end, tool_start, tool_end, first_end = instance_callback.events[:6]
    second_execute, second_end, shutdown_start, shutdown_end = instance_callback.events[6:]
    assert first_execute["inputs"] == {
        "code": "echo(value=payload)",
        "variables": {"payload": "first"},
    }
    assert startup_start["parent_call_id"] == first_execute["call_id"]
    assert tool_start["parent_call_id"] == first_execute["call_id"]
    assert startup_end["call_id"] == startup_start["call_id"]
    assert tool_end["call_id"] == tool_start["call_id"]
    assert tool_end["outputs"] == "first"
    assert first_end["call_id"] == first_execute["call_id"]
    assert first_end["outputs"] == "first"
    assert second_end["call_id"] == second_execute["call_id"]
    assert second_end["outputs"] is final_output
    assert shutdown_end["call_id"] == shutdown_start["call_id"]


@pytest.mark.deno
def test_explicit_startup_callbacks_fire_for_each_idempotent_start_call():
    callback = RecordingCallback()
    interpreter = dspy.PythonInterpreter(callbacks=[callback])

    interpreter.start()
    interpreter.start()
    interpreter.shutdown()

    starts = events_for(callback, "startup_start")
    ends = events_for(callback, "startup_end")
    assert len(starts) == len(ends) == 2
    assert all(event["inputs"] == {} for event in starts)
    assert all(event["parent_call_id"] is None for event in starts)
    assert [event["call_id"] for event in starts] == [event["call_id"] for event in ends]
    assert all(event["outputs"] is None and event["exception"] is None for event in ends)


@pytest.mark.deno
def test_real_interpreter_execute_error_is_reported_and_propagated():
    callback = RecordingCallback()
    with dspy.PythonInterpreter(callbacks=[callback]) as interpreter:
        with pytest.raises(CodeExecutionError) as exc_info:
            interpreter.execute("raise ValueError('boom')")

    execute_end = events_for(callback, "execute_end")[0]
    assert execute_end["outputs"] is None
    assert execute_end["exception"] is exc_info.value
    assert events_for(callback, "startup_end")[0]["exception"] is None
    assert events_for(callback, "shutdown_end")[0]["exception"] is None
