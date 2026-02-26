import asyncio
import threading
from dataclasses import dataclass
from typing import Any

from dspy.dsp.utils.settings import settings
from dspy.utils.callback import BaseCallback


@dataclass
class StreamResponse:
    predict_name: str
    signature_field_name: str
    chunk: str
    is_last_chunk: bool


@dataclass
class StatusMessage:
    """Dataclass that wraps a status message for status streaming."""

    message: str


# Module-level cache of the event loop that owns the stream consumer.
# Set by streamify() so that background threads (e.g. dspy.Parallel) can
# safely schedule items onto the correct loop.
_consumer_loop: asyncio.AbstractEventLoop | None = None
_consumer_loop_lock = threading.Lock()


def _set_consumer_loop(loop: asyncio.AbstractEventLoop):
    global _consumer_loop
    with _consumer_loop_lock:
        _consumer_loop = loop


def sync_send_to_stream(stream, message):
    """Send a message to an anyio MemoryObjectSendStream from any thread."""
    try:
        asyncio.get_running_loop()
        # Inside a running event loop — send_nowait is safe here.
        stream.send_nowait(message)
    except RuntimeError:
        # No running event loop (e.g. dspy.Parallel's ThreadPoolExecutor).
        # Use call_soon_threadsafe to schedule the send on the consumer's loop.
        with _consumer_loop_lock:
            loop = _consumer_loop

        if loop is None:
            stream.send_nowait(message)
            return

        done = threading.Event()
        exc_holder = []

        def _do_send():
            try:
                stream.send_nowait(message)
            except Exception as e:
                exc_holder.append(e)
            finally:
                done.set()

        loop.call_soon_threadsafe(_do_send)
        done.wait()
        if exc_holder:
            raise exc_holder[0]


class StatusMessageProvider:
    """Provides customizable status message streaming for DSPy programs.

    This class serves as a base for creating custom status message providers. Users can subclass
    and override its methods to define specific status messages for different stages of program execution,
    each method must return a string.

    Example:
    ```python
    class MyStatusMessageProvider(StatusMessageProvider):
        def lm_start_status_message(self, instance, inputs):
            return f"Calling LM with inputs {inputs}..."

        def module_end_status_message(self, outputs):
            return f"Module finished with output: {outputs}!"

    program = dspy.streamify(dspy.Predict("q->a"), status_message_provider=MyStatusMessageProvider())
    ```
    """

    def tool_start_status_message(self, instance: Any, inputs: dict[str, Any]):
        """Status message before a `dspy.Tool` is called."""
        return f"Calling tool {instance.name}..."

    def tool_end_status_message(self, outputs: Any):
        """Status message after a `dspy.Tool` is called."""
        return "Tool calling finished! Querying the LLM with tool calling results..."

    def module_start_status_message(self, instance: Any, inputs: dict[str, Any]):
        """Status message before a `dspy.Module` or `dspy.Predict` is called."""
        pass

    def module_end_status_message(self, outputs: Any):
        """Status message after a `dspy.Module` or `dspy.Predict` is called."""
        pass

    def lm_start_status_message(self, instance: Any, inputs: dict[str, Any]):
        """Status message before a `dspy.LM` is called."""
        pass

    def lm_end_status_message(self, outputs: Any):
        """Status message after a `dspy.LM` is called."""
        pass


class StatusStreamingCallback(BaseCallback):
    def __init__(self, status_message_provider: StatusMessageProvider | None = None):
        self.status_message_provider = status_message_provider or StatusMessageProvider()

    def on_tool_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        stream = settings.send_stream
        if stream is None or instance.name == "finish":
            return

        status_message = self.status_message_provider.tool_start_status_message(instance, inputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

    def on_tool_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ):
        stream = settings.send_stream
        if stream is None or outputs == "Completed.":
            return

        status_message = self.status_message_provider.tool_end_status_message(outputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        stream = settings.send_stream
        if stream is None:
            return

        status_message = self.status_message_provider.lm_start_status_message(instance, inputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ):
        stream = settings.send_stream
        if stream is None:
            return

        status_message = self.status_message_provider.lm_end_status_message(outputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

    def on_module_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        stream = settings.send_stream
        if stream is None:
            return

        status_message = self.status_message_provider.module_start_status_message(instance, inputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

    def on_module_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ):
        stream = settings.send_stream
        if stream is None:
            return

        status_message = self.status_message_provider.module_end_status_message(outputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))
