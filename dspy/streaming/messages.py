import asyncio
import concurrent.futures
from dataclasses import dataclass
from typing import Any

from asyncer import syncify

from dspy.dsp.utils.settings import settings
from dspy.utils.callback import BaseCallback


@dataclass
class StreamResponse:
    """Dataclass representing a single streaming chunk from a DSPy prediction.

    Attributes:
        predict_name: The name of the Predict module that produced this chunk.
        signature_field_name: The output field name in the signature this chunk belongs to.
        chunk: The text content of this streaming chunk.
        is_last_chunk: Whether this is the final chunk for the given field.

    Examples:
        ```python
        response = StreamResponse(
            predict_name="my_predict",
            signature_field_name="answer",
            chunk="Hello",
            is_last_chunk=False,
        )
        ```
    """

    predict_name: str
    signature_field_name: str
    chunk: str
    is_last_chunk: bool


@dataclass
class StatusMessage:
    """Dataclass that wraps a status message for status streaming."""

    message: str


def sync_send_to_stream(stream, message):
    """Send message to stream in a sync context, regardless of event loop state."""

    async def _send():
        await stream.send(message)

    try:
        asyncio.get_running_loop()

        # If we're in an event loop, offload to a new thread with its own event loop
        def run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(_send())
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_new_loop)
            return future.result()
    except RuntimeError:
        # Not in an event loop, safe to use a new event loop in this thread
        return syncify(_send)()


class StatusMessageProvider:
    """Provides customizable status message streaming for DSPy programs.

    This class serves as a base for creating custom status message providers. Users can subclass
    and override its methods to define specific status messages for different stages of program execution,
    each method must return a string.

    Examples:
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
    """Callback that streams status messages to a DSPy send stream during program execution.

    Hooks into DSPy's callback system to forward lifecycle events (tool start/end,
    LM start/end, module start/end) as :class:`StatusMessage` objects to the active
    stream, using a :class:`StatusMessageProvider` to generate the message text.

    Args:
        status_message_provider: Provider that generates status message strings for
            each lifecycle event. Defaults to a base :class:`StatusMessageProvider`
            which only implements tool-level messages.

    Examples:
        ```python
        import dspy

        class MyProvider(dspy.StatusMessageProvider):
            def lm_start_status_message(self, instance, inputs):
                return "Querying the LM..."

        program = dspy.streamify(
            dspy.Predict("q->a"),
            status_message_provider=MyProvider(),
        )
        ```
    """

    def __init__(self, status_message_provider: StatusMessageProvider | None = None):
        self.status_message_provider = status_message_provider or StatusMessageProvider()

    def on_tool_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """Send a status message when a tool is about to be called.

        Args:
            call_id: Unique identifier for this callback invocation.
            instance: The :class:`dspy.Tool` instance being called.
            inputs: Keyword arguments passed to the tool.
        """
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
        """Send a status message after a tool call completes.

        Args:
            call_id: Unique identifier for this callback invocation.
            outputs: The outputs returned by the tool, or ``None`` if an error occurred.
            exception: Exception raised during the tool call, or ``None`` on success.
        """
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
        """Send a status message before an LM call is made.

        Args:
            call_id: Unique identifier for this callback invocation.
            instance: The :class:`dspy.LM` instance being called.
            inputs: Keyword arguments passed to the LM.
        """
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
        """Send a status message after an LM call completes.

        Args:
            call_id: Unique identifier for this callback invocation.
            outputs: The outputs returned by the LM, or ``None`` if an error occurred.
            exception: Exception raised during the LM call, or ``None`` on success.
        """
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
        """Send a status message before a module or Predict call.

        Args:
            call_id: Unique identifier for this callback invocation.
            instance: The :class:`dspy.Module` or :class:`dspy.Predict` instance being called.
            inputs: Keyword arguments passed to the module.
        """
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
        """Send a status message after a module or Predict call completes.

        Args:
            call_id: Unique identifier for this callback invocation.
            outputs: The outputs returned by the module, or ``None`` if an error occurred.
            exception: Exception raised during the module call, or ``None`` on success.
        """
        stream = settings.send_stream
        if stream is None:
            return

        status_message = self.status_message_provider.module_end_status_message(outputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))
