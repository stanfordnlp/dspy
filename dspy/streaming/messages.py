from dataclasses import dataclass
from typing import Any, Dict, Optional

from asyncer import syncify

from dspy.dsp.utils.settings import settings
from dspy.utils.callback import BaseCallback


@dataclass
class StreamResponse:
    predict_name: str
    signature_field_name: str
    chunk: str


@dataclass
class StatusMessage:
    """Dataclass that wraps a status message for status streaming."""

    message: str


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

    def tool_start_status_message(self, instance: Any, inputs: Dict[str, Any]):
        """Status message before a `dspy.Tool` is called."""
        return f"Calling tool {instance.name}..."

    def tool_end_status_message(self, outputs: Any):
        """Status message after a `dspy.Tool` is called."""
        return "Tool calling finished! Querying the LLM with tool calling results..."

    def module_start_status_message(self, instance: Any, inputs: Dict[str, Any]):
        """Status message before a `dspy.Module` or `dspy.Predict` is called."""
        pass

    def module_end_status_message(self, outputs: Any):
        """Status message after a `dspy.Module` or `dspy.Predict` is called."""
        pass

    def lm_start_status_message(self, instance: Any, inputs: Dict[str, Any]):
        """Status message before a `dspy.LM` is called."""
        pass

    def lm_end_status_message(self, outputs: Any):
        """Status message after a `dspy.LM` is called."""
        pass


class StatusStreamingCallback(BaseCallback):
    def __init__(self, status_message_provider: Optional[StatusMessageProvider] = None):
        self.status_message_provider = status_message_provider or StatusMessageProvider()

    def on_tool_start(
        self,
        call_id: str,
        instance: Any,
        inputs: Dict[str, Any],
    ):
        stream = settings.send_stream
        if stream is None or instance.name == "finish":
            return

        @syncify
        async def send_status():
            status_message = self.status_message_provider.tool_start_status_message(instance, inputs)
            if status_message:
                await stream.send(StatusMessage(status_message))

        send_status()

    def on_tool_end(
        self,
        call_id: str,
        outputs: Optional[Dict[str, Any]],
        exception: Optional[Exception] = None,
    ):
        stream = settings.send_stream
        if stream is None or outputs == "Completed.":
            return

        @syncify
        async def send_status():
            status_message = self.status_message_provider.tool_end_status_message(outputs)
            if status_message:
                await stream.send(StatusMessage(status_message))

        send_status()

    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: Dict[str, Any],
    ):
        stream = settings.send_stream
        if stream is None:
            return

        @syncify
        async def send_status():
            status_message = self.status_message_provider.lm_start_status_message(instance, inputs)
            if status_message:
                await stream.send(StatusMessage(status_message))

        send_status()

    def on_lm_end(
        self,
        call_id: str,
        outputs: Optional[Dict[str, Any]],
        exception: Optional[Exception] = None,
    ):
        stream = settings.send_stream
        if stream is None:
            return

        @syncify
        async def send_status():
            status_message = self.status_message_provider.lm_end_status_message(outputs)
            if status_message:
                await stream.send(StatusMessage(status_message))

        send_status()

    def on_module_start(
        self,
        call_id: str,
        instance: Any,
        inputs: Dict[str, Any],
    ):
        stream = settings.send_stream
        if stream is None:
            return

        @syncify
        async def send_status():
            status_message = self.status_message_provider.module_start_status_message(instance, inputs)
            if status_message:
                await stream.send(StatusMessage(status_message))

        send_status()

    def on_module_end(
        self,
        call_id: str,
        outputs: Optional[Dict[str, Any]],
        exception: Optional[Exception] = None,
    ):
        stream = settings.send_stream
        if stream is None:
            return

        @syncify
        async def send_status():
            status_message = self.status_message_provider.module_end_status_message(outputs)
            if status_message:
                await stream.send(StatusMessage(status_message))

        send_status()
