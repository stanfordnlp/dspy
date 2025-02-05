import logging
from asyncio import iscoroutinefunction
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Dict, Optional

import litellm
import ujson
from anyio import create_memory_object_stream, create_task_group
from anyio.streams.memory import MemoryObjectSendStream
from asyncer import syncify

from dspy.dsp.utils.settings import settings
from dspy.primitives.prediction import Prediction
from dspy.utils.asyncify import asyncify
from dspy.utils.callback import BaseCallback

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.primitives.program import Module


@dataclass
class StatusMessage:
    message: str


class StatusMessageProvider:
    def tool_start_status_message(self, instance: Any):
        return f"Calling tool {instance.name}..."

    def tool_end_status_message(self, outputs: Any):
        return "Tool calling finished! Querying the LLM with tool calling results..."

    def module_start_status_message(self, instance: Any):
        pass

    def module_end_status_message(self, outputs: Any):
        pass

    def lm_start_status_message(self, instance: Any):
        pass

    def lm_end_status_message(self, outputs: Any):
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
            status_message = self.status_message_provider.tool_start_status_message(instance)
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
            status_message = self.status_message_provider.lm_start_status_message(instance)
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
            status_message = self.status_message_provider.module_start_status_message(instance)
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


def streamify(
    program: "Module",
    status_message_provider: Optional[StatusMessageProvider] = None,
) -> Callable[[Any, Any], Awaitable[Any]]:
    """
    Wrap a DSPy program so that it streams its outputs incrementally, rather than returning them
    all at once.

    Args:
        program: The DSPy program to wrap with streaming functionality.
        status_message_provider: A custom status message generator to use instead of the default one. Users can
            implement their own status message generator to customize the status messages and what module to generate
            status messages for.

    Returns:
        A function that takes the same arguments as the original program, but returns an async
            generator that yields the program's outputs incrementally.

    Example:

    ```python
    import asyncio
    import dspy

    # Create the program and wrap it with streaming functionality
    program = dspy.streamify(dspy.Predict("q->a"))

    # Use the program with streaming output
    async def use_streaming():
        return await program(q="Why did a chicken cross the kitchen?")

    output = asyncio.run(use_streaming())
    print(output)
    ```
    """
    if not iscoroutinefunction(program):
        program = asyncify(program)

    callbacks = settings.callbacks
    status_streaming_callback = StatusStreamingCallback(status_message_provider)
    if not any(isinstance(c, StatusStreamingCallback) for c in callbacks):
        callbacks.append(status_streaming_callback)
    settings.configure(callbacks=callbacks)

    async def generator(args, kwargs, stream: MemoryObjectSendStream):
        with settings.context(send_stream=stream):
            prediction = await program(*args, **kwargs)

        await stream.send(prediction)

    async def streamer(*args, **kwargs):
        send_stream, receive_stream = create_memory_object_stream(16)
        async with create_task_group() as tg, send_stream, receive_stream:
            tg.start_soon(generator, args, kwargs, send_stream)

            async for value in receive_stream:
                yield value
                if isinstance(value, Prediction):
                    return

    return streamer


async def streaming_response(streamer: AsyncGenerator) -> AsyncGenerator:
    """
    Convert a DSPy program output stream to an OpenAI-compatible output stream that can be
    used by a service as an API response to a streaming request.

    Args:
        streamer: An async generator that yields values from a DSPy program output stream.
    Returns:
        An async generator that yields OpenAI-compatible streaming response chunks.
    """
    async for value in streamer:
        if isinstance(value, Prediction):
            data = {"prediction": {k: v for k, v in value.items(include_dspy=False)}}
            yield f"data: {ujson.dumps(data)}\n\n"
        elif isinstance(value, litellm.ModelResponseStream):
            data = {"chunk": value.json()}
            yield f"data: {ujson.dumps(data)}\n\n"
        elif isinstance(value, str) and value.startswith("data:"):
            # The chunk value is an OpenAI-compatible streaming chunk value,
            # e.g. "data: {"finish_reason": "stop", "index": 0, "is_finished": True, ...}",
            # so yield it directly
            yield value
        else:
            raise ValueError(f"Unknown chunk value type: {value}")
    yield "data: [DONE]\n\n"
