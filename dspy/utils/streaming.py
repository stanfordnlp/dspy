import logging
from asyncio import iscoroutinefunction
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Optional

import litellm
import ujson
from anyio import create_memory_object_stream, create_task_group
from anyio.streams.memory import MemoryObjectSendStream
from asyncer import syncify

from dspy.dsp.utils.settings import settings
from dspy.primitives.prediction import Prediction
from dspy.primitives.program import Module
from dspy.utils.asyncify import asyncify
from dspy.utils.callback import BaseCallback

logger = logging.getLogger(__name__)


@dataclass
class StatusMessage:
    message: str


class StatusStreamingCallback(BaseCallback):
    def on_tool_start(
        self,
        call_id: str,
        instance: Any,
        inputs: Dict[str, Any],
    ):
        stream = settings.send_stream
        if stream is None:
            return

        @syncify
        async def send_tool_calling_status():
            await stream.send(StatusMessage(f"Calling tool {instance.name}..."))

        send_tool_calling_status()

    def on_tool_end(
        self,
        call_id: str,
        outputs: Optional[Dict[str, Any]],
        exception: Optional[Exception] = None,
    ):
        stream = settings.send_stream
        if stream is None:
            return

        @syncify
        async def send_tool_calling_status():
            await stream.send(StatusMessage("Tool calling finished! Querying the LLM with tool calling results..."))

        send_tool_calling_status()


def streamify(program: Module) -> Callable[[Any, Any], Awaitable[Any]]:
    """
    Wrap a DSPy program so that it streams its outputs incrementally, rather than returning them
    all at once.

    Args:
        program: The DSPy program to wrap with streaming functionality.

    Returns:
        A function that takes the same arguments as the original program, but returns an async
            generator that yields the program's outputs incrementally.

    Example:

    ```python
    class TestSignature(dspy.Signature):
        input_text: str = dspy.InputField()
        output_text: str = dspy.OutputField()

    # Create the program and wrap it with streaming functionality
    program = dspy.streamify(dspy.Predict(TestSignature))

    # Use the program with streaming output
    async def use_streaming():
        output_stream = program(input_text="Test")
        async for value in output_stream:
            print(value)  # Print each streamed value incrementally
    ```
    """
    import dspy

    if not iscoroutinefunction(program):
        program = asyncify(program)

    callbacks = dspy.settings.callbacks
    if not any(isinstance(c, StatusStreamingCallback) for c in callbacks):
        callbacks.append(StatusStreamingCallback())
    dspy.settings.configure(callbacks=callbacks)

    async def generator(args, kwargs, stream: MemoryObjectSendStream):
        with dspy.settings.context(send_stream=stream):
            prediction = await program(*args, **kwargs)

        await stream.send(prediction)

    async def streamer(*args, **kwargs):
        send_stream, receive_stream = create_memory_object_stream(16)
        async with create_task_group() as tg, send_stream, receive_stream:
            tg.start_soon(generator, args, kwargs, send_stream)

            async for value in receive_stream:
                yield value
                if isinstance(value, StatusMessage):
                    logger.info(value.message)
                if isinstance(value, Prediction):
                    return

    return streamer


async def get_final_output(streamer: AsyncGenerator) -> Prediction:
    prediction = None
    async for value in streamer:
        if isinstance(value, Prediction):
            prediction = value

    return prediction


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
