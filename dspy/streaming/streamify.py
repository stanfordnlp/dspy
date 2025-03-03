import logging
from asyncio import iscoroutinefunction
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, List, Optional

import litellm
import ujson
from anyio import create_memory_object_stream, create_task_group
from anyio.streams.memory import MemoryObjectSendStream
from litellm import ModelResponseStream

from dspy.dsp.utils.settings import settings
from dspy.primitives.prediction import Prediction
from dspy.streaming.messages import StatusMessage, StatusMessageProvider, StatusStreamingCallback
from dspy.streaming.streaming_listener import StreamListener, find_predictor_for_stream_listeners
from dspy.utils.asyncify import asyncify

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.primitives.program import Module


def streamify(
    program: "Module",
    status_message_provider: Optional[StatusMessageProvider] = None,
    stream_listeners: Optional[List[StreamListener]] = None,
    include_final_prediction_in_output_stream: bool = True,
) -> Callable[[Any, Any], Awaitable[Any]]:
    """
    Wrap a DSPy program so that it streams its outputs incrementally, rather than returning them
    all at once. It also provides status messages to the user to indicate the progress of the program, and users
    can implement their own status message provider to customize the status messages and what module to generate
    status messages for.

    Args:
        program: The DSPy program to wrap with streaming functionality.
        status_message_provider: A custom status message generator to use instead of the default one. Users can
            implement their own status message generator to customize the status messages and what module to generate
            status messages for.
        stream_listeners: A list of stream listeners to capture the streaming output of specific fields of sub predicts
            in the program. When provided, only the target fields in the target predict will be streamed to the user.
        include_final_prediction_in_output_stream: Whether to include the final prediction in the output stream, only
            useful when `stream_listeners` is provided. If `False`, the final prediction will not be included in the
            output stream. When the program hit cache, or no listeners captured anything, the final prediction will
            still be included in the output stream even if this is `False`.

    Returns:
        A function that takes the same arguments as the original program, but returns an async
            generator that yields the program's outputs incrementally.

    Example:

    ```python
    import asyncio
    import dspy

    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))
    # Create the program and wrap it with streaming functionality
    program = dspy.streamify(dspy.Predict("q->a"))

    # Use the program with streaming output
    async def use_streaming():
        output = program(q="Why did a chicken cross the kitchen?")
        return_value = None
        async for value in output:
            if isinstance(value, dspy.Prediction):
                return_value = value
            else:
                print(value)
        return return_value

    output = asyncio.run(use_streaming())
    print(output)
    ```

    Example with custom status message provider:
    ```python
    import asyncio
    import dspy

    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    class MyStatusMessageProvider(StatusMessageProvider):
        def module_start_status_message(self, instance, inputs):
            return f"Predicting..."

        def tool_end_status_message(self, outputs):
            return f"Tool calling finished with output: {outputs}!"

    # Create the program and wrap it with streaming functionality
    program = dspy.streamify(dspy.Predict("q->a"), status_message_provider=MyStatusMessageProvider())

    # Use the program with streaming output
    async def use_streaming():
        output = program(q="Why did a chicken cross the kitchen?")
        return_value = None
        async for value in output:
            if isinstance(value, dspy.Prediction):
                return_value = value
            else:
                print(value)
        return return_value

    output = asyncio.run(use_streaming())
    print(output)
    ```

    Example with stream listeners:

    ```python
    import asyncio
    import dspy

    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False))

    # Create the program and wrap it with streaming functionality
    predict = dspy.Predict("question->answer, reasoning")
    stream_listeners = [
        dspy.streaming.StreamListener(signature_field_name="answer"),
        dspy.streaming.StreamListener(signature_field_name="reasoning"),
    ]
    stream_predict = dspy.streamify(predict, stream_listeners=stream_listeners)

    async def use_streaming():
        output = stream_predict(
            question="why did a chicken cross the kitchen?",
            include_final_prediction_in_output_stream=False,
        )
        return_value = None
        async for value in output:
            print(value)
        return return_value

    output = asyncio.run(use_streaming())
    print(output)
    ```

    You should see the streaming chunks (in the format of `dspy.streaming.StreamResponse`) in the console output.
    """
    stream_listeners = stream_listeners or []
    if len(stream_listeners) > 0:
        predict_id_to_listener = find_predictor_for_stream_listeners(program, stream_listeners)
    else:
        predict_id_to_listener = {}

    if not iscoroutinefunction(program):
        program = asyncify(program)

    callbacks = settings.callbacks
    status_streaming_callback = StatusStreamingCallback(status_message_provider)
    if not any(isinstance(c, StatusStreamingCallback) for c in callbacks):
        callbacks.append(status_streaming_callback)

    async def generator(args, kwargs, stream: MemoryObjectSendStream):
        with settings.context(send_stream=stream, callbacks=callbacks, stream_listeners=stream_listeners):
            prediction = await program(*args, **kwargs)

        await stream.send(prediction)

    async def streamer(*args, **kwargs):
        send_stream, receive_stream = create_memory_object_stream(16)
        async with create_task_group() as tg, send_stream, receive_stream:
            tg.start_soon(generator, args, kwargs, send_stream)

            async for value in receive_stream:
                if isinstance(value, ModelResponseStream):
                    if len(predict_id_to_listener) == 0:
                        # No listeners are configured, yield the chunk directly for backwards compatibility.
                        yield value
                    else:
                        # We are receiving a chunk from the LM's response stream, delgate it to the listeners to
                        # determine if we should yield a value to the user.
                        output = None
                        for listener in predict_id_to_listener[value.predict_id]:
                            # There should be at most one listener provides a return value.
                            output = listener.receive(value) or output
                        if output:
                            yield output
                elif isinstance(value, StatusMessage):
                    yield value
                elif isinstance(value, Prediction):
                    if include_final_prediction_in_output_stream:
                        yield value
                    elif (
                        len(stream_listeners) == 0
                        or any(listener.cache_hit for listener in stream_listeners)
                        or not any(listener.stream_start for listener in stream_listeners)
                    ):
                        yield value
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
