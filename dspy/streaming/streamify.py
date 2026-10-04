import asyncio
import contextvars
import copy
import logging
import threading
from asyncio import iscoroutinefunction
from collections import defaultdict
from queue import Queue
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Generator

import orjson

from dspy.dsp.utils.settings import settings
from dspy.primitives.prediction import Prediction
from dspy.streaming.messages import StatusMessage, StatusMessageProvider, StatusStreamingCallback
from dspy.streaming.streaming_listener import StreamListener, find_predictor_for_stream_listeners
from dspy.utils.asyncify import asyncify
from dspy.utils.lazy_import import require

anyio = require("anyio")

logger = logging.getLogger(__name__)


def _is_litellm_model_response_stream(value: Any) -> bool:
    from dspy.clients.engines.streaming import EngineChunk

    if isinstance(value, EngineChunk):
        return True
    cls = type(value)
    return cls.__name__ == "ModelResponseStream" and cls.__module__.startswith("litellm")


if TYPE_CHECKING:
    from anyio.streams.memory import MemoryObjectSendStream

    from dspy.primitives.module import Module


def _single_failure(exc: BaseException) -> BaseException | None:
    """The one exception inside a (possibly nested) exception group, or None
    when ``exc`` is not a group or holds several distinct failures."""
    if not hasattr(exc, "exceptions"):
        return None
    leaves: list[BaseException] = []
    pending = [exc]
    while pending:
        current = pending.pop()
        children = getattr(current, "exceptions", None)
        if children is None:
            leaves.append(current)
        else:
            pending.extend(children)
    return leaves[0] if len(leaves) == 1 else None


def streamify(
    program: "Module",
    status_message_provider: StatusMessageProvider | None = None,
    stream_listeners: list[StreamListener] | None = None,
    include_final_prediction_in_output_stream: bool = True,
    is_async_program: bool = False,
    async_streaming: bool = True,
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
        is_async_program: Whether the program is async. If `False`, the program will be wrapped with `asyncify`,
            otherwise the program will be called with `acall`.
        async_streaming: Whether to return an async generator or a sync generator. If `False`, the streaming will be
            converted to a sync generator.

    Returns:
        A function that takes the same arguments as the original program, but returns an async
            generator that yields the program's outputs incrementally.

    Examples:

    ```python
    import asyncio
    import dspy

    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
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

    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

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

    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False))

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
            if isinstance(value, dspy.Prediction):
                return_value = value
            else:
                print(value)
        return return_value

    output = asyncio.run(use_streaming())
    print(output)
    ```

    You should see the streaming chunks (in the format of `dspy.streaming.StreamResponse`) in the console output.
    """
    stream_listeners = stream_listeners or []
    if len(stream_listeners) > 0:
        # Resolve each listener's `predict`/`predict_name` against the program once, here, so an
        # ambiguous or unknown field name still raises at streamify()-wrap time rather than on the
        # first streamed call. The returned mapping is discarded: every top-level call below builds
        # its own listener copies and its own predict-id mapping (see `_fresh_stream_listeners`),
        # so concurrent calls to the returned streamer do not share the listeners' parser state.
        find_predictor_for_stream_listeners(program, stream_listeners)

    if is_async_program:
        program = program.acall
    elif not iscoroutinefunction(program):
        program = asyncify(program)

    callbacks = list(settings.callbacks)
    status_streaming_callback = StatusStreamingCallback(status_message_provider)
    if not any(isinstance(c, StatusStreamingCallback) for c in callbacks):
        callbacks.append(status_streaming_callback)

    def _fresh_stream_listeners() -> tuple[list[StreamListener], dict[int, list[StreamListener]]]:
        # Give each top-level call its own StreamListener copies instead of reusing the listeners
        # closed over above. Two overlapping calls to the same streamer would otherwise mutate the
        # identical StreamListener objects (field_start_queue, field_end_queue, stream_start,
        # stream_end, cache_hit, json_adapter_state), so one call's chunks could silently corrupt
        # another call's in-flight parser state. A shallow copy keeps the listener's class and its
        # configuration (`predict`, `predict_name`, `allow_reuse`, `adapter_identifiers`); the JSON
        # adapter state is copied explicitly and `reset()` then rebinds the base class's per-stream
        # queues and flags on the copy (a subclass that adds per-stream mutable state must rebind
        # it in `reset()` too, see `StreamListener.reset`). A listener that appears more than once
        # in `stream_listeners` is copied once per call, so repeated entries keep sharing one
        # parser as before. The predict-id mapping is rebuilt from the already-resolved
        # predictors, without walking the program again.
        call_stream_listeners = []
        call_predict_id_to_listener = defaultdict(list)
        copies: dict[int, StreamListener] = {}
        for listener in stream_listeners:
            fresh = copies.get(id(listener))
            if fresh is None:
                fresh = copy.copy(listener)
                fresh.json_adapter_state = dict(listener.json_adapter_state)
                fresh.reset()
                copies[id(listener)] = fresh
                call_predict_id_to_listener[id(fresh.predict)].append(fresh)
            call_stream_listeners.append(fresh)
        return call_stream_listeners, call_predict_id_to_listener

    async def generator(args, kwargs, stream: "MemoryObjectSendStream", call_stream_listeners: list[StreamListener]):
        with settings.context(send_stream=stream, callbacks=callbacks, stream_listeners=call_stream_listeners):
            prediction = await program(*args, **kwargs)

        await stream.send(prediction)

    async def async_streamer(*args, **kwargs):
        try:
            async for value in _stream_program(args, kwargs):
                yield value
        except BaseException as exc:
            # The task group reports failures as an exception group. One
            # failure is the program's own exception; callers wrote
            # `except dspy.LMError`, and it must catch it.
            leaf = _single_failure(exc)
            if leaf is None:
                raise
            raise leaf from None

    async def _stream_program(args, kwargs):
        # Fresh listener copies (and a fresh predict-id mapping) for THIS call only, so this call
        # does not see another concurrent call's receive/finalize mutations, and vice versa.
        call_stream_listeners, call_predict_id_to_listener = _fresh_stream_listeners()

        send_stream, receive_stream = anyio.create_memory_object_stream(16)
        async with anyio.create_task_group() as tg, send_stream, receive_stream:
            tg.start_soon(generator, args, kwargs, send_stream, call_stream_listeners)

            async for value in receive_stream:
                if _is_litellm_model_response_stream(value):
                    if len(call_predict_id_to_listener) == 0:
                        # No listeners are configured, yield the chunk directly for backwards compatibility.
                        yield value
                    else:
                        # We are receiving a chunk from the LM's response stream, delegate it to the listeners to
                        # determine if we should yield a value to the user.
                        for listener in call_predict_id_to_listener[value.predict_id]:
                            # In some special cases such as Citation API, it is possible that multiple listeners
                            # return values at the same time due to the chunk buffer of the listener.
                            if output := listener.receive(value):
                                yield output
                elif isinstance(value, StatusMessage):
                    yield value
                elif isinstance(value, Prediction):
                    # Flush remaining buffered tokens before yielding the Prediction instance
                    for listener in call_stream_listeners:
                        if final_chunk := listener.finalize():
                            yield final_chunk

                    if include_final_prediction_in_output_stream:
                        yield value
                    elif (
                        len(call_stream_listeners) == 0
                        or any(listener.cache_hit for listener in call_stream_listeners)
                        or not any(listener.stream_start for listener in call_stream_listeners)
                    ):
                        yield value
                    return
                else:
                    # This wildcard case allows for customized streaming behavior.
                    # It is useful when a users have a custom LM which returns stream chunks in a custom format.
                    # We let those chunks pass through to the user to handle them as needed.
                    yield value

    if async_streaming:
        return async_streamer
    else:

        def sync_streamer(*args, **kwargs):
            output = async_streamer(*args, **kwargs)
            return apply_sync_streaming(output)

        return sync_streamer


def apply_sync_streaming(async_generator: AsyncGenerator) -> Generator:
    """Convert the async streaming generator to a sync generator."""
    queue = Queue()  # Queue to hold items from the async generator
    stop_sentinel = object()  # Sentinel to signal the generator is complete
    exception_sentinel = object()

    # To propagate prediction request ID context to the child thread
    context = contextvars.copy_context()

    def producer():
        """Runs in a background thread to fetch items asynchronously."""

        async def runner():
            try:
                async for item in async_generator:
                    queue.put(item)
            except BaseException as exc:
                queue.put((exception_sentinel, exc))
            finally:
                # Signal completion
                queue.put(stop_sentinel)

        context.run(asyncio.run, runner())

    # Start the producer in a background thread
    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    # Consume items from the queue
    while True:
        item = queue.get()  # Block until an item is available
        if item is stop_sentinel:
            break
        if isinstance(item, tuple) and len(item) == 2 and item[0] is exception_sentinel:
            raise item[1]
        yield item


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
            data = {"prediction": dict(value.items(include_dspy=False))}
            yield f"data: {orjson.dumps(data).decode()}\n\n"
        elif _is_litellm_model_response_stream(value):
            data = {"chunk": value.json()}
            yield f"data: {orjson.dumps(data).decode()}\n\n"
        elif isinstance(value, str) and value.startswith("data:"):
            # The chunk value is an OpenAI-compatible streaming chunk value,
            # e.g. "data: {"finish_reason": "stop", "index": 0, "is_finished": True, ...}",
            # so yield it directly
            yield value
        else:
            raise ValueError(f"Unknown chunk value type: {value}")
    yield "data: [DONE]\n\n"
