import asyncio
import contextvars
import logging
import threading
from asyncio import iscoroutinefunction
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Generator

import orjson
from anyio import create_memory_object_stream, create_task_group
from anyio.streams.memory import MemoryObjectSendStream

from dspy.dsp.utils.settings import settings
from dspy.primitives.prediction import Prediction
from dspy.streaming.messages import StatusMessage, StatusMessageProvider, StatusStreamingCallback
from dspy.streaming.streaming_listener import StreamListener, find_predictor_for_stream_listeners
from dspy.utils.asyncify import asyncify

logger = logging.getLogger(__name__)


def _is_litellm_model_response_stream(value: Any) -> bool:
    from dspy.clients.engines.streaming import EngineChunk

    if isinstance(value, EngineChunk):
        return True
    cls = type(value)
    return cls.__name__ == "ModelResponseStream" and cls.__module__.startswith("litellm")


if TYPE_CHECKING:
    from dspy.primitives.module import Module


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
            converted to a sync generator backed by a background producer; close the generator (or wrap it in
            `contextlib.closing`) when you may stop consuming early, so the producer and the LM stream are released
            deterministically (see `apply_sync_streaming`).

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
        predict_id_to_listener = find_predictor_for_stream_listeners(program, stream_listeners)
    else:
        predict_id_to_listener = {}

    if is_async_program:
        program = program.acall
    elif not iscoroutinefunction(program):
        program = asyncify(program)

    callbacks = list(settings.callbacks)
    status_streaming_callback = StatusStreamingCallback(status_message_provider)
    if not any(isinstance(c, StatusStreamingCallback) for c in callbacks):
        callbacks.append(status_streaming_callback)

    async def generator(args, kwargs, stream: MemoryObjectSendStream):
        with settings.context(send_stream=stream, callbacks=callbacks, stream_listeners=stream_listeners):
            prediction = await program(*args, **kwargs)

        await stream.send(prediction)

    async def async_streamer(*args, **kwargs):
        send_stream, receive_stream = create_memory_object_stream(16)
        async with create_task_group() as tg, send_stream, receive_stream:
            tg.start_soon(generator, args, kwargs, send_stream)

            async for value in receive_stream:
                if _is_litellm_model_response_stream(value):
                    if len(predict_id_to_listener) == 0:
                        # No listeners are configured, yield the chunk directly for backwards compatibility.
                        yield value
                    else:
                        # We are receiving a chunk from the LM's response stream, delegate it to the listeners to
                        # determine if we should yield a value to the user.
                        for listener in predict_id_to_listener[value.predict_id]:
                            # In some special cases such as Citation API, it is possible that multiple listeners
                            # return values at the same time due to the chunk buffer of the listener.
                            if output := listener.receive(value):
                                yield output
                elif isinstance(value, StatusMessage):
                    yield value
                elif isinstance(value, Prediction):
                    # Flush remaining buffered tokens before yielding the Prediction instance
                    for listener in stream_listeners:
                        if final_chunk := listener.finalize():
                            yield final_chunk

                    if include_final_prediction_in_output_stream:
                        yield value
                    elif (
                        len(stream_listeners) == 0
                        or any(listener.cache_hit for listener in stream_listeners)
                        or not any(listener.stream_start for listener in stream_listeners)
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


# Matches the anyio buffer `async_streamer` uses: the producer may run at
# most this far ahead of the consumer, so abandoning a stream leaves only a
# bounded handful of chunks consumed, not the whole completion.
_SYNC_STREAM_BUFFER_SIZE = 16

# How long a producer blocked on backpressure sleeps before re-checking the
# queue. Only reached when the buffer is full — i.e. the consumer is slower
# than the stream — and it is the producer's cancellation point (#10406).
_SYNC_STREAM_BACKPRESSURE_POLL_SECONDS = 0.01


def apply_sync_streaming(async_generator: AsyncGenerator) -> Generator:
    """Convert the async streaming generator to a sync generator.

    The returned generator owns a background producer (thread + event loop +
    the upstream LM stream). Like any resource-backed generator, it releases
    them when it is closed: on exhaustion, on ``close()``, or when the last
    reference is dropped (garbage collection). Breaking out of a ``for`` loop
    while KEEPING the reference does not close a generator in Python — until
    it is closed, the producer stays parked on a full buffer (it consumes at
    most one bounded buffer past what was read, never the whole stream). Use
    ``contextlib.closing`` when consumption may stop early:

    ```python
    from contextlib import closing

    with closing(apply_sync_streaming(stream)) as sync_stream:
        for chunk in sync_stream:
            if is_enough(chunk):
                break  # closed on exit — producer released deterministically
    ```
    """
    # Bounded so the producer cannot race arbitrarily far ahead of the
    # consumer; sized like the async path's memory-object stream.
    queue = Queue(maxsize=_SYNC_STREAM_BUFFER_SIZE)
    stop_sentinel = object()  # Sentinel to signal the generator is complete
    exception_sentinel = object()

    # To propagate prediction request ID context to the child thread
    context = contextvars.copy_context()

    # Close handshake: when the consumer abandons a partially consumed
    # stream (break / close() / GC), it flips `close_requested` and cancels
    # the runner task on its own loop. Without this the producer thread kept
    # pumping the async generator to exhaustion — consuming (and paying for)
    # the rest of the LM stream and buffering it into the queue with nobody
    # reading (#10406).
    close_requested = False
    runner_ready = threading.Event()
    runner_loop: asyncio.AbstractEventLoop | None = None
    runner_task: asyncio.Task | None = None

    def put_trailing(obj):
        """Queue a sentinel without ever wedging the producer.

        On the live path the consumer is draining, so a blocked put resolves
        itself. After a close request nobody drains; evicting the oldest
        unread item is safe — the consumer has already walked away from it.
        """
        while True:
            if not close_requested:
                try:
                    queue.put(obj, timeout=0.1)
                    return
                except Full:
                    continue
            try:
                queue.put_nowait(obj)
                return
            except Full:
                try:
                    queue.get_nowait()
                except Empty:
                    pass

    def producer():
        """Runs in a background thread to fetch items asynchronously."""

        async def runner():
            nonlocal runner_loop, runner_task
            runner_loop = asyncio.get_running_loop()
            runner_task = asyncio.current_task()
            runner_ready.set()
            try:
                async for item in async_generator:
                    # Awaitable backpressure: a plain blocking put would hold
                    # the event loop and make the task uncancellable.
                    while True:
                        try:
                            queue.put_nowait(item)
                            break
                        except Full:
                            await asyncio.sleep(_SYNC_STREAM_BACKPRESSURE_POLL_SECONDS)
            except BaseException as exc:
                if isinstance(exc, asyncio.CancelledError) and close_requested:
                    # The consumer walked away — a requested shutdown, not an
                    # error to report.
                    pass
                else:
                    put_trailing((exception_sentinel, exc))
            finally:
                try:
                    await async_generator.aclose()
                except BaseException as exc:
                    # The stream is over for the consumer either way; a close
                    # complaint (e.g. a task group unwinding, see #10380) must
                    # not displace the outcome already on the queue.
                    logger.debug("Error closing the async stream: %s", exc)
                # Signal completion
                put_trailing(stop_sentinel)

        context.run(asyncio.run, runner())

    # Start the producer in a background thread
    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    # Consume items from the queue
    finished = False
    try:
        while True:
            item = queue.get()  # Block until an item is available
            if item is stop_sentinel:
                finished = True
                break
            if isinstance(item, tuple) and len(item) == 2 and item[0] is exception_sentinel:
                finished = True  # the producer already stopped on its own
                raise item[1]
            yield item
    finally:
        if not finished:
            # Abandoned mid-stream: stop the producer instead of letting it
            # drain the rest of the upstream into the void. The generator can
            # only be closed while suspended at a yield, so by now the runner
            # has published its loop and task — the wait is belt and braces.
            close_requested = True
            if runner_ready.wait(timeout=1.0) and runner_loop is not None and runner_task is not None:
                try:
                    runner_loop.call_soon_threadsafe(runner_task.cancel)
                except RuntimeError:
                    # The loop already closed — the producer finished on its own.
                    pass
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning(
                    "The background streaming producer did not stop within its shutdown "
                    "timeout; the stream may keep being consumed until it ends."
                )


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
