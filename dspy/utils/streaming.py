from asyncio import iscoroutinefunction
from typing import AsyncGenerator

from anyio import create_task_group, create_memory_object_stream
from anyio.streams.memory import MemoryObjectSendStream
import ujson


def streamify(program):
    import dspy

    if not iscoroutinefunction(program):
        program = dspy.asyncify(program)

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
                if isinstance(value, dspy.Prediction):
                    return

    return streamer


async def streaming_response(streamer: AsyncGenerator):
    import dspy
    import litellm

    async for value in streamer:
        if isinstance(value, dspy.Prediction):
            data = {"prediction": value.labels().toDict()}
        elif isinstance(value, litellm.ModelResponse):
            data = {"chunk": value.json()}

        yield f"data: {ujson.dumps(data)}\n\n"
    yield "data: [DONE]\n\n"
