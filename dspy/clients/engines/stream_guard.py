"""One stream protocol guard for native, compatibility and custom engines."""

from dspy._vendor.lm15.errors import error_class_for_code
from dspy._vendor.lm15.types import STREAM_EVENT_CLASSES
from dspy.clients.engines.lifecycle import aclosing_stream, closing_stream
from dspy.lm15 import StreamAssemblyError


def error_from_event(event, *, provider=None):
    return error_class_for_code(event.error.code)(
        event.error.message, provider=provider, provider_code=event.error.provider_code,
    )


class _Sequence:
    def __init__(self, provider):
        self.provider = provider
        self.started = False
        self.ended = False

    def accept(self, event):
        if not isinstance(event, STREAM_EVENT_CLASSES):
            raise StreamAssemblyError(f"Engine.stream yielded {type(event).__name__}, not an lm15 stream event")
        if self.ended:
            raise StreamAssemblyError("Engine stream emitted an event after its final end event")
        if event.type == "error":
            raise error_from_event(event, provider=self.provider)
        if event.type == "start":
            if self.started:
                raise StreamAssemblyError("Engine stream emitted more than one start event")
            self.started = True
        elif not self.started:
            raise StreamAssemblyError("Engine stream emitted data before its start event")
        elif event.type == "end":
            self.ended = True

    def finish(self):
        if not self.ended:
            raise StreamAssemblyError("Engine stream ended without a completion event")


def checked_stream(source, *, provider=None):
    sequence = _Sequence(provider)
    with closing_stream(source):
        for event in source:
            sequence.accept(event)
            yield event
        sequence.finish()


async def achecked_stream(source, *, provider=None):
    sequence = _Sequence(provider)
    async with aclosing_stream(source):
        async for event in source:
            sequence.accept(event)
            yield event
        sequence.finish()
