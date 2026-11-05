import pydantic

import dspy


def test_basic_extract_custom_type_from_annotation():
    class Event(dspy.Type):
        event_name: str
        start_date_time: str
        end_date_time: str | None
        location: str | None

    class ExtractEvent(dspy.Signature):
        """Extract all events from the email content."""

        email: str = dspy.InputField()
        event: Event = dspy.OutputField()

    assert dspy.Type.extract_custom_type_from_annotation(ExtractEvent.output_fields["event"].annotation) == [Event]

    class ExtractEvents(dspy.Signature):
        """Extract all events from the email content."""

        email: str = dspy.InputField()
        events: list[Event] = dspy.OutputField()

    assert dspy.Type.extract_custom_type_from_annotation(ExtractEvents.output_fields["events"].annotation) == [Event]


def test_extract_custom_type_from_annotation_with_nested_type():
    class Event(dspy.Type):
        event_name: str
        start_date_time: str
        end_date_time: str | None
        location: str | None

    class EventIdentifier(dspy.Type):
        model_config = pydantic.ConfigDict(frozen=True)  # Make it hashable
        event_id: str
        event_name: str

    class ExtractEvents(dspy.Signature):
        """Extract all events from the email content."""

        email: str = dspy.InputField()
        events: list[dict[EventIdentifier, Event]] = dspy.OutputField()

    assert dspy.Type.extract_custom_type_from_annotation(ExtractEvents.output_fields["events"].annotation) == [
        EventIdentifier,
        Event,
    ]


def test_custom_input_preserves_text_continuation():
    from dspy.lm15 import ContinuationState, TextPart

    source = TextPart(
        "annotated",
        continuation=(ContinuationState(provider="example", kind="opaque", data={"id": 7}),),
    )

    class Annotated(dspy.Type):
        def format(self):
            return [source]

    class QA(dspy.Signature):
        context: Annotated = dspy.InputField()
        answer: str = dspy.OutputField()

    lm = dspy.utils.DummyLM([{"answer": "ok"}])
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        assert dspy.Predict(QA)(context=Annotated()).answer == "ok"

    assert source in lm.history[0]["request"].messages[-1].parts
