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
