import pydantic
import pytest

import dspy
from dspy.adapters._legacy_type_markers import _legacy_content_block_to_lm_part


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


@pytest.mark.parametrize(
    ("image_url", "expected_detail"),
    [
        ({"url": "https://example.com/chart.png", "detail": "high"}, "high"),
        ({"url": "data:image/png;base64,YWJj", "detail": "low"}, "low"),
        ({"url": "https://example.com/plain.png"}, None),
        ({"url": "https://example.com/plain.png", "detail": "original"}, None),
    ],
)
def test_legacy_image_marker_preserves_supported_detail(image_url, expected_detail):
    block = {"type": "image_url", "image_url": image_url}

    assert _legacy_content_block_to_lm_part(block).detail == expected_detail
