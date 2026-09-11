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


def test_legacy_image_marker_preserves_detail():
    from dspy.adapters._legacy_type_markers import _legacy_content_block_to_lm_part
    from dspy.core.types import LMImagePart

    url_part = _legacy_content_block_to_lm_part(
        {"type": "image_url", "image_url": {"url": "https://example.com/chart.png", "detail": "high"}}
    )
    assert isinstance(url_part, LMImagePart)
    assert url_part.url == "https://example.com/chart.png"
    assert url_part.detail == "high"

    data_part = _legacy_content_block_to_lm_part(
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj", "detail": "low"}}
    )
    assert isinstance(data_part, LMImagePart)
    assert data_part.data == "YWJj"
    assert data_part.media_type == "image/png"
    assert data_part.detail == "low"

    no_detail_part = _legacy_content_block_to_lm_part(
        {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}}
    )
    assert isinstance(no_detail_part, LMImagePart)
    assert no_detail_part.detail is None
