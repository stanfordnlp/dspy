import pydantic
import pytest

from dspy.experimental import Document


def test_document_validate_input():
    # Create a `Document` instance with valid data.
    doc = Document(data="The Earth orbits the Sun.")
    assert doc.data == "The Earth orbits the Sun."

    with pytest.raises(pydantic.ValidationError):
        # Try to create a `Document` instance with invalid type.
        Document(data=123)


def test_document_in_nested_type():
    class Wrapper(pydantic.BaseModel):
        document: Document

    doc = Document(data="Hello, world!")
    wrapper = Wrapper(document=doc)
    assert wrapper.document.data == "Hello, world!"


def test_document_with_all_fields():
    doc = Document(
        data="Water boils at 100°C at standard pressure.",
        title="Physics Facts",
        media_type="application/pdf",
        context="Laboratory conditions"
    )
    assert doc.data == "Water boils at 100°C at standard pressure."
    assert doc.title == "Physics Facts"
    assert doc.media_type == "application/pdf"
    assert doc.context == "Laboratory conditions"


def test_document_format():
    doc = Document(
        data="The sky is blue.",
        title="Color Facts",
        media_type="text/plain"
    )

    formatted = doc.format()

    assert isinstance(formatted, list)
    assert len(formatted) == 1

    doc_block = formatted[0]
    assert doc_block["type"] == "document"
    assert doc_block["source"]["type"] == "text"
    assert doc_block["source"]["media_type"] == "text/plain"
    assert doc_block["source"]["data"] == "The sky is blue."
    assert doc_block["title"] == "Color Facts"
    assert doc_block["citations"]["enabled"] is True
