import pydantic
import pytest

from dspy.experimental import Document
from dspy.lm15 import DocumentPart, TextPart


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

    # Plain text is read by the model as text, framed by its title.
    assert formatted == [TextPart("Title: Color Facts\n"), TextPart("The sky is blue.")]


def test_document_format_pdf_uses_base64_source():
    doc = Document(
        data="cGRm",
        media_type="application/pdf",
    )

    [doc_part] = doc.format()

    # PDF documents become one inline document part.
    assert doc_part == DocumentPart(data="cGRm", media_type="application/pdf")
