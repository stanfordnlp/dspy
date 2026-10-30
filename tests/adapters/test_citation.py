import json

import pydantic
import pytest

import dspy
from dspy.experimental import Citations
from dspy.lm15 import CitationPart, Message, Response, TextPart, Usage


def test_citation_validate_input():
    citation = Citations.Citation(
        cited_text="The Earth orbits the Sun.",
        document_index=0,
        start_char_index=0,
        end_char_index=23,
        supported_text="The Earth orbits the Sun."
    )
    assert citation.cited_text == "The Earth orbits the Sun."
    assert citation.document_index == 0
    assert citation.start_char_index == 0
    assert citation.end_char_index == 23
    assert citation.type == "char_location"
    assert citation.supported_text == "The Earth orbits the Sun."

    # Providers that report no offsets still produce a valid citation.
    minimal = Citations.Citation(cited_text="text")
    assert minimal.document_index is None
    with pytest.raises(pydantic.ValidationError):
        Citations.Citation(document_index=0)


def test_citations_in_nested_type():
    class Wrapper(pydantic.BaseModel):
        citations: Citations

    citation = Citations.Citation(
        cited_text="Hello, world!",
        document_index=0,
        start_char_index=0,
        end_char_index=13,
        supported_text="Hello, world!"
    )
    citations = Citations(citations=[citation])
    wrapper = Wrapper(citations=citations)
    assert wrapper.citations.citations[0].cited_text == "Hello, world!"


def test_citation_with_all_fields():
    citation = Citations.Citation(
        cited_text="Water boils at 100°C.",
        document_index=1,
        document_title="Physics Facts",
        start_char_index=10,
        end_char_index=31,
        supported_text="Water boils at 100°C."
    )
    assert citation.cited_text == "Water boils at 100°C."
    assert citation.document_index == 1
    assert citation.document_title == "Physics Facts"
    assert citation.start_char_index == 10
    assert citation.end_char_index == 31
    assert citation.supported_text == "Water boils at 100°C."


def test_citation_format():
    citation = Citations.Citation(
        cited_text="The sky is blue.",
        document_index=0,
        document_title="Weather Guide",
        start_char_index=5,
        end_char_index=21,
        supported_text="The sky is blue."
    )

    formatted = citation.format()

    assert formatted["type"] == "char_location"
    assert formatted["cited_text"] == "The sky is blue."
    assert formatted["document_index"] == 0
    assert formatted["document_title"] == "Weather Guide"
    assert formatted["start_char_index"] == 5
    assert formatted["end_char_index"] == 21
    assert formatted["supported_text"] == "The sky is blue."


def test_citations_format():
    citations = Citations(citations=[
        Citations.Citation(
            cited_text="First citation",
            document_index=0,
            start_char_index=0,
            end_char_index=14,
            supported_text="First citation"
        ),
        Citations.Citation(
            cited_text="Second citation",
            document_index=1,
            document_title="Source",
            start_char_index=20,
            end_char_index=35,
            supported_text="Second citation"
        )
    ])

    formatted = json.loads(citations.format())

    assert isinstance(formatted, list)
    assert len(formatted) == 2
    assert formatted[0]["cited_text"] == "First citation"
    assert formatted[1]["cited_text"] == "Second citation"
    assert formatted[1]["document_title"] == "Source"


def test_citations_from_dict_list():
    citations_data = [
        {
            "cited_text": "The sky is blue",
            "document_index": 0,
            "document_title": "Weather Guide",
            "start_char_index": 0,
            "end_char_index": 15,
            "supported_text": "The sky was blue yesterday."
        }
    ]

    citations = Citations.from_dict_list(citations_data)

    assert len(citations.citations) == 1
    assert citations.citations[0].cited_text == "The sky is blue"
    assert citations.citations[0].document_title == "Weather Guide"


def test_citations_postprocessing():
    from dspy.adapters.chat_adapter import ChatAdapter
    from dspy.signatures.signature import Signature

    class CitationSignature(Signature):
        """Test signature with citations."""
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        citations: Citations = dspy.OutputField()

    adapter = ChatAdapter(native_response_types=[Citations])

    response = Response(
        id=None, model="claude", finish_reason="stop", usage=Usage(),
        message=Message.assistant([
            TextPart("[[ ## answer ## ]]\nThe answer is blue.\n\n[[ ## citations ## ]]\n[]"),
            CitationPart(text="The sky is blue", title="Weather Guide"),
        ]),
    )

    result = adapter._call_postprocess(CitationSignature.delete("citations"), CitationSignature, [response])

    assert len(result) == 1
    assert "citations" in result[0]
    assert isinstance(result[0]["citations"], Citations)
    assert len(result[0]["citations"]) == 1
    assert result[0]["citations"][0].cited_text == "The sky is blue"
    assert result[0]["citations"][0].document_title == "Weather Guide"


def test_citation_extraction_from_lm_response():
    response = Response(
        id=None, model="m", finish_reason="stop", usage=Usage(),
        message=Message.assistant([
            TextPart("answer"),
            CitationPart(text="The sky is blue", title="Weather Guide", url="https://example.com/weather"),
        ]),
    )

    citations = Citations.parse_lm_response(response)

    assert citations is not None
    assert len(citations) == 1
    assert citations[0].cited_text == "The sky is blue"
    assert citations[0].document_title == "Weather Guide"
    assert citations[0].url == "https://example.com/weather"
    assert Citations.parse_lm_response(Response(
        id=None, model="m", finish_reason="stop", usage=Usage(), message=Message.assistant("plain"),
    )) is None
