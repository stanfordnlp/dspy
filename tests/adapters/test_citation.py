import pydantic
import pytest

import dspy


def test_citation_validate_input():
    # Create a `dspy.Citations.Citation` instance with valid data.
    citation = dspy.Citations.Citation(
        cited_text="The Earth orbits the Sun.",
        document_index=0,
        start_char_index=0,
        end_char_index=23
    )
    assert citation.cited_text == "The Earth orbits the Sun."
    assert citation.document_index == 0
    assert citation.start_char_index == 0
    assert citation.end_char_index == 23
    assert citation.type == "char_location"

    with pytest.raises(pydantic.ValidationError):
        # Try to create a `dspy.Citations.Citation` instance with missing required field.
        dspy.Citations.Citation(cited_text="text")


def test_citations_in_nested_type():
    class Wrapper(pydantic.BaseModel):
        citations: dspy.Citations

    citation = dspy.Citations.Citation(
        cited_text="Hello, world!",
        document_index=0,
        start_char_index=0,
        end_char_index=13
    )
    citations = dspy.Citations(citations=[citation])
    wrapper = Wrapper(citations=citations)
    assert wrapper.citations.citations[0].cited_text == "Hello, world!"


def test_citation_with_all_fields():
    citation = dspy.Citations.Citation(
        cited_text="Water boils at 100°C.",
        document_index=1,
        document_title="Physics Facts",
        start_char_index=10,
        end_char_index=31
    )
    assert citation.cited_text == "Water boils at 100°C."
    assert citation.document_index == 1
    assert citation.document_title == "Physics Facts"
    assert citation.start_char_index == 10
    assert citation.end_char_index == 31


def test_citation_format():
    citation = dspy.Citations.Citation(
        cited_text="The sky is blue.",
        document_index=0,
        document_title="Weather Guide",
        start_char_index=5,
        end_char_index=21
    )

    formatted = citation.format()

    assert formatted["type"] == "char_location"
    assert formatted["cited_text"] == "The sky is blue."
    assert formatted["document_index"] == 0
    assert formatted["document_title"] == "Weather Guide"
    assert formatted["start_char_index"] == 5
    assert formatted["end_char_index"] == 21


def test_citations_format():
    citations = dspy.Citations(citations=[
        dspy.Citations.Citation(
            cited_text="First citation",
            document_index=0,
            start_char_index=0,
            end_char_index=14
        ),
        dspy.Citations.Citation(
            cited_text="Second citation",
            document_index=1,
            document_title="Source",
            start_char_index=20,
            end_char_index=35
        )
    ])

    formatted = citations.format()

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
            "end_char_index": 15
        }
    ]

    citations = dspy.Citations.from_dict_list(citations_data)

    assert len(citations.citations) == 1
    assert citations.citations[0].cited_text == "The sky is blue"
    assert citations.citations[0].document_title == "Weather Guide"


def test_citations_postprocessing():
    """Test that citations are properly processed in adapter postprocessing."""
    from dspy.adapters.chat_adapter import ChatAdapter
    from dspy.signatures.signature import Signature

    class CitationSignature(Signature):
        """Test signature with citations."""
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        citations: dspy.Citations = dspy.OutputField()

    adapter = ChatAdapter()

    # Mock outputs with citations - need valid parsed text in ChatAdapter format
    outputs = [{
        "text": "[[ ## answer ## ]]\nThe answer is blue.\n\n[[ ## citations ## ]]\n[]",
        "citations": [
            {
                "cited_text": "The sky is blue",
                "document_index": 0,
                "document_title": "Weather Guide",
                "start_char_index": 10,
                "end_char_index": 25
            }
        ]
    }]

    # Process with citation signature
    result = adapter._call_postprocess(
        CitationSignature,
        CitationSignature,
        outputs
    )

    # Should have Citations object in the result
    assert len(result) == 1
    assert "citations" in result[0]
    assert isinstance(result[0]["citations"], dspy.Citations)
    assert len(result[0]["citations"]) == 1
    assert result[0]["citations"][0].cited_text == "The sky is blue"


def test_citation_extraction_from_lm_response():
    """Test citation extraction from mock LM response."""
    from unittest.mock import MagicMock

    from dspy.clients.base_lm import BaseLM

    # Create a mock response with citations in new LiteLLM format
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()

    # Mock provider_specific_fields with citations (Anthropic format)
    mock_message.provider_specific_fields = {
        "citations": [
            {
                "type": "char_location",
                "cited_text": "The sky is blue",
                "document_index": 0,
                "document_title": "Weather Guide",
                "start_char_index": 10,
                "end_char_index": 25
            }
        ]
    }

    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    # Create BaseLM instance and test citation extraction
    lm = BaseLM(model="test")
    citations = lm._extract_citations_from_response(mock_response, mock_choice)

    assert citations is not None
    assert len(citations) == 1
    assert citations[0]["cited_text"] == "The sky is blue"
    assert citations[0]["document_index"] == 0
    assert citations[0]["document_title"] == "Weather Guide"
    assert citations[0]["start_char_index"] == 10
    assert citations[0]["end_char_index"] == 25
