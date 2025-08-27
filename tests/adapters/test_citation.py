from unittest.mock import MagicMock

import dspy
from dspy.adapters.types import Citations
from dspy.signatures.signature import Signature


class CitationSignature(Signature):
    """Test signature with citations."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citations: Citations = dspy.OutputField()


def test_citation_type():
    """Test the individual Citation type."""
    citation = Citations.Citation(
        text="The sky is blue",
        source="Weather Guide",
        start=10,
        end=25
    )

    assert citation.text == "The sky is blue"
    assert citation.source == "Weather Guide"
    assert citation.start == 10
    assert citation.end == 25

    # Test formatting
    formatted = citation.format()
    assert '"The sky is blue"' in formatted
    assert "Weather Guide" in formatted


def test_citation_type_with_dict_source():
    """Test Citation with dict source."""
    citation = Citations.Citation(
        text="The sky is blue",
        source={"title": "Weather Guide", "url": "http://example.com"}
    )

    formatted = citation.format()
    assert '"The sky is blue"' in formatted
    assert "Weather Guide" in formatted
    assert "http://example.com" in formatted


def test_citations_container_type():
    """Test the Citations container type."""
    citations_data = [
        {"text": "The sky is blue", "source": "Weather Guide"},
        {"text": "Water boils at 100°C", "source": "Physics Book"}
    ]

    citations = Citations.from_dict_list(citations_data)

    assert len(citations) == 2
    assert citations[0].text == "The sky is blue"
    assert citations[1].text == "Water boils at 100°C"

    # Test iteration
    citation_texts = [c.text for c in citations]
    assert "The sky is blue" in citation_texts
    assert "Water boils at 100°C" in citation_texts


def test_citations_description():
    """Test Citations description method."""
    desc = Citations.description()
    assert "citation" in desc.lower()
    assert "source" in desc.lower()


def test_citations_format():
    """Test Citations format method."""
    citations_data = [
        {"text": "The sky is blue", "source": "Weather Guide", "start": 10, "end": 25}
    ]
    citations = Citations.from_dict_list(citations_data)

    formatted = citations.format()
    assert isinstance(formatted, list)
    assert len(formatted) == 1
    assert formatted[0]["text"] == "The sky is blue"
    assert formatted[0]["source"] == "Weather Guide"


def test_citations_validation():
    """Test Citations validation with different input formats."""
    # Test with from_dict_list
    citations1 = Citations.from_dict_list([{"text": "Hello", "source": "World"}])
    assert len(citations1) == 1

    # Test direct construction with citations list
    citations2 = Citations(citations=[Citations.Citation(text="Hello", source="World")])
    assert len(citations2) == 1


def test_citation_field_detection():
    """Test that Citations fields are properly detected by adapter."""
    from dspy.adapters.chat_adapter import ChatAdapter

    adapter = ChatAdapter()

    # Test citation field detection
    citation_fields = adapter._get_citation_output_field_names(CitationSignature)
    assert "citations" in citation_fields


def test_citation_extraction_from_lm_response():
    """Test citation extraction from mock LM response."""
    from dspy.clients.base_lm import BaseLM

    # Create a mock response with citations
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()

    # Mock provider_specific_fields with citations (Anthropic format)
    mock_message.provider_specific_fields = {
        "citations": [
            {
                "quote": "The sky is blue",
                "source": "Weather Guide",
                "start": 10,
                "end": 25
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
    assert citations[0]["text"] == "The sky is blue"
    assert citations[0]["source"] == "Weather Guide"
    assert citations[0]["start"] == 10
    assert citations[0]["end"] == 25


def test_citations_postprocessing():
    """Test that citations are properly processed in adapter postprocessing."""
    from dspy.adapters.chat_adapter import ChatAdapter

    adapter = ChatAdapter()

    # Mock outputs with citations - need valid parsed text in ChatAdapter format
    outputs = [{
        "text": "[[ ## answer ## ]]\nThe answer is blue.\n\n[[ ## citations ## ]]\n[]",
        "citations": [
            {
                "text": "The sky is blue",
                "source": "Weather Guide",
                "start": 10,
                "end": 25
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
    assert isinstance(result[0]["citations"], Citations)
    assert len(result[0]["citations"]) == 1
    assert result[0]["citations"][0].text == "The sky is blue"


def test_citations_without_citations():
    """Test that processing works when no citations are present."""
    from dspy.adapters.chat_adapter import ChatAdapter

    adapter = ChatAdapter()

    # Mock outputs without citations
    outputs = [{
        "text": "[[ ## answer ## ]]\nThe answer is blue.\n\n[[ ## citations ## ]]\n[]"
    }]

    # Process with citation signature
    result = adapter._call_postprocess(
        CitationSignature,
        CitationSignature,
        outputs
    )

    # Should still work, with None for citations (since no citations in LM response)
    assert len(result) == 1
    # Note: When no citations are in LM response, citations field should be None
    # but the field gets set to empty list from parsing. Let's verify it's empty.
    assert result[0]["citations"] is None or (
        isinstance(result[0]["citations"], Citations) and len(result[0]["citations"]) == 0
    )


def test_citation_imports():
    """Test that Citations can be imported from main dspy module."""
    assert hasattr(dspy, "Citations")
    assert dspy.Citations is Citations

    # Individual Citation should only be accessible via Citations.Citation
    assert not hasattr(dspy, "Citation")


def test_citation_access_pattern():
    """Test that Citation class is accessible as Citations.Citation (like ToolCalls pattern)."""
    # Test that we can create Citation objects via Citations.Citation
    citation = Citations.Citation(text="Hello", source="World")
    assert citation.text == "Hello"
    assert citation.source == "World"

    # Test that Citations.Citation is the correct class
    assert hasattr(Citations, "Citation")
    assert isinstance(citation, Citations.Citation)
