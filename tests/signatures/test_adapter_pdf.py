import pytest
from typing import Optional
from unittest import mock

import dspy
import litellm
from dspy.signatures.signature import Signature


# Mock LM class similar to the one in test_adapter_image.py
class MockLiteLLM:
    def __init__(self, response=None):
        self.history = []
        self.response = response or {}

    def complete(self, prompt=None, messages=None, **kwargs):
        self.history.append({"prompt": prompt, "messages": messages, "kwargs": kwargs})
        return self.response


def setup_predictor(signature_class, mock_response):
    """Set up a predictor with a MockLiteLLM for testing PDF type."""
    lm = MockLiteLLM(response=mock_response)
    predictor = dspy.Predict(signature_class, lm=lm)
    return predictor, lm


def count_messages_with_file_pattern(messages):
    """Count how many messages contain file content in their format."""
    count = 0
    for message in messages or []:
        if message.get("role") == "user":
            content = message.get("content", [])
            if isinstance(content, list):
                for content_part in content:
                    if content_part.get("type") == "file":
                        count += 1
    return count


def test_pdf_url_support():
    """Test support for PDF files from URLs"""
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

    # Create a dspy.PDF object from the PDF URL with download=True
    pdf_doc = dspy.PDF.from_url(pdf_url, download=True)

    # The data URI should contain application/pdf in the MIME type
    assert "data:application/pdf" in pdf_doc.url
    assert ";base64," in pdf_doc.url

    # Test using it in a predictor
    class PDFSignature(dspy.Signature):
        document: dspy.PDF = dspy.InputField(desc="A PDF document")
        summary: str = dspy.OutputField(desc="A summary of the PDF")

    predictor, lm = setup_predictor(PDFSignature, {"summary": "This is a dummy PDF"})
    result = predictor(document=pdf_doc)

    assert result.summary == "This is a dummy PDF"
    assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 1

    # Check that file was properly formatted
    messages = lm.history[-1]["messages"]
    user_message = next(m for m in messages if m["role"] == "user")
    content = user_message["content"]
    
    # Find the file content part
    file_content = next(c for c in content if c["type"] == "file")
    
    # Verify it has the right format for document understanding
    assert "file" in file_content
    assert "file_data" in file_content["file"]
    assert pdf_doc.url == file_content["file"]["file_data"]


def test_pdf_url_without_download():
    """Test support for PDF files from URLs without download"""
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

    # Create a dspy.PDF object from the PDF URL without download
    pdf_doc = dspy.PDF.from_url(pdf_url, download=False)

    # Should maintain the original URL
    assert pdf_doc.url == pdf_url

    # Test using it in a predictor
    class PDFSignature(dspy.Signature):
        document: dspy.PDF = dspy.InputField(desc="A PDF document")
        summary: str = dspy.OutputField(desc="A summary of the PDF")

    predictor, lm = setup_predictor(PDFSignature, {"summary": "This is a PDF from URL"})
    result = predictor(document=pdf_doc)

    assert result.summary == "This is a PDF from URL"
    assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 1

    # Check that file was properly formatted
    messages = lm.history[-1]["messages"]
    user_message = next(m for m in messages if m["role"] == "user")
    content = user_message["content"]
    
    # Find the file content part
    file_content = next(c for c in content if c["type"] == "file")
    
    # Verify it has the right format for document understanding with URL
    assert "file" in file_content
    assert "file_id" in file_content["file"]
    assert "format" in file_content["file"]
    assert pdf_doc.url == file_content["file"]["file_id"]
    assert file_content["file"]["format"] == "application/pdf"


def test_optional_pdf_field():
    """Test that optional PDF fields are not required"""

    class OptionalPDFSignature(dspy.Signature):
        document: Optional[dspy.PDF] = dspy.InputField(desc="An optional PDF document")
        summary: str = dspy.OutputField(desc="A summary")

    predictor, lm = setup_predictor(OptionalPDFSignature, {"summary": "No PDF provided"})
    result = predictor(document=None)
    
    assert result.summary == "No PDF provided"
    assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 0


def test_model_compatibility_check():
    """Test that model compatibility is checked correctly"""
    
    # Test with a supported model (no mocking needed with our fallback implementation)
    assert dspy.PDF.supports_pdf_input("anthropic/claude-3-opus") is True
    assert dspy.PDF.supports_pdf_input("vertex/gemini-pro") is True
    assert dspy.PDF.supports_pdf_input("bedrock/anthropic.claude-3-sonnet") is True
    
    # Test with an unsupported model
    assert dspy.PDF.supports_pdf_input("openai/gpt-3.5-turbo") is False
    assert dspy.PDF.supports_pdf_input("openai/gpt-4") is False
    
    # Test with empty or None model
    assert dspy.PDF.supports_pdf_input("") is False
    assert dspy.PDF.supports_pdf_input(None) is False