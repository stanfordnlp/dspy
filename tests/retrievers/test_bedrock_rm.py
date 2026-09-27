"""Tests for the Amazon Bedrock Knowledge Base retriever module."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_boto3_client():
    with patch("boto3.client") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


def test_bedrock_rm_init():
    """Test BedrockRM initialization."""
    from dspy.retrievers.bedrock_rm import BedrockRM

    rm = BedrockRM(knowledge_base_id="TEST123456", region_name="us-west-2", k=10)
    assert rm.knowledge_base_id == "TEST123456"
    assert rm.region_name == "us-west-2"
    assert rm.k == 10


@patch.dict("os.environ", {"KNOWLEDGE_BASE_ID": "ENV_KB", "AWS_REGION": "eu-west-1"})
def test_bedrock_rm_init_from_env():
    """Test BedrockRM reads from environment variables."""
    from dspy.retrievers.bedrock_rm import BedrockRM

    rm = BedrockRM()
    assert rm.knowledge_base_id == "ENV_KB"
    assert rm.region_name == "eu-west-1"


def test_bedrock_rm_forward_managed(mock_boto3_client):
    """Test forward() with managed search configuration."""
    from dspy.retrievers.bedrock_rm import BedrockRM

    mock_boto3_client.retrieve.return_value = {
        "retrievalResults": [
            {
                "content": {"text": "Document about RAG."},
                "location": {"s3Location": {"uri": "s3://bucket/doc1.pdf"}},
                "score": 0.95,
            },
            {
                "content": {"text": "Another relevant document."},
                "location": {"s3Location": {"uri": "s3://bucket/doc2.pdf"}},
                "score": 0.88,
            },
        ]
    }

    rm = BedrockRM(knowledge_base_id="TEST123456", k=3)
    result = rm.forward("What is RAG?")

    # Verify API called with managed config
    mock_boto3_client.retrieve.assert_called_once()
    call_kwargs = mock_boto3_client.retrieve.call_args.kwargs
    assert call_kwargs["knowledgeBaseId"] == "TEST123456"
    assert "managedSearchConfiguration" in call_kwargs["retrievalConfiguration"]
    assert call_kwargs["retrievalConfiguration"]["managedSearchConfiguration"]["numberOfResults"] == 3

    # Verify result is a Prediction with passages
    assert isinstance(result, list)
    assert len(result) == 2
    assert "Document about RAG" in result[0].long_text


def test_bedrock_rm_forward_multiple_queries(mock_boto3_client):
    """Test forward() with multiple queries."""
    from dspy.retrievers.bedrock_rm import BedrockRM

    mock_boto3_client.retrieve.return_value = {
        "retrievalResults": [
            {
                "content": {"text": "Result."},
                "location": {"s3Location": {"uri": "s3://bucket/r.pdf"}},
                "score": 0.9,
            },
        ]
    }

    rm = BedrockRM(knowledge_base_id="TEST123456")
    result = rm.forward(["query1", "query2"])

    assert mock_boto3_client.retrieve.call_count == 2
    assert len(result) == 2


def test_bedrock_rm_forward_empty_results(mock_boto3_client):
    """Test forward() with no results."""
    from dspy.retrievers.bedrock_rm import BedrockRM

    mock_boto3_client.retrieve.return_value = {"retrievalResults": []}

    rm = BedrockRM(knowledge_base_id="TEST123456")
    result = rm.forward("obscure query")

    assert result == []


def test_bedrock_rm_forward_error(mock_boto3_client):
    """Test forward() raises on API error."""
    from dspy.retrievers.bedrock_rm import BedrockRM

    mock_boto3_client.retrieve.side_effect = Exception("Service unavailable")

    rm = BedrockRM(knowledge_base_id="TEST123456")
    with pytest.raises(RuntimeError, match="Error retrieving from Bedrock KB"):
        rm.forward("test query")


def test_bedrock_rm_override_k(mock_boto3_client):
    """Test that k can be overridden in forward()."""
    from dspy.retrievers.bedrock_rm import BedrockRM

    mock_boto3_client.retrieve.return_value = {"retrievalResults": []}

    rm = BedrockRM(knowledge_base_id="TEST123456", k=3)
    rm.forward("test", k=7)

    call_kwargs = mock_boto3_client.retrieve.call_args.kwargs
    assert call_kwargs["retrievalConfiguration"]["managedSearchConfiguration"]["numberOfResults"] == 7
