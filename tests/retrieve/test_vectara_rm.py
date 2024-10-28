import pytest
import os
import responses
import json
from dspy.primitives.example import Example
from dspy.retrieve.vectara_rm import VectaraRM, remove_snippet

# Constants for testing
MOCK_CUSTOMER_ID = "test-customer"
MOCK_CORPUS_ID = "test-corpus"
MOCK_API_KEY = "test-api-key"
MOCK_API_URL = "https://api.vectara.io/v1/query"

@pytest.fixture
def mock_env_variables(monkeypatch):
    """Setup environment variables for testing"""
    monkeypatch.setenv("VECTARA_CUSTOMER_ID", MOCK_CUSTOMER_ID)
    monkeypatch.setenv("VECTARA_CORPUS_ID", MOCK_CORPUS_ID)
    monkeypatch.setenv("VECTARA_API_KEY", MOCK_API_KEY)

@pytest.fixture
def mock_response_data():
    """Mock response data from Vectara API"""
    return {
        "responseSet": [{
            "response": [
                {
                    "text": "<%START%>First test document<%END%>",
                    "score": 0.95
                },
                {
                    "text": "<%START%>Second test document<%END%>",
                    "score": 0.85
                }
            ]
        }]
    }

@pytest.fixture
def vectara_rm():
    """Create VectaraRM instance with test credentials"""
    return VectaraRM(
        vectara_customer_id=MOCK_CUSTOMER_ID,
        vectara_corpus_id=MOCK_CORPUS_ID,
        vectara_api_key=MOCK_API_KEY,
        k=2
    )

@responses.activate
def test_vectara_rm_basic_retrieval(vectara_rm, mock_response_data):
    """Test basic retrieval functionality"""
    responses.add(
        responses.POST,
        MOCK_API_URL,
        json=mock_response_data,
        status=200
    )
    
    query = "test query"
    results = [Example(long_text=x) if isinstance(x, str) else Example(**x) for x in vectara_rm.forward(query, k=2)]
    
    assert len(results) == 2
    assert isinstance(results[0], Example)
    assert results[0].long_text == "First test document"
    assert results[1].long_text == "Second test document"

@responses.activate
def test_vectara_rm_custom_k(vectara_rm, mock_response_data):
    """Test retrieval with custom k parameter"""
    responses.add(
        responses.POST,
        MOCK_API_URL,
        json=mock_response_data,
        status=200
    )
    
    query = "test query"
    results = [Example(long_text=x) if isinstance(x, str) else Example(**x) for x in vectara_rm.forward(query, k=1)]
    
    assert len(results) == 1
    assert results[0].long_text == "First test document"

def test_vectara_rm_env_variables(mock_env_variables):
    """Test initialization using environment variables"""
    rm = VectaraRM()
    assert rm._vectara_customer_id == MOCK_CUSTOMER_ID
    assert rm._vectara_corpus_id == MOCK_CORPUS_ID
    assert rm._vectara_api_key == MOCK_API_KEY

@responses.activate
def test_vectara_rm_error_handling(vectara_rm):
    """Test error handling for failed API requests"""
    responses.add(
        responses.POST,
        MOCK_API_URL,
        status=400,
        json={"error": "Bad Request"}
    )
    
    query = "test query"
    results = [Example(long_text=x) if isinstance(x, str) else Example(**x) for x in vectara_rm.forward(query, k=2)]
    assert len(results) == 0

def test_remove_snippet():
    """Test snippet removal function"""
    text = "<%START%>Test text<%END%>"
    assert remove_snippet(text) == "Test text"

@responses.activate
def test_vectara_rm_multiple_queries(vectara_rm, mock_response_data):
    """Test handling of multiple queries"""
    responses.add(
        responses.POST,
        MOCK_API_URL,
        json=mock_response_data,
        status=200
    )
    
    queries = ["query1", "query2"]
    results = [Example(long_text=x) if isinstance(x, str) else Example(**x) for x in vectara_rm.forward(queries, k=2)]
    
    assert len(results) == 2
    assert all(isinstance(result, Example) for result in results)

@responses.activate
def test_vectara_rm_empty_query_handling(vectara_rm, mock_response_data):
    """Test handling of empty queries"""
    responses.add(
        responses.POST,
        MOCK_API_URL,
        json=mock_response_data,
        status=200
    )
    
    queries = ["", "valid query", None]
    results = [Example(long_text=x) if isinstance(x, str) else Example(**x) for x in vectara_rm.forward(queries, k=2)]
    
    assert len(results) == 2  # Should only process the valid query

@pytest.mark.parametrize("corpus_id", [
    "single-corpus",
    "corpus1,corpus2,corpus3"
])
@responses.activate
def test_vectara_rm_multiple_corpus_ids(corpus_id, mock_response_data):
    """Test handling of single and multiple corpus IDs"""
    responses.add(
        responses.POST,
        MOCK_API_URL,
        json=mock_response_data,
        status=200
    )
    
    rm = VectaraRM(
        vectara_customer_id=MOCK_CUSTOMER_ID,
        vectara_corpus_id=corpus_id,
        vectara_api_key=MOCK_API_KEY
    )
    
    results = [Example(long_text=x) if isinstance(x, str) else Example(**x) for x in rm.forward("test query", k=2)]
    assert len(results) > 0
    
    # Verify the request payload
    request = responses.calls[0].request
    request_body = json.loads(request.body)
    corpus_ids = corpus_id.split(',')
    
    # Check if the correct number of corpus keys were included
    assert len(request_body['query'][0]['corpusKey']) == len(corpus_ids)
    
    # Verify each corpus ID was properly included
    for i, cid in enumerate(corpus_ids):
        assert request_body['query'][0]['corpusKey'][i]['corpusId'] == cid

def test_vectara_rm_default_parameters():
    """Test default parameters of VectaraRM"""
    rm = VectaraRM(
        vectara_customer_id=MOCK_CUSTOMER_ID,
        vectara_corpus_id=MOCK_CORPUS_ID,
        vectara_api_key=MOCK_API_KEY
    )
    
    assert rm.k == 5  # Default k value
    assert rm._n_sentences_before == 2
    assert rm._n_sentences_after == 2
    assert rm._vectara_timeout == 120