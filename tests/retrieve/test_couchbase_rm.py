import pytest
from unittest.mock import Mock, patch
from datetime import timedelta

from dspy.retrieve.couchbase_rm import CouchbaseRM, Embedder
from couchbase.options import ClusterOptions, SearchOptions
from couchbase.vector_search import VectorQuery, VectorSearch
from couchbase.bucket import Bucket
from couchbase.scope import Scope
from couchbase.collection import Collection
from couchbase.result import SearchResult, GetResult
import os

# Mock data
MOCK_EMBEDDING = [0.1, 0.2, 0.3]
MOCK_SEARCH_RESULTS = [
    {"id": "doc1", "fields": {"text": "Sample text 1"}, "score": 0.9},
    {"id": "doc2", "fields": {"text": "Sample text 2"}, "score": 0.8},
]

@pytest.fixture
def mock_cluster():
    with patch('dspy.retrieve.couchbase_rm.Cluster') as mock:
        cluster = Mock()
        mock.return_value = cluster
        yield cluster

@pytest.fixture()
def mock_embedder():
    embedder = Mock()  # Ensure it's callable
    embedder.return_value = [[0.1, 0.2, 0.3]]  # Correctly sets return value
    return embedder

@pytest.fixture
def mock_openai_embeddings():
    """Mock OpenAI embedding API call inside the Embedder class."""
    with patch("dspy.retrieve.couchbase_rm.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]  # Mocked embedding
        mock_client.embeddings.create.return_value = mock_embedding_response
        yield mock_openai

@pytest.fixture
def couchbase_rm(mock_cluster, mock_embedder):
    rm = CouchbaseRM(
        cluster_connection_string="couchbase://localhost",
        bucket="test_bucket",
        index_name="test_index",
        k=2,
        embedding_provider="openai",
        embedding_model="text-embedding-ada-002",
        embedding_function=mock_embedder
    )
    # Mock the embedder call to return iterable result
    rm.embedder = mock_embedder
    return rm

def test_initialization(couchbase_rm):
    assert couchbase_rm.bucket_name == "test_bucket"
    assert couchbase_rm.index_name == "test_index"
    assert couchbase_rm.k == 2
    assert couchbase_rm.is_global_index == False

def test_forward_global_index(couchbase_rm, mock_cluster):
    # Mock search response
    mock_response = Mock(spec=SearchResult)
    mock_response.rows.return_value = MOCK_SEARCH_RESULTS
    mock_cluster.search.return_value = mock_response

    # Fix: Set up bucket mock since it's used in the code
    mock_bucket = Mock(spec=Bucket)
    mock_cluster.bucket.return_value = mock_bucket
    
    # Set is_global_index to True for this test
    couchbase_rm.is_global_index = True

    # Test single query
    result = couchbase_rm.forward("test query")
    
    assert len(result) == 2
    assert result[0].long_text == "Sample text 1"
    assert result[1].long_text == "Sample text 2"

    # Verify search was called with correct parameters
    mock_cluster.search.assert_called_once()

def test_forward_scoped_index(mock_cluster, mock_embedder):
     # Set up mock bucket and scope
    mock_scope = Mock(spec=Scope)
    mock_bucket = Mock(spec=Bucket)
    mock_bucket.scope.return_value = mock_scope
    mock_cluster.bucket.return_value = mock_bucket

    # Mock scope search response
    mock_response = Mock(spec=SearchResult)
    mock_response.rows.return_value = [
        {"fields": {"text": "Sample text 1"}, "id": "doc1", "score": 0.9},
        {"fields": {"text": "Sample text 2"}, "id": "doc2", "score": 0.8}
    ]
    mock_scope.search.return_value = mock_response

    # Create RM with scoped index
    rm = CouchbaseRM(
        cluster_connection_string="couchbase://localhost",
        bucket="test_bucket",
        index_name="test_index",
        scope="test_scope",
        k=2,
        embedding_provider="openai",
        embedding_model="text-embedding-ada-002",
        embedding_function=mock_embedder
    )

    # Test search
    result = rm.forward("test query")
    
    assert len(result) == 2
    assert result[0].long_text == "Sample text 1"
    assert result[1].long_text == "Sample text 2"

    # Verify scope search was called
    mock_scope.search.assert_called_once()

def test_kv_get_text(mock_cluster, mock_openai_embeddings):

    # Mock KV get response
    mock_collection = Mock(spec=Collection)
    mock_get_result = Mock()
    mock_get_result.value = {"text": "Sample KV text"}
    mock_get_result.success = True
    
    mock_kv_response = Mock(spec=GetResult)
    mock_kv_response.all_ok = True
    mock_kv_response.results = {"doc1": mock_get_result}
    
    mock_collection.get_multi.return_value = mock_kv_response
    
    # Set up mock bucket, scope, and collection
    mock_scope = Mock(spec=Scope)
    mock_scope.collection.return_value = mock_collection
    mock_bucket = Mock(spec=Bucket)
    mock_bucket.scope.return_value = mock_scope
    mock_cluster.bucket.return_value = mock_bucket

    # Mock search response
    mock_response = Mock(spec=SearchResult)
    mock_response.rows.return_value = [ Mock(id="doc1", score=0.9)]
    mock_scope.search.return_value = mock_response

    os.environ["OPENAI_API_KEY"] = "test-key"
    # Create RM with KV get enabled
    rm = CouchbaseRM(
        cluster_connection_string="couchbase://localhost",
        bucket="test_bucket",
        index_name="test_index",
        scope="test_scope",
        collection="test_collection",
        k=2,
        use_kv_get_text=True,
        embedding_provider="openai",
        embedding_model="text-embedding-ada-002"
    )

    # Test search with KV get
    result = rm.forward("test query")
    
    assert len(result) == 1
    assert result[0].long_text == "Sample KV text"

    # Verify collection get_multi was called
    mock_collection.get_multi.assert_called_once_with(keys=["doc1"])

def test_embedder_initialization():
    with patch('dspy.retrieve.couchbase_rm.OpenAI') as mock_openai:
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            embedder = Embedder(provider="openai", model="test-model")
            assert embedder.model == "test-model"
            mock_openai.assert_called_once()

def test_embedder_missing_api_key():
    with patch.dict('os.environ', clear=True):
        with pytest.raises(ValueError, match="Environment variable OPENAI_API_KEY must be set"):
            Embedder(provider="openai", model="test-model") 