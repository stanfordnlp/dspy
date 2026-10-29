import sys
import types
from unittest.mock import MagicMock

fake_chromadb = types.ModuleType("chromadb")
fake_chromadb.Client = MagicMock
sys.modules.setdefault("chromadb", fake_chromadb)

from dspy.retrievers.chromadb_rm import ChromadbRM  # noqa: E402


def test_forward_wraps_documents_as_long_text():
    fake_collection = MagicMock()
    fake_collection.query.return_value = {"documents": [["doc1", "doc2"]]}
    fake_client = MagicMock()
    fake_client.get_collection.return_value = fake_collection

    rm = ChromadbRM("my_collection", fake_client, k=2)
    result = rm.forward("some query")

    assert [dd["long_text"] for dd in result] == ["doc1", "doc2"]


def test_forward_passes_k_to_query():
    fake_collection = MagicMock()
    fake_collection.query.return_value = {"documents": [["doc1", "doc2"]]}
    fake_client = MagicMock()
    fake_client.get_collection.return_value = fake_collection

    rm = ChromadbRM("my_collection", fake_client, k=2)
    rm.forward("some query")

    fake_collection.query.assert_called_with(query_texts=["some query"], n_results=2)


def test_forward_filters_empty_queries():
    fake_collection = MagicMock()
    fake_collection.query.return_value = {"documents": [["doc1", "doc2"]]}
    fake_client = MagicMock()
    fake_client.get_collection.return_value = fake_collection

    rm = ChromadbRM("my_collection", fake_client, k=2)
    rm.forward(["", "real question"])

    assert fake_collection.query.call_count == 1
    fake_collection.query.assert_called_with(query_texts=["real question"], n_results=2)
