from unittest.mock import MagicMock, patch

import pytest

import dspy
from dspy.clients.cache import Cache
from dspy.dsp.colbertv2 import colbertv2_get_request_v2, colbertv2_post_request_v2


def test_get_request_raises_on_server_error():
    mock_response = MagicMock()
    mock_response.json.return_value = {"error": True, "message": "connection failed"}

    with patch("dspy.dsp.colbertv2.requests.get", return_value=mock_response):
        with pytest.raises(ValueError, match="connection failed"):
            colbertv2_get_request_v2("http://test", "query", k=3)


def test_post_request_raises_on_server_error():
    mock_response = MagicMock()
    mock_response.json.return_value = {"error": True, "message": "server error"}

    with patch("dspy.dsp.colbertv2.requests.post", return_value=mock_response):
        with pytest.raises(ValueError, match="server error"):
            colbertv2_post_request_v2("http://test2", "query", k=3)


def test_get_request_success():
    mock_response = MagicMock()
    mock_response.json.return_value = {"topk": [{"text": "doc1", "score": 0.9}]}

    with patch("dspy.dsp.colbertv2.requests.get", return_value=mock_response):
        result = colbertv2_get_request_v2("http://test3", "query", k=3)
        assert result[0]["long_text"] == "doc1"


def test_post_request_success():
    mock_response = MagicMock()
    mock_response.json.return_value = {"topk": [{"text": "doc1", "score": 0.9}]}

    with patch("dspy.dsp.colbertv2.requests.post", return_value=mock_response):
        result = colbertv2_post_request_v2("http://test4", "query", k=3)
        assert result[0]["text"] == "doc1"


def test_post_request_success_adds_long_text():
    mock_response = MagicMock()
    mock_response.json.return_value = {"topk": [{"text": "doc1", "score": 0.9}]}

    with patch("dspy.dsp.colbertv2.requests.post", return_value=mock_response):
        result = colbertv2_post_request_v2("http://test5", "query", k=3)

    assert result[0]["long_text"] == "doc1"


def test_colbert_post_wrapper_and_retrieve_use_long_text():
    mock_response = MagicMock()
    mock_response.json.return_value = {"topk": [{"text": "doc1", "score": 0.9}]}

    with patch("dspy.dsp.colbertv2.requests.post", return_value=mock_response):
        retriever = dspy.ColBERTv2(url="http://test6", post_requests=True)
        assert retriever("query", simplify=True) == ["doc1"]
        with dspy.context(rm=retriever):
            assert dspy.Retrieve(k=1)("query").passages == ["doc1"]


@pytest.mark.parametrize("cache_layer", ["colbertv2_post_request_v2", "colbertv2_post_request_v2_wrapped"])
def test_colbert_post_reads_legacy_disk_cache(tmp_path, monkeypatch, cache_layer):
    passages = [{"text": "doc1", "score": 0.9}, {"text": "doc2", "score": 0.8}]
    request = {
        "positional_arg_0": "http://cached-post",
        "positional_arg_1": "query",
        "positional_arg_2": 2,
        "_fn_identifier": f"dspy.dsp.colbertv2.{cache_layer}",
    }
    cache = Cache(enable_disk_cache=True, enable_memory_cache=False, disk_cache_dir=tmp_path)
    with cache.disk_cache:
        cache.put(request, passages)

    # Reopen the disk cache to simulate upgrading with responses from an older version.
    cache = Cache(enable_disk_cache=True, enable_memory_cache=False, disk_cache_dir=tmp_path)
    monkeypatch.setattr(dspy, "cache", cache)
    with cache.disk_cache, patch("dspy.dsp.colbertv2.requests.post") as post:
        retriever = dspy.ColBERTv2(url="http://cached-post", post_requests=True)
        result = retriever("query", k=2)
        assert result == [{**passage, "long_text": passage["text"]} for passage in passages]
        assert retriever("query", k=2, simplify=True) == ["doc1", "doc2"]
        with dspy.context(rm=retriever):
            assert dspy.Retrieve(k=2)("query").passages == ["doc1", "doc2"]
        post.assert_not_called()
        assert cache.get(request) == passages


@pytest.mark.parametrize("post_requests", [False, True])
@pytest.mark.parametrize("passages", [[], [{"long_text": "", "text": "fallback"}], [{"long_text": "doc1"}]])
def test_colbert_preserves_normalized_passages(post_requests, passages):
    request = "colbertv2_post_request" if post_requests else "colbertv2_get_request"
    with patch(f"dspy.dsp.colbertv2.{request}", return_value=passages):
        retriever = dspy.ColBERTv2(post_requests=post_requests)
        assert retriever("query") == passages
        assert retriever("query", simplify=True) == [passage["long_text"] for passage in passages]
