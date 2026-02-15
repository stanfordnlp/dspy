from unittest.mock import MagicMock, patch

import pytest

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
