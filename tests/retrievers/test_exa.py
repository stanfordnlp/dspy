import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_exa_import(monkeypatch):
    """Ensure exa_py is importable even if not installed."""
    import sys

    if "exa_py" not in sys.modules:
        mock_exa_module = MagicMock()
        monkeypatch.setitem(sys.modules, "exa_py", mock_exa_module)


@pytest.fixture
def mock_exa_client():
    with patch("dspy.retrievers.exa._make_client") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


def _make_search_result(title="Test", url="https://example.com", text="Some text"):
    result = MagicMock()
    result.title = title
    result.url = url
    result.text = text
    result.summary = None
    result.highlights = None
    return result


class TestExaRM:
    def test_forward_single_query(self, mock_exa_client):
        from dspy.retrievers.exa import ExaRM

        mock_result = _make_search_result(text="passage one")
        mock_exa_client.search.return_value = MagicMock(results=[mock_result])

        rm = ExaRM(exa_api_key="test-key", k=3)
        prediction = rm.forward("test query")

        assert prediction.passages == ["passage one"]
        assert prediction.urls == ["https://example.com"]
        mock_exa_client.search.assert_called_once()

    def test_forward_multiple_queries(self, mock_exa_client):
        from dspy.retrievers.exa import ExaRM

        r1 = _make_search_result(text="result 1", url="https://a.com")
        r2 = _make_search_result(text="result 2", url="https://b.com")
        mock_exa_client.search.side_effect = [
            MagicMock(results=[r1]),
            MagicMock(results=[r2]),
        ]

        rm = ExaRM(exa_api_key="test-key", k=1)
        prediction = rm.forward(["query 1", "query 2"])

        assert len(prediction.passages) == 2
        assert prediction.passages[0] == "result 1"
        assert prediction.passages[1] == "result 2"

    def test_forward_empty_query_filtered(self, mock_exa_client):
        from dspy.retrievers.exa import ExaRM

        rm = ExaRM(exa_api_key="test-key")
        mock_exa_client.search.return_value = MagicMock(results=[])
        rm.forward(["", "valid query"])

        # Only the valid query should be searched
        assert mock_exa_client.search.call_count == 1

    def test_forward_passes_filters(self, mock_exa_client):
        from dspy.retrievers.exa import ExaRM

        mock_exa_client.search.return_value = MagicMock(results=[])

        rm = ExaRM(
            exa_api_key="test-key",
            include_domains=["example.com"],
            exclude_domains=["bad.com"],
            start_published_date="2024-01-01",
            end_published_date="2024-12-31",
            search_type="neural",
        )
        rm.forward("test")

        call_kwargs = mock_exa_client.search.call_args
        assert call_kwargs[1]["include_domains"] == ["example.com"]
        assert call_kwargs[1]["exclude_domains"] == ["bad.com"]
        assert call_kwargs[1]["start_published_date"] == "2024-01-01"
        assert call_kwargs[1]["end_published_date"] == "2024-12-31"
        assert call_kwargs[1]["type"] == "neural"

    def test_k_override(self, mock_exa_client):
        from dspy.retrievers.exa import ExaRM

        mock_exa_client.search.return_value = MagicMock(results=[])

        rm = ExaRM(exa_api_key="test-key", k=3)
        rm.forward("test", k=10)

        call_kwargs = mock_exa_client.search.call_args
        assert call_kwargs[1]["num_results"] == 10


class TestExaSearchTool:
    def test_call(self, mock_exa_client):
        from dspy.retrievers.exa import ExaSearchTool

        r = _make_search_result(title="Result", url="https://example.com", text="content here")
        mock_exa_client.search.return_value = MagicMock(results=[r])

        tool = ExaSearchTool(api_key="test-key")
        output = tool("test query")

        assert "Result" in output
        assert "https://example.com" in output
        assert "content here" in output

    def test_call_with_filters(self, mock_exa_client):
        from dspy.retrievers.exa import ExaSearchTool

        mock_exa_client.search.return_value = MagicMock(results=[])

        tool = ExaSearchTool(
            api_key="test-key",
            include_domains=["example.com"],
            search_type="keyword",
        )
        tool("test")

        call_kwargs = mock_exa_client.search.call_args
        assert call_kwargs[1]["include_domains"] == ["example.com"]
        assert call_kwargs[1]["type"] == "keyword"

    def test_docstring_present(self, mock_exa_client):
        from dspy.retrievers.exa import ExaSearchTool

        tool = ExaSearchTool(api_key="test-key")
        assert tool.__call__.__doc__ is not None


class TestExaContentsTool:
    def test_call(self, mock_exa_client):
        from dspy.retrievers.exa import ExaContentsTool

        r = _make_search_result(title="Page", url="https://example.com/page", text="extracted content")
        mock_exa_client.get_contents.return_value = MagicMock(results=[r])

        tool = ExaContentsTool(api_key="test-key")
        output = tool("https://example.com/page")

        assert "extracted content" in output
        mock_exa_client.get_contents.assert_called_once()

    def test_no_content_found(self, mock_exa_client):
        from dspy.retrievers.exa import ExaContentsTool

        mock_exa_client.get_contents.return_value = MagicMock(results=[])

        tool = ExaContentsTool(api_key="test-key")
        output = tool("https://example.com/empty")

        assert "No content found" in output


class TestExaFindSimilarTool:
    def test_call(self, mock_exa_client):
        from dspy.retrievers.exa import ExaFindSimilarTool

        r = _make_search_result(title="Similar", url="https://similar.com", text="similar content")
        mock_exa_client.find_similar.return_value = MagicMock(results=[r])

        tool = ExaFindSimilarTool(api_key="test-key")
        output = tool("https://example.com")

        assert "Similar" in output
        assert "similar content" in output
        mock_exa_client.find_similar.assert_called_once()


class TestMakeClient:
    def test_missing_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Exa API key"):
                from dspy.retrievers.exa import _make_client

                _make_client()

    def test_env_var_fallback(self):
        with patch.dict(os.environ, {"EXA_API_KEY": "env-key"}):
            with patch("dspy.retrievers.exa.Exa") as mock_exa_cls:
                mock_instance = MagicMock()
                mock_instance.headers = {}
                mock_exa_cls.return_value = mock_instance

                from dspy.retrievers.exa import _make_client

                client = _make_client()
                mock_exa_cls.assert_called_once_with(api_key="env-key", base_url="https://api.exa.ai")
                assert client.headers["x-exa-integration"] == "exa-dspy"


class TestFormatResults:
    def test_basic_formatting(self):
        from dspy.retrievers.exa import _format_results

        r = _make_search_result(title="Test Title", url="https://test.com", text="Test content")
        output = _format_results([r])

        assert "Title: Test Title" in output
        assert "URL: https://test.com" in output
        assert "Content: Test content" in output

    def test_truncation(self):
        from dspy.retrievers.exa import _format_results

        r = _make_search_result(text="x" * 2000)
        output = _format_results([r], max_chars=100)

        assert "..." in output

    def test_with_summary_and_highlights(self):
        from dspy.retrievers.exa import _format_results

        r = _make_search_result()
        r.summary = "A brief summary"
        r.highlights = ["highlight one", "highlight two"]
        output = _format_results([r])

        assert "Summary: A brief summary" in output
        assert "highlight one" in output
        assert "highlight two" in output
