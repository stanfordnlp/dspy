"""Unit tests for GoodMemRM, GoodMemClient, and make_goodmem_tools.

All HTTP calls are mocked so no live GoodMem server is required.

Run with:
    python -m pytest tests/retrievers/test_goodmem_rm.py -v
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from dspy.dsp.utils import dotdict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    *,
    ok=True,
    status_code=200,
    json_data=None,
    text=None,
    raise_for_status_error=None,
):
    """Create a mock ``requests.Response``."""
    resp = MagicMock()
    resp.ok = ok
    resp.status_code = status_code
    if json_data is not None:
        resp.json.return_value = json_data
    if text is not None:
        resp.text = text
    else:
        resp.text = json.dumps(json_data) if json_data else ""
    if raise_for_status_error:
        # For _raise_for_status, we just set ok=False
        resp.ok = False
        resp.json.return_value = {"error": "test error"}
    return resp


# Sample NDJSON responses
NDJSON_WITH_RESULTS = "\n".join(
    [
        json.dumps({"resultSetBoundary": {"resultSetId": "rs-1", "boundary": "START"}}),
        json.dumps(
            {
                "retrievedItem": {
                    "chunk": {
                        "chunk": {"chunkId": "c-1", "chunkText": "Hello world", "memoryId": "mem-1"},
                        "relevanceScore": 0.95,
                        "memoryIndex": 0,
                    }
                }
            }
        ),
        json.dumps({"memoryDefinition": {"memoryId": "mem-1", "spaceId": "sp-1"}}),
        json.dumps({"resultSetBoundary": {"resultSetId": "rs-1", "boundary": "END"}}),
    ]
)

NDJSON_EMPTY = json.dumps({"resultSetBoundary": {"resultSetId": "rs-empty", "boundary": "START"}})

NDJSON_SSE_FORMAT = "\n".join(
    [
        "event: message",
        'data: {"resultSetBoundary": {"resultSetId": "rs-sse"}}',
        "",
        "event: message",
        'data: {"retrievedItem": {"chunk": {"chunk": {"chunkId": "c-sse", "chunkText": "SSE text", "memoryId": "mem-sse"}, "relevanceScore": 0.9, "memoryIndex": 0}}}',
    ]
)


# ---------------------------------------------------------------------------
# GoodMemClient tests
# ---------------------------------------------------------------------------


class TestGoodMemClient:
    """Tests for :class:`GoodMemClient`."""

    def _make_client(self, **kwargs):
        from dspy.utils.goodmem.client import GoodMemClient

        defaults = {
            "api_key": "test-key",
            "base_url": "http://localhost:8080",
            "verify_ssl": False,
        }
        defaults.update(kwargs)
        return GoodMemClient(**defaults)

    # ---- Init ----

    def test_trailing_slash_removed(self):
        c = self._make_client(base_url="http://localhost:8080/")
        assert c.base_url == "http://localhost:8080"

    def test_headers_contain_api_key(self):
        c = self._make_client()
        h = c._headers()
        assert h["X-API-Key"] == "test-key"
        assert h["Content-Type"] == "application/json"
        assert h["Accept"] == "application/json"

    def test_headers_custom_accept(self):
        c = self._make_client()
        h = c._headers(accept="application/x-ndjson")
        assert h["Accept"] == "application/x-ndjson"

    # ---- NDJSON parsing ----

    def test_parse_ndjson_plain(self):
        from dspy.utils.goodmem.client import GoodMemClient

        text = (
            '{"resultSetBoundary":{"resultSetId":"rs1"}}\n'
            '{"retrievedItem":{"chunk":{"chunk":{"chunkId":"c1","chunkText":"hello","memoryId":"m1"},"relevanceScore":0.9}}}\n'
        )
        items = GoodMemClient._parse_ndjson(text)
        assert len(items) == 2
        assert items[0]["resultSetBoundary"]["resultSetId"] == "rs1"
        assert items[1]["retrievedItem"]["chunk"]["chunk"]["chunkText"] == "hello"

    def test_parse_ndjson_sse_format(self):
        from dspy.utils.goodmem.client import GoodMemClient

        text = (
            "event: message\n"
            'data: {"resultSetBoundary":{"resultSetId":"rs2"}}\n'
            "\n"
            'data: {"retrievedItem":{"chunk":{"chunk":{"chunkId":"c2","chunkText":"world"},"relevanceScore":0.8}}}\n'
        )
        items = GoodMemClient._parse_ndjson(text)
        assert len(items) == 2

    def test_parse_ndjson_ignores_bad_json(self):
        from dspy.utils.goodmem.client import GoodMemClient

        text = '{"valid":true}\nnot-json\n{"also":true}\n'
        items = GoodMemClient._parse_ndjson(text)
        assert len(items) == 2

    # ---- MIME type ----

    def test_get_mime_type_known(self):
        from dspy.utils.goodmem.client import GoodMemClient

        assert GoodMemClient._get_mime_type("pdf") == "application/pdf"
        assert GoodMemClient._get_mime_type("PNG") == "image/png"
        assert GoodMemClient._get_mime_type(".jpg") == "image/jpeg"
        assert GoodMemClient._get_mime_type("txt") == "text/plain"
        assert GoodMemClient._get_mime_type("md") == "text/markdown"
        assert GoodMemClient._get_mime_type("docx") == (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    def test_get_mime_type_unknown(self):
        from dspy.utils.goodmem.client import GoodMemClient

        assert GoodMemClient._get_mime_type("xyz") is None
        assert GoodMemClient._get_mime_type("") is None

    # ---- list_embedders ----

    @patch("dspy.utils.goodmem.client.requests.get")
    def test_list_embedders_dict_response(self, mock_get):
        mock_get.return_value = _make_response(json_data={"embedders": [{"embedderId": "e1", "displayName": "Test"}]})
        result = self._make_client().list_embedders()
        assert result == [{"embedderId": "e1", "displayName": "Test"}]

    @patch("dspy.utils.goodmem.client.requests.get")
    def test_list_embedders_list_response(self, mock_get):
        mock_get.return_value = _make_response(json_data=[{"embedderId": "e2"}])
        result = self._make_client().list_embedders()
        assert result == [{"embedderId": "e2"}]

    @patch("dspy.utils.goodmem.client.requests.get")
    def test_list_embedders_http_error(self, mock_get):
        mock_get.return_value = _make_response(raise_for_status_error=True)
        with pytest.raises(RuntimeError, match="GoodMem API error"):
            self._make_client().list_embedders()

    # ---- list_spaces ----

    @patch("dspy.utils.goodmem.client.requests.get")
    def test_list_spaces_array_response(self, mock_get):
        mock_get.return_value = _make_response(json_data=[{"spaceId": "s1", "name": "test"}])
        result = self._make_client().list_spaces()
        assert result == [{"spaceId": "s1", "name": "test"}]

    @patch("dspy.utils.goodmem.client.requests.get")
    def test_list_spaces_object_response(self, mock_get):
        mock_get.return_value = _make_response(json_data={"spaces": [{"spaceId": "s1"}]})
        result = self._make_client().list_spaces()
        assert result == [{"spaceId": "s1"}]

    @patch("dspy.utils.goodmem.client.requests.get")
    def test_list_spaces_empty(self, mock_get):
        mock_get.return_value = _make_response(json_data={"spaces": []})
        result = self._make_client().list_spaces()
        assert result == []

    # ---- get_space ----

    @patch("dspy.utils.goodmem.client.requests.get")
    def test_get_space(self, mock_get):
        mock_get.return_value = _make_response(json_data={"spaceId": "s1", "name": "my-space"})
        result = self._make_client().get_space("s1")
        assert result["spaceId"] == "s1"
        mock_get.assert_called_once()
        assert "/v1/spaces/s1" in mock_get.call_args[0][0]

    # ---- create_space ----

    @patch("dspy.utils.goodmem.client.requests.post")
    @patch("dspy.utils.goodmem.client.requests.get")
    def test_create_space_new(self, mock_get, mock_post):
        mock_get.return_value = _make_response(json_data=[])
        mock_post.return_value = _make_response(json_data={"spaceId": "new-id", "name": "new-space"})
        result = self._make_client().create_space("new-space", "emb-1")
        assert result["reused"] is False
        assert result["spaceId"] == "new-id"
        assert result["success"] is True

    @patch("dspy.utils.goodmem.client.requests.post")
    @patch("dspy.utils.goodmem.client.requests.get")
    def test_create_space_idempotent(self, mock_get, mock_post):
        mock_get.return_value = _make_response(json_data=[{"spaceId": "existing-id", "name": "my-space"}])
        result = self._make_client().create_space("my-space", "emb-1")
        assert result["reused"] is True
        assert result["spaceId"] == "existing-id"
        mock_post.assert_not_called()

    @patch("dspy.utils.goodmem.client.requests.post")
    @patch("dspy.utils.goodmem.client.requests.get")
    def test_create_space_list_fails_still_creates(self, mock_get, mock_post):
        """If list_spaces fails, create_space should still try to create."""
        mock_get.return_value = _make_response(raise_for_status_error=True)
        mock_post.return_value = _make_response(json_data={"spaceId": "new-id", "name": "test"})
        result = self._make_client().create_space("test", "emb-1")
        assert result["success"] is True
        assert result["reused"] is False

    @patch("dspy.utils.goodmem.client.requests.post")
    @patch("dspy.utils.goodmem.client.requests.get")
    def test_create_space_post_error_propagates(self, mock_get, mock_post):
        mock_get.return_value = _make_response(json_data=[])
        mock_post.return_value = _make_response(raise_for_status_error=True)
        with pytest.raises(RuntimeError, match="GoodMem API error"):
            self._make_client().create_space("fail", "emb-1")

    # ---- update_space ----

    @patch("dspy.utils.goodmem.client.requests.put")
    def test_update_space(self, mock_put):
        mock_put.return_value = _make_response(json_data={"spaceId": "s1", "name": "renamed"})
        result = self._make_client().update_space("s1", name="renamed")
        assert result["name"] == "renamed"
        body = mock_put.call_args[1]["json"]
        assert body["name"] == "renamed"

    def test_update_space_both_labels_raises(self):
        with pytest.raises(ValueError, match="Cannot use both"):
            self._make_client().update_space("s1", replace_labels={"a": "b"}, merge_labels={"c": "d"})

    # ---- delete_space ----

    @patch("dspy.utils.goodmem.client.requests.delete")
    def test_delete_space(self, mock_delete):
        mock_delete.return_value = _make_response(json_data={})
        result = self._make_client().delete_space("s1")
        assert result["success"] is True
        assert result["spaceId"] == "s1"

    # ---- create_memory ----

    @patch("dspy.utils.goodmem.client.requests.post")
    def test_create_memory_text(self, mock_post):
        mock_post.return_value = _make_response(
            json_data={"memoryId": "mem-1", "spaceId": "sp-1", "processingStatus": "PENDING"}
        )
        result = self._make_client().create_memory("sp-1", text_content="Hello world")
        assert result["success"] is True
        assert result["memoryId"] == "mem-1"
        assert result["contentType"] == "text/plain"
        body = mock_post.call_args[1]["json"]
        assert body["originalContent"] == "Hello world"
        assert body["contentType"] == "text/plain"
        assert "originalContentB64" not in body

    @patch("dspy.utils.goodmem.client.requests.post")
    def test_create_memory_text_file(self, mock_post):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("file content here")
            tmp_path = f.name
        try:
            mock_post.return_value = _make_response(
                json_data={"memoryId": "mem-f", "spaceId": "sp-1", "processingStatus": "PENDING"}
            )
            result = self._make_client().create_memory("sp-1", file_path=tmp_path)
            assert result["success"] is True
            assert result["contentType"] == "text/plain"
            body = mock_post.call_args[1]["json"]
            assert body["originalContent"] == "file content here"
            assert "originalContentB64" not in body
        finally:
            os.unlink(tmp_path)

    @patch("dspy.utils.goodmem.client.requests.post")
    def test_create_memory_binary_file(self, mock_post):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-fake-content")
            tmp_path = f.name
        try:
            mock_post.return_value = _make_response(
                json_data={"memoryId": "mem-pdf", "spaceId": "sp-1", "processingStatus": "PENDING"}
            )
            result = self._make_client().create_memory("sp-1", file_path=tmp_path)
            assert result["success"] is True
            assert result["contentType"] == "application/pdf"
            body = mock_post.call_args[1]["json"]
            assert "originalContentB64" in body
            assert "originalContent" not in body
            # Verify base64 round-trip
            decoded = base64.b64decode(body["originalContentB64"])
            assert decoded == b"%PDF-fake-content"
        finally:
            os.unlink(tmp_path)

    @patch("dspy.utils.goodmem.client.requests.post")
    def test_create_memory_file_takes_priority(self, mock_post):
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("# Markdown content")
            tmp_path = f.name
        try:
            mock_post.return_value = _make_response(
                json_data={"memoryId": "mem-md", "spaceId": "sp-1", "processingStatus": "PENDING"}
            )
            result = self._make_client().create_memory("sp-1", text_content="ignored text", file_path=tmp_path)
            assert result["contentType"] == "text/markdown"
            body = mock_post.call_args[1]["json"]
            assert body["originalContent"] == "# Markdown content"
        finally:
            os.unlink(tmp_path)

    @patch("dspy.utils.goodmem.client.requests.post")
    def test_create_memory_with_metadata(self, mock_post):
        mock_post.return_value = _make_response(
            json_data={"memoryId": "mem-m", "spaceId": "sp-1", "processingStatus": "PENDING"}
        )
        self._make_client().create_memory(
            "sp-1",
            text_content="test",
            source="src",
            author="auth",
            tags="a,b,c",
            metadata={"extra": "value"},
        )
        body = mock_post.call_args[1]["json"]
        assert body["metadata"]["source"] == "src"
        assert body["metadata"]["author"] == "auth"
        assert body["metadata"]["tags"] == ["a", "b", "c"]
        assert body["metadata"]["extra"] == "value"

    def test_create_memory_no_content_raises(self):
        with pytest.raises(ValueError, match="No content provided"):
            self._make_client().create_memory("space-id")

    @patch("dspy.utils.goodmem.client.requests.post")
    def test_create_memory_http_error(self, mock_post):
        mock_post.return_value = _make_response(raise_for_status_error=True)
        with pytest.raises(RuntimeError, match="GoodMem API error"):
            self._make_client().create_memory("sp-1", text_content="test")

    # ---- retrieve_memories ----

    @patch("dspy.utils.goodmem.client.requests.post")
    def test_retrieve_with_results(self, mock_post):
        mock_post.return_value = _make_response(text=NDJSON_WITH_RESULTS)
        result = self._make_client().retrieve_memories("What is this?", ["sp-1"], wait_for_indexing=False)
        assert result["success"] is True
        assert result["totalResults"] == 1
        assert result["results"][0]["chunkId"] == "c-1"
        assert result["results"][0]["chunkText"] == "Hello world"
        assert result["results"][0]["relevanceScore"] == 0.95
        assert result["resultSetId"] == "rs-1"
        assert len(result["memories"]) == 1

    @patch("dspy.utils.goodmem.client.requests.post")
    def test_retrieve_sse_format(self, mock_post):
        mock_post.return_value = _make_response(text=NDJSON_SSE_FORMAT)
        result = self._make_client().retrieve_memories("test", ["sp-1"], wait_for_indexing=False)
        assert result["success"] is True
        assert result["totalResults"] == 1
        assert result["results"][0]["chunkId"] == "c-sse"

    @patch("dspy.utils.goodmem.client.requests.post")
    def test_retrieve_ndjson_accept_header(self, mock_post):
        mock_post.return_value = _make_response(text=NDJSON_WITH_RESULTS)
        self._make_client().retrieve_memories("test", ["sp-1"], wait_for_indexing=False)
        headers = mock_post.call_args[1]["headers"]
        assert headers["Accept"] == "application/x-ndjson"

    def test_retrieve_empty_space_ids_raises(self):
        with pytest.raises(ValueError, match="At least one space ID"):
            self._make_client().retrieve_memories("query", "")

    def test_retrieve_blank_space_ids_filtered(self):
        with pytest.raises(ValueError, match="At least one space ID"):
            self._make_client().retrieve_memories("query", ["", " ", ""])

    def test_retrieve_comma_string_space_ids(self):
        """Comma-separated string is split into list."""
        c = self._make_client()
        with patch("dspy.utils.goodmem.client.requests.post") as mock_post:
            mock_post.return_value = _make_response(text=NDJSON_WITH_RESULTS)
            c.retrieve_memories("q", "sp-1, sp-2", wait_for_indexing=False)
        body = mock_post.call_args[1]["json"]
        assert body["spaceKeys"] == [{"spaceId": "sp-1"}, {"spaceId": "sp-2"}]

    @patch("dspy.utils.goodmem.client.requests.post")
    def test_retrieve_wait_timeout(self, mock_post):
        mock_post.return_value = _make_response(text=NDJSON_EMPTY)
        c = self._make_client(poll_timeout=0, poll_interval=0)
        result = c.retrieve_memories("test", ["sp-1"], wait_for_indexing=True)
        assert result["success"] is True
        assert result["totalResults"] == 0
        assert "No results found" in result.get("message", "")

    @patch("dspy.utils.goodmem.client.requests.post")
    def test_retrieve_http_error(self, mock_post):
        mock_post.return_value = _make_response(raise_for_status_error=True)
        with pytest.raises(RuntimeError, match="GoodMem API error"):
            self._make_client().retrieve_memories("test", ["sp-1"], wait_for_indexing=False)

    # ---- get_memory ----

    @patch("dspy.utils.goodmem.client.requests.get")
    def test_get_memory_with_content(self, mock_get):
        meta_resp = _make_response(json_data={"memoryId": "m1", "processingStatus": "COMPLETED"})
        content_resp = _make_response(json_data={"text": "Hello world"})
        mock_get.side_effect = [meta_resp, content_resp]
        result = self._make_client().get_memory("m1", include_content=True)
        assert result["success"] is True
        assert result["memory"]["memoryId"] == "m1"
        assert result["content"] == {"text": "Hello world"}
        assert mock_get.call_count == 2
        urls = [c[0][0] for c in mock_get.call_args_list]
        assert urls[0].endswith("/v1/memories/m1")
        assert urls[1].endswith("/v1/memories/m1/content")

    @patch("dspy.utils.goodmem.client.requests.get")
    def test_get_memory_without_content(self, mock_get):
        mock_get.return_value = _make_response(json_data={"memoryId": "m1", "status": "COMPLETED"})
        result = self._make_client().get_memory("m1", include_content=False)
        assert result["success"] is True
        assert result["memory"]["memoryId"] == "m1"
        assert "content" not in result
        assert mock_get.call_count == 1

    @patch("dspy.utils.goodmem.client.requests.get")
    def test_get_memory_content_error_sets_content_error(self, mock_get):
        """When /content endpoint fails, contentError is set instead of raising."""
        meta_resp = _make_response(json_data={"memoryId": "m1", "processingStatus": "PROCESSING"})
        content_resp = _make_response(raise_for_status_error=True)
        mock_get.side_effect = [meta_resp, content_resp]
        result = self._make_client().get_memory("m1", include_content=True)
        assert result["success"] is True
        assert "content" not in result
        assert "contentError" in result
        assert "Failed to fetch content" in result["contentError"]

    @patch("dspy.utils.goodmem.client.requests.get")
    def test_get_memory_metadata_error_propagates(self, mock_get):
        mock_get.return_value = _make_response(raise_for_status_error=True)
        with pytest.raises(RuntimeError, match="GoodMem API error"):
            self._make_client().get_memory("nonexistent")

    # ---- list_memories ----

    @patch("dspy.utils.goodmem.client.requests.get")
    def test_list_memories_dict_response(self, mock_get):
        mock_get.return_value = _make_response(json_data={"memories": [{"memoryId": "m1"}]})
        result = self._make_client().list_memories("s1")
        assert result == [{"memoryId": "m1"}]

    @patch("dspy.utils.goodmem.client.requests.get")
    def test_list_memories_list_response(self, mock_get):
        mock_get.return_value = _make_response(json_data=[{"memoryId": "m1"}])
        result = self._make_client().list_memories("s1")
        assert result == [{"memoryId": "m1"}]

    @patch("dspy.utils.goodmem.client.requests.get")
    def test_list_memories_with_params(self, mock_get):
        mock_get.return_value = _make_response(json_data={"memories": []})
        self._make_client().list_memories(
            "s1", status_filter="COMPLETED", sort_by="created_at", sort_order="DESCENDING"
        )
        params = mock_get.call_args[1]["params"]
        assert params["statusFilter"] == "COMPLETED"
        assert params["sortBy"] == "created_at"
        assert params["sortOrder"] == "DESCENDING"

    # ---- delete_memory ----

    @patch("dspy.utils.goodmem.client.requests.delete")
    def test_delete_memory(self, mock_delete):
        mock_delete.return_value = _make_response(json_data={})
        result = self._make_client().delete_memory("mem-1")
        assert result["success"] is True
        assert result["memoryId"] == "mem-1"
        assert "message" in result

    @patch("dspy.utils.goodmem.client.requests.delete")
    def test_delete_memory_error_propagates(self, mock_delete):
        mock_delete.return_value = _make_response(raise_for_status_error=True)
        with pytest.raises(RuntimeError, match="GoodMem API error"):
            self._make_client().delete_memory("nonexistent")

    # ---- create_embedder ----

    @patch("dspy.utils.goodmem.client.requests.post")
    def test_create_embedder(self, mock_post):
        mock_post.return_value = _make_response(json_data={"embedderId": "e-new", "displayName": "Test"})
        result = self._make_client().create_embedder(
            display_name="Test",
            provider_type="OPENAI",
            endpoint_url="https://api.openai.com/v1",
            model_identifier="text-embedding-3-large",
            dimensionality=1536,
        )
        assert result["embedderId"] == "e-new"
        body = mock_post.call_args[1]["json"]
        assert body["displayName"] == "Test"
        assert body["dimensionality"] == 1536


# ---------------------------------------------------------------------------
# GoodMemRM tests
# ---------------------------------------------------------------------------


class TestGoodMemRM:
    """Tests for :class:`GoodMemRM`."""

    def test_init_requires_api_key(self):
        from dspy.retrievers.goodmem_rm import GoodMemRM

        with pytest.raises(ValueError, match="API key is required"):
            GoodMemRM(space_ids=["s1"], base_url="http://localhost:8080")

    def test_init_requires_base_url(self):
        from dspy.retrievers.goodmem_rm import GoodMemRM

        with pytest.raises(ValueError, match="base URL is required"):
            GoodMemRM(space_ids=["s1"], api_key="key")

    def test_init_from_env_vars(self):
        from dspy.retrievers.goodmem_rm import GoodMemRM

        with patch.dict(
            os.environ,
            {
                "GOODMEM_API_KEY": "env-key",
                "GOODMEM_BASE_URL": "https://env.test",
            },
        ):
            rm = GoodMemRM(space_ids=["s1"])
            assert rm._client.api_key == "env-key"
            assert rm._client.base_url == "https://env.test"

    def test_init_accepts_comma_separated_spaces(self):
        from dspy.retrievers.goodmem_rm import GoodMemRM

        rm = GoodMemRM(
            space_ids="s1, s2, s3",
            api_key="key",
            base_url="http://localhost:8080",
        )
        assert rm.space_ids == ["s1", "s2", "s3"]

    def test_default_k_is_3(self):
        from dspy.retrievers.goodmem_rm import GoodMemRM

        rm = GoodMemRM(space_ids=["s1"], api_key="key", base_url="http://localhost:8080")
        assert rm.k == 3

    def test_inherits_from_retrieve(self):
        import dspy
        from dspy.retrievers.goodmem_rm import GoodMemRM

        rm = GoodMemRM(space_ids=["s1"], api_key="key", base_url="http://localhost:8080")
        assert isinstance(rm, dspy.Retrieve)

    def test_forward_returns_dotdict_list(self):
        from dspy.retrievers.goodmem_rm import GoodMemRM

        rm = GoodMemRM(space_ids=["s1"], api_key="key", base_url="http://localhost:8080")
        mock_result = {
            "success": True,
            "results": [
                {"chunkText": "passage one", "relevanceScore": 0.9},
                {"chunkText": "passage two", "relevanceScore": 0.8},
            ],
            "memories": [],
            "totalResults": 2,
            "query": "test query",
        }
        with patch.object(rm._client, "retrieve_memories", return_value=mock_result):
            passages = rm.forward("test query", k=2)

        assert len(passages) == 2
        assert all(isinstance(p, dotdict) for p in passages)
        assert passages[0]["long_text"] == "passage one"
        assert passages[1].long_text == "passage two"

    def test_forward_multiple_queries(self):
        from dspy.retrievers.goodmem_rm import GoodMemRM

        rm = GoodMemRM(space_ids=["s1"], api_key="key", base_url="http://localhost:8080")
        mock_result = {
            "success": True,
            "results": [{"chunkText": "answer", "relevanceScore": 0.95}],
            "memories": [],
            "totalResults": 1,
            "query": "",
        }
        with patch.object(rm._client, "retrieve_memories", return_value=mock_result) as mock_ret:
            passages = rm.forward(["q1", "q2"], k=1)
        assert mock_ret.call_count == 2
        assert len(passages) == 2

    def test_forward_skips_empty_chunks(self):
        from dspy.retrievers.goodmem_rm import GoodMemRM

        rm = GoodMemRM(space_ids=["s1"], api_key="key", base_url="http://localhost:8080")
        mock_result = {
            "success": True,
            "results": [
                {"chunkText": "", "relevanceScore": 0.5},
                {"chunkText": "real text", "relevanceScore": 0.9},
            ],
            "memories": [],
            "totalResults": 2,
            "query": "q",
        }
        with patch.object(rm._client, "retrieve_memories", return_value=mock_result):
            passages = rm.forward("q")
        assert len(passages) == 1
        assert passages[0].long_text == "real text"

    def test_forward_empty_results(self):
        from dspy.retrievers.goodmem_rm import GoodMemRM

        rm = GoodMemRM(space_ids=["s1"], api_key="key", base_url="http://localhost:8080")
        mock_result = {
            "success": True,
            "results": [],
            "memories": [],
            "totalResults": 0,
            "query": "q",
        }
        with patch.object(rm._client, "retrieve_memories", return_value=mock_result):
            passages = rm.forward("q")
        assert passages == []

    def test_forward_filters_empty_queries(self):
        from dspy.retrievers.goodmem_rm import GoodMemRM

        rm = GoodMemRM(space_ids=["s1"], api_key="key", base_url="http://localhost:8080")
        mock_result = {
            "success": True,
            "results": [{"chunkText": "answer", "relevanceScore": 0.9}],
            "memories": [],
            "totalResults": 1,
            "query": "q",
        }
        with patch.object(rm._client, "retrieve_memories", return_value=mock_result) as mock_ret:
            passages = rm.forward(["", "real query", ""])
        assert mock_ret.call_count == 1
        assert len(passages) == 1


# ---------------------------------------------------------------------------
# Tool factory tests
# ---------------------------------------------------------------------------


class TestMakeGoodmemTools:
    """Tests for :func:`make_goodmem_tools`."""

    def _make_tools(self):
        from dspy.utils.goodmem import GoodMemClient, make_goodmem_tools

        client = GoodMemClient(api_key="k", base_url="http://localhost:8080", verify_ssl=False)
        return make_goodmem_tools(client)

    def test_returns_11_tools(self):
        assert len(self._make_tools()) == 11

    def test_tool_names(self):
        names = {t.__name__ for t in self._make_tools()}
        expected = {
            "create_space",
            "list_spaces",
            "get_space",
            "update_space",
            "delete_space",
            "create_memory",
            "retrieve_memories",
            "get_memory",
            "list_memories",
            "delete_memory",
            "list_embedders",
        }
        assert names == expected

    def test_tools_have_docstrings(self):
        for t in self._make_tools():
            assert t.__doc__, f"Tool {t.__name__} has no docstring"

    def test_tools_have_type_hints(self):
        import inspect

        for t in self._make_tools():
            hints = inspect.signature(t)
            assert hints.return_annotation is not inspect.Parameter.empty, f"Tool {t.__name__} missing return type hint"

    def test_dspy_tool_wrapping(self):
        import dspy

        for fn in self._make_tools():
            wrapped = dspy.Tool(fn)
            assert wrapped.name == fn.__name__
            assert wrapped.desc
