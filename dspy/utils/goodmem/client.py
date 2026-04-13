"""GoodMem API client for DSPy.

Provides a Python HTTP wrapper around all 11 GoodMem API operations using the
``requests`` library.  Handles X-API-Key authentication, trailing-slash removal,
NDJSON response parsing, polling with a configurable timeout, PDF base64
encoding, idempotent space creation, and response-format variability.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from typing import Any, ClassVar

import requests


class GoodMemClient:
    """Low-level HTTP client for the GoodMem REST API.

    Args:
        api_key: GoodMem API key (sent as ``X-API-Key`` header).
        base_url: Base URL of the GoodMem API server
            (e.g. ``https://api.goodmem.ai`` or ``http://localhost:8080``).
        verify_ssl: Whether to verify TLS certificates. Defaults to ``True``.
        poll_timeout: Maximum seconds to poll when waiting for indexing
            results during retrieval. Defaults to ``10``.
        poll_interval: Seconds between polling attempts. Defaults to ``5``.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        verify_ssl: bool = True,
        poll_timeout: int = 10,
        poll_interval: int = 5,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.verify_ssl = verify_ssl
        self.poll_timeout = poll_timeout
        self.poll_interval = poll_interval

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self, *, accept: str = "application/json") -> dict[str, str]:
        return {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": accept,
        }

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _raise_for_status(self, resp: requests.Response) -> None:
        if not resp.ok:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"GoodMem API error {resp.status_code}: {detail}")

    @staticmethod
    def _parse_ndjson(text: str) -> list[dict[str, Any]]:
        """Parse an NDJSON (or SSE-wrapped NDJSON) response body."""
        items: list[dict[str, Any]] = []
        for raw_line in text.strip().split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            # Strip SSE ``data:`` prefix if present.
            if line.startswith("data:"):
                line = line[5:].strip()
            # Skip SSE control lines.
            if line.startswith("event:") or line == "":
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return items

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------

    # Default chunking configuration used when creating spaces.
    DEFAULT_CHUNKING_CONFIG: ClassVar[dict[str, Any]] = {
        "recursive": {
            "chunkSize": 256,
            "chunkOverlap": 25,
            "separators": ["\n\n", "\n", ". ", " ", ""],
            "keepStrategy": "KEEP_END",
            "separatorIsRegex": False,
            "lengthMeasurement": "CHARACTER_COUNT",
        }
    }

    def create_space(
        self,
        name: str,
        embedder_id: str,
        *,
        chunking_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new space, or return an existing one with the same name.

        Implements idempotent creation: lists existing spaces first and reuses
        a space that already has the requested *name*.

        Args:
            name: Unique space name.
            embedder_id: Embedder UUID to associate with the space.
            chunking_config: Optional chunking configuration dict.  When
                ``None`` a sensible default (recursive, 256 chars) is used.

        Returns:
            A dict with ``spaceId``, ``name``, ``embedderId``, ``reused``
            flag, and a status ``message``.

        Raises:
            RuntimeError: If the GoodMem API returns an error response.
        """
        try:
            spaces = self.list_spaces()
            for s in spaces:
                if s.get("name") == name:
                    return {
                        "success": True,
                        "spaceId": s["spaceId"],
                        "name": s["name"],
                        "embedderId": embedder_id,
                        "message": "Space already exists, reusing existing space",
                        "reused": True,
                    }
        except Exception:
            pass  # If listing fails, proceed to create.

        body: dict[str, Any] = {
            "name": name,
            "spaceEmbedders": [{"embedderId": embedder_id}],
            "defaultChunkingConfig": chunking_config or self.DEFAULT_CHUNKING_CONFIG,
        }
        resp = requests.post(
            self._url("/v1/spaces"),
            headers=self._headers(),
            json=body,
            verify=self.verify_ssl,
        )
        self._raise_for_status(resp)
        data = resp.json()
        return {
            "success": True,
            "spaceId": data["spaceId"],
            "name": data["name"],
            "embedderId": embedder_id,
            "message": "Space created successfully",
            "reused": False,
        }

    def list_spaces(self) -> list[dict[str, Any]]:
        """Return a list of all spaces.

        Returns:
            A list of space dicts, each containing ``spaceId``, ``name``,
            and configuration details.

        Raises:
            RuntimeError: If the GoodMem API returns an error response.
        """
        resp = requests.get(
            self._url("/v1/spaces"),
            headers=self._headers(),
            verify=self.verify_ssl,
        )
        self._raise_for_status(resp)
        body = resp.json()
        return body if isinstance(body, list) else body.get("spaces", [])

    def get_space(self, space_id: str) -> dict[str, Any]:
        """Fetch a single space by ID.

        Args:
            space_id: The UUID of the space to fetch.

        Returns:
            The full space object as a dict.

        Raises:
            RuntimeError: If the GoodMem API returns an error response.
        """
        resp = requests.get(
            self._url(f"/v1/spaces/{space_id}"),
            headers=self._headers(),
            verify=self.verify_ssl,
        )
        self._raise_for_status(resp)
        return resp.json()

    def update_space(
        self,
        space_id: str,
        *,
        name: str | None = None,
        public_read: bool | None = None,
        replace_labels: dict[str, str] | None = None,
        merge_labels: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Update a space's name, labels, or public access settings.

        Args:
            space_id: The UUID of the space to update.
            name: New name for the space, or ``None`` to keep the current name.
            public_read: Whether to allow unauthenticated read access.
            replace_labels: Labels dict that replaces all existing labels.
            merge_labels: Labels dict that merges into existing labels.

        Returns:
            The updated space object as a dict.

        Raises:
            ValueError: If both *replace_labels* and *merge_labels* are
                provided (they are mutually exclusive).
            RuntimeError: If the GoodMem API returns an error response.
        """
        if replace_labels and merge_labels:
            raise ValueError("Cannot use both replace_labels and merge_labels at the same time.")
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if public_read is not None:
            body["publicRead"] = public_read
        if replace_labels:
            body["replaceLabels"] = replace_labels
        if merge_labels:
            body["mergeLabels"] = merge_labels

        resp = requests.put(
            self._url(f"/v1/spaces/{space_id}"),
            headers=self._headers(),
            json=body,
            verify=self.verify_ssl,
        )
        self._raise_for_status(resp)
        return resp.json()

    def delete_space(self, space_id: str) -> dict[str, Any]:
        """Delete a space and all associated data.

        Args:
            space_id: The UUID of the space to delete.

        Returns:
            A confirmation dict with ``success``, ``spaceId``, and ``message``.

        Raises:
            RuntimeError: If the GoodMem API returns an error response.
        """
        resp = requests.delete(
            self._url(f"/v1/spaces/{space_id}"),
            headers=self._headers(),
            verify=self.verify_ssl,
        )
        self._raise_for_status(resp)
        return {"success": True, "spaceId": space_id, "message": "Space deleted successfully"}

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------

    @staticmethod
    def _get_mime_type(extension: str) -> str | None:
        """Map a file extension to a MIME type (matching the reference)."""
        mime_map: dict[str, str] = {
            "pdf": "application/pdf",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp",
            "txt": "text/plain",
            "html": "text/html",
            "md": "text/markdown",
            "csv": "text/csv",
            "json": "application/json",
            "xml": "application/xml",
            "doc": "application/msword",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "xls": "application/vnd.ms-excel",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "ppt": "application/vnd.ms-powerpoint",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }
        ext = extension.lower().lstrip(".")
        return mime_map.get(ext) or mimetypes.guess_type(f"file.{ext}")[0]

    def create_memory(
        self,
        space_id: str,
        *,
        text_content: str | None = None,
        file_path: str | None = None,
        source: str | None = None,
        author: str | None = None,
        tags: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new memory from text or a file.

        If *file_path* is provided it takes priority over *text_content*.
        Binary files (PDF, images, ...) are base64-encoded; text-type files
        are read as UTF-8.

        Args:
            space_id: The UUID of the space to store the memory in.
            text_content: Plain text content (sent as ``text/plain``).
            file_path: Path to a file to upload. MIME type is auto-detected
                from the extension.
            source: Value for ``metadata.source``.
            author: Value for ``metadata.author``.
            tags: Comma-separated tags stored as ``metadata.tags`` array.
            metadata: Additional key-value metadata merged with the above.

        Returns:
            A dict with ``memoryId``, ``spaceId``, processing ``status``,
            ``contentType``, and a ``message``.

        Raises:
            ValueError: If neither *text_content* nor *file_path* is provided.
            RuntimeError: If the GoodMem API returns an error response.
        """
        body: dict[str, Any] = {"spaceId": space_id}

        if file_path:
            ext = os.path.splitext(file_path)[1].lstrip(".")
            mime = self._get_mime_type(ext) or "application/octet-stream"

            if mime.startswith("text/"):
                with open(file_path, encoding="utf-8") as fh:
                    body["contentType"] = mime
                    body["originalContent"] = fh.read()
            else:
                with open(file_path, "rb") as fh:
                    body["contentType"] = mime
                    body["originalContentB64"] = base64.b64encode(fh.read()).decode("ascii")
        elif text_content:
            body["contentType"] = "text/plain"
            body["originalContent"] = text_content
        else:
            raise ValueError("No content provided. Supply text_content or file_path.")

        # Merge metadata fields.
        merged: dict[str, Any] = {}
        if metadata:
            merged.update(metadata)
        if source:
            merged["source"] = source
        if author:
            merged["author"] = author
        if tags:
            merged["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
        if merged:
            body["metadata"] = merged

        resp = requests.post(
            self._url("/v1/memories"),
            headers=self._headers(),
            json=body,
            verify=self.verify_ssl,
        )
        self._raise_for_status(resp)
        data = resp.json()
        return {
            "success": True,
            "memoryId": data.get("memoryId"),
            "spaceId": data.get("spaceId"),
            "status": data.get("processingStatus", "PENDING"),
            "contentType": body["contentType"],
            "message": "Memory created successfully",
        }

    def retrieve_memories(
        self,
        query: str,
        space_ids: str | list[str],
        *,
        max_results: int = 5,
        include_memory_definition: bool = True,
        wait_for_indexing: bool = True,
    ) -> dict[str, Any]:
        """Semantic retrieval across one or more spaces.

        Parses the NDJSON streaming response.  When *wait_for_indexing* is
        ``True`` the call polls up to ``self.poll_timeout`` seconds (default 10)
        if zero results are returned (memories may still be processing).

        Args:
            query: Natural-language query for semantic search.
            space_ids: A single space UUID, a comma-separated string, or a
                list of space UUIDs to search across.
            max_results: Maximum number of matching chunks to return.
            include_memory_definition: Fetch full memory metadata alongside
                each matched chunk.
            wait_for_indexing: Retry when no results are found, up to
                ``self.poll_timeout`` seconds.

        Returns:
            A dict with ``results`` (list of chunk dicts), ``memories``
            (list of memory definition dicts), ``totalResults``, ``query``,
            and ``resultSetId``.

        Raises:
            ValueError: If *space_ids* is empty after parsing.
            RuntimeError: If the GoodMem API returns an error response.
        """
        if isinstance(space_ids, str):
            space_ids = [sid.strip() for sid in space_ids.split(",") if sid.strip()]
        else:
            space_ids = [sid for sid in space_ids if sid and sid.strip()]

        if not space_ids:
            raise ValueError("At least one space ID is required.")

        space_keys = [{"spaceId": sid} for sid in space_ids]
        request_body: dict[str, Any] = {
            "message": query,
            "spaceKeys": space_keys,
            "requestedSize": max_results,
            "fetchMemory": include_memory_definition,
        }

        start = time.time()

        while True:
            resp = requests.post(
                self._url("/v1/memories:retrieve"),
                headers=self._headers(accept="application/x-ndjson"),
                json=request_body,
                verify=self.verify_ssl,
            )
            self._raise_for_status(resp)

            items = self._parse_ndjson(resp.text)

            results: list[dict[str, Any]] = []
            memories: list[dict[str, Any]] = []
            result_set_id = ""

            for item in items:
                if "resultSetBoundary" in item:
                    result_set_id = item["resultSetBoundary"].get("resultSetId", "")
                elif "memoryDefinition" in item:
                    memories.append(item["memoryDefinition"])
                elif "retrievedItem" in item:
                    ri = item["retrievedItem"]
                    chunk_data = ri.get("chunk", {})
                    inner_chunk = chunk_data.get("chunk", {})
                    results.append(
                        {
                            "chunkId": inner_chunk.get("chunkId"),
                            "chunkText": inner_chunk.get("chunkText"),
                            "memoryId": inner_chunk.get("memoryId"),
                            "relevanceScore": chunk_data.get("relevanceScore"),
                            "memoryIndex": chunk_data.get("memoryIndex"),
                        }
                    )

            if results or not wait_for_indexing:
                return {
                    "success": True,
                    "resultSetId": result_set_id,
                    "results": results,
                    "memories": memories,
                    "totalResults": len(results),
                    "query": query,
                }

            elapsed = time.time() - start
            if elapsed >= self.poll_timeout:
                return {
                    "success": True,
                    "resultSetId": result_set_id,
                    "results": results,
                    "memories": memories,
                    "totalResults": 0,
                    "query": query,
                    "message": (
                        f"No results found after waiting {self.poll_timeout} seconds "
                        "for indexing. Memories may still be processing."
                    ),
                }

            time.sleep(self.poll_interval)

    def get_memory(self, memory_id: str, *, include_content: bool = True) -> dict[str, Any]:
        """Fetch a memory's metadata and optionally its content.

        Args:
            memory_id: The UUID of the memory.
            include_content: Also fetch the original document content via a
                second API call.

        Returns:
            A dict with ``memory`` (metadata) and optionally ``content``
            or ``contentError`` if the content fetch failed.

        Raises:
            RuntimeError: If the GoodMem API returns an error response for
                the metadata request.
        """
        resp = requests.get(
            self._url(f"/v1/memories/{memory_id}"),
            headers=self._headers(),
            verify=self.verify_ssl,
        )
        self._raise_for_status(resp)
        result: dict[str, Any] = {"success": True, "memory": resp.json()}

        if include_content:
            try:
                content_resp = requests.get(
                    self._url(f"/v1/memories/{memory_id}/content"),
                    headers=self._headers(),
                    verify=self.verify_ssl,
                )
                self._raise_for_status(content_resp)
                result["content"] = content_resp.json()
            except Exception as exc:
                result["contentError"] = f"Failed to fetch content: {exc}"

        return result

    def list_memories(
        self,
        space_id: str,
        *,
        status_filter: str | None = None,
        include_content: bool = False,
        sort_by: str | None = None,
        sort_order: str | None = None,
    ) -> list[dict[str, Any]]:
        """List all memories in a space with optional filtering and sorting.

        Args:
            space_id: The UUID of the space to list memories from.
            status_filter: Filter by processing status (``PENDING``,
                ``PROCESSING``, ``COMPLETED``, or ``FAILED``).
            include_content: Include original document content in each memory.
            sort_by: Field to sort by (``created_at`` or ``updated_at``).
            sort_order: ``ASCENDING`` or ``DESCENDING``.

        Returns:
            A list of memory dicts.

        Raises:
            RuntimeError: If the GoodMem API returns an error response.
        """
        params: dict[str, str] = {}
        if include_content:
            params["includeContent"] = "true"
        if status_filter:
            params["statusFilter"] = status_filter
        if sort_by:
            params["sortBy"] = sort_by
        if sort_order:
            params["sortOrder"] = sort_order

        resp = requests.get(
            self._url(f"/v1/spaces/{space_id}/memories"),
            headers=self._headers(),
            params=params,
            verify=self.verify_ssl,
        )
        self._raise_for_status(resp)
        body = resp.json()
        return body if isinstance(body, list) else body.get("memories", [])

    def delete_memory(self, memory_id: str) -> dict[str, Any]:
        """Permanently delete a memory.

        Args:
            memory_id: The UUID of the memory to delete.

        Returns:
            A confirmation dict with ``success``, ``memoryId``, and ``message``.

        Raises:
            RuntimeError: If the GoodMem API returns an error response.
        """
        resp = requests.delete(
            self._url(f"/v1/memories/{memory_id}"),
            headers=self._headers(),
            verify=self.verify_ssl,
        )
        self._raise_for_status(resp)
        return {
            "success": True,
            "memoryId": memory_id,
            "message": "Memory deleted successfully",
        }

    # ------------------------------------------------------------------
    # Embedders
    # ------------------------------------------------------------------

    def create_embedder(
        self,
        display_name: str,
        provider_type: str,
        endpoint_url: str,
        model_identifier: str,
        dimensionality: int,
        distribution_type: str = "DENSE",
        *,
        api_path: str | None = None,
        credentials: dict[str, Any] | None = None,
        supported_modalities: list[str] | None = None,
        labels: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Register a new embedder configuration on the GoodMem server.

        Args:
            display_name: User-facing name for the embedder.
            provider_type: One of OPENAI, VLLM, TEI, LLAMA_CPP, VOYAGE, COHERE, JINA.
            endpoint_url: Base URL of the embedding service.
            model_identifier: Model name/identifier.
            dimensionality: Output vector dimensions.
            distribution_type: DENSE or SPARSE.
            api_path: Optional API sub-path (defaults vary by provider).
            credentials: Optional authentication payload.
            supported_modalities: e.g. ["TEXT"]. Defaults to TEXT.
            labels: Optional key-value labels.

        Returns:
            The created embedder dict (includes ``embedderId``).
        """
        body: dict[str, Any] = {
            "displayName": display_name,
            "providerType": provider_type,
            "endpointUrl": endpoint_url,
            "modelIdentifier": model_identifier,
            "dimensionality": dimensionality,
            "distributionType": distribution_type,
        }
        if api_path:
            body["apiPath"] = api_path
        if credentials:
            body["credentials"] = credentials
        if supported_modalities:
            body["supportedModalities"] = supported_modalities
        if labels:
            body["labels"] = labels

        resp = requests.post(
            self._url("/v1/embedders"),
            headers=self._headers(),
            json=body,
            verify=self.verify_ssl,
        )
        self._raise_for_status(resp)
        return resp.json()

    def list_embedders(self) -> list[dict[str, Any]]:
        """Return all available embedder models.

        Returns:
            A list of embedder dicts, each containing ``embedderId`` and
            model configuration.

        Raises:
            RuntimeError: If the GoodMem API returns an error response.
        """
        resp = requests.get(
            self._url("/v1/embedders"),
            headers=self._headers(),
            verify=self.verify_ssl,
        )
        self._raise_for_status(resp)
        body = resp.json()
        return body if isinstance(body, list) else body.get("embedders", [])
