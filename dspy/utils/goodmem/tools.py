"""Factory that produces GoodMem tool functions for ``dspy.Tool`` / ``dspy.ReAct``.

Usage::

    from dspy.utils.goodmem import GoodMemClient, make_goodmem_tools

    client = GoodMemClient(api_key="...", base_url="...")
    tools = make_goodmem_tools(client)            # list of plain callables
    dspy_tools = [dspy.Tool(t) for t in tools]    # wrap for ReAct
"""

from __future__ import annotations

from typing import Any

from dspy.utils.goodmem.client import GoodMemClient


def make_goodmem_tools(client: GoodMemClient) -> list:
    """Return a list of plain Python functions that wrap every GoodMem operation.

    Each function carries full type hints and a docstring so that
    ``dspy.Tool(fn)`` can automatically infer name, description, and
    argument schemas.

    Args:
        client: An initialised :class:`GoodMemClient`.

    Returns:
        A list of 11 callable functions (one per GoodMem operation).
    """

    # ------------------------------------------------------------------
    # Space operations
    # ------------------------------------------------------------------

    def create_space(name: str, embedder_id: str) -> dict[str, Any]:
        """Create a new space or reuse an existing one.

        A space is a logical container for organising related memories,
        configured with an embedder that converts text to vector embeddings.
        If a space with the given name already exists its ID is returned
        instead of creating a duplicate.

        Args:
            name: A unique name for the space.
            embedder_id: The ID of the embedder model for similarity search.

        Returns:
            A dict with spaceId, name, reused flag, and a status message.
        """
        return client.create_space(name, embedder_id)

    def list_spaces() -> dict[str, Any]:
        """List all spaces in the GoodMem account.

        Returns each space with its ID, name, labels, embedder
        configuration, and access settings.

        Returns:
            A dict with a ``spaces`` list and ``totalSpaces`` count.
        """
        spaces = client.list_spaces()
        return {"success": True, "spaces": spaces, "totalSpaces": len(spaces)}

    def get_space(space_id: str) -> dict[str, Any]:
        """Fetch a specific space by its ID.

        Args:
            space_id: The UUID of the space.

        Returns:
            A dict with the full space object.
        """
        return {"success": True, "space": client.get_space(space_id)}

    def update_space(
        space_id: str,
        name: str = "",
        public_read: bool = False,
        replace_labels: str = "",
        merge_labels: str = "",
    ) -> dict[str, Any]:
        """Update an existing space.

        You can change the name, public-read flag, and labels.  Embedders
        and chunking configuration are immutable after creation.

        Replace Labels and Merge Labels are mutually exclusive.  Pass them
        as JSON strings (e.g. ``'{\"project\": \"legal\"}'``).

        Args:
            space_id: The UUID of the space to update.
            name: New name for the space (empty string to keep current).
            public_read: Allow unauthenticated users to read this space.
            replace_labels: JSON string of labels to replace all existing labels.
            merge_labels: JSON string of labels to merge into existing labels.

        Returns:
            A dict with the updated space object and a status message.
        """
        import json as _json

        rl = _json.loads(replace_labels) if replace_labels else None
        ml = _json.loads(merge_labels) if merge_labels else None

        space = client.update_space(
            space_id,
            name=name or None,
            public_read=public_read,
            replace_labels=rl,
            merge_labels=ml,
        )
        return {"success": True, "space": space, "message": "Space updated successfully"}

    def delete_space(space_id: str) -> dict[str, Any]:
        """Permanently delete a space and all associated memories, chunks,
        and vector embeddings.

        Args:
            space_id: The UUID of the space to delete.

        Returns:
            A dict confirming deletion.
        """
        return client.delete_space(space_id)

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------

    def create_memory(
        space_id: str,
        text_content: str = "",
        file_path: str = "",
        source: str = "",
        author: str = "",
        tags: str = "",
        metadata: str = "",
    ) -> dict[str, Any]:
        """Store a document as a new memory in a space.

        The memory is processed asynchronously -- chunked into searchable
        pieces and embedded into vectors.  Supply either a file path (PDF,
        DOCX, image, etc.) or plain text.  If both are provided the file
        takes priority.

        Metadata fields (source, author, tags) are merged with the optional
        JSON ``metadata`` string.

        Args:
            space_id: The UUID of the space to store the memory in.
            text_content: Plain text content (sent as text/plain).
            file_path: Absolute path to a file to upload (auto-detects MIME type).
            source: Where this memory came from (stored in metadata.source).
            author: The author of the content (stored in metadata.author).
            tags: Comma-separated tags (stored as metadata.tags array).
            metadata: Extra key-value metadata as a JSON string.

        Returns:
            A dict with memoryId, processing status, and a message.
        """
        import json as _json

        md = _json.loads(metadata) if metadata else None
        return client.create_memory(
            space_id,
            text_content=text_content or None,
            file_path=file_path or None,
            source=source or None,
            author=author or None,
            tags=tags or None,
            metadata=md,
        )

    def retrieve_memories(
        query: str,
        space_ids: str,
        max_results: int = 5,
        include_memory_definition: bool = True,
        wait_for_indexing: bool = True,
    ) -> dict[str, Any]:
        """Perform similarity-based semantic retrieval across one or more spaces.

        Returns matching chunks ranked by relevance, with optional full
        memory definitions.  When *wait_for_indexing* is true the call
        retries until results appear or the polling timeout is reached.

        Args:
            query: A natural-language query for semantic search.
            space_ids: Comma-separated space UUIDs to search across.
            max_results: Maximum number of matching chunks to return.
            include_memory_definition: Fetch full memory metadata alongside chunks.
            wait_for_indexing: Retry when no results are found (memories may be processing).

        Returns:
            A dict with results list, memories list, and totalResults.
        """
        return client.retrieve_memories(
            query,
            space_ids,
            max_results=max_results,
            include_memory_definition=include_memory_definition,
            wait_for_indexing=wait_for_indexing,
        )

    def get_memory(
        memory_id: str,
        include_content: bool = True,
    ) -> dict[str, Any]:
        """Fetch a specific memory by its ID.

        Returns metadata, processing status, and optionally the original
        content.

        Args:
            memory_id: The UUID of the memory.
            include_content: Also fetch the original document content.

        Returns:
            A dict with memory metadata and optionally content.
        """
        return client.get_memory(memory_id, include_content=include_content)

    def list_memories(
        space_id: str,
        status_filter: str = "",
        include_content: bool = False,
        sort_by: str = "",
        sort_order: str = "",
    ) -> dict[str, Any]:
        """List all memories in a space.

        Optionally filter by processing status (PENDING, PROCESSING,
        COMPLETED, FAILED) and include original content.

        Args:
            space_id: The UUID of the space.
            status_filter: Filter by processing status (empty for all).
            include_content: Include original document content.
            sort_by: Field to sort by (created_at or updated_at).
            sort_order: ASCENDING or DESCENDING.

        Returns:
            A dict with memories list, totalMemories count, and spaceId.
        """
        memories = client.list_memories(
            space_id,
            status_filter=status_filter or None,
            include_content=include_content,
            sort_by=sort_by or None,
            sort_order=sort_order or None,
        )
        return {
            "success": True,
            "memories": memories,
            "totalMemories": len(memories),
            "spaceId": space_id,
        }

    def delete_memory(memory_id: str) -> dict[str, Any]:
        """Permanently delete a memory and its associated chunks and
        vector embeddings.

        Args:
            memory_id: The UUID of the memory to delete.

        Returns:
            A dict confirming deletion.
        """
        return client.delete_memory(memory_id)

    # ------------------------------------------------------------------
    # Embedders
    # ------------------------------------------------------------------

    def list_embedders() -> dict[str, Any]:
        """List all available embedder models.

        Embedders convert text into vector representations for similarity
        search.  Use the returned embedder ID when creating a new space.

        Returns:
            A dict with embedders list and totalEmbedders count.
        """
        embedders = client.list_embedders()
        return {
            "success": True,
            "embedders": embedders,
            "totalEmbedders": len(embedders),
        }

    return [
        create_space,
        list_spaces,
        get_space,
        update_space,
        delete_space,
        create_memory,
        retrieve_memories,
        get_memory,
        list_memories,
        delete_memory,
        list_embedders,
    ]
