"""GoodMem retriever module for DSPy.

Provides :class:`GoodMemRM`, a ``dspy.Retrieve`` subclass that performs
semantic retrieval against one or more GoodMem spaces and returns results
in the ``dotdict({"long_text": ...})`` format expected by
``dspy.configure(rm=...)``.

Follows the same patterns as :mod:`dspy.retrievers.weaviate_rm` (dotdict
return) and :mod:`dspy.retrievers.databricks_rm` (HTTP via ``requests``).

Example::

    from dspy.retrievers.goodmem_rm import GoodMemRM

    rm = GoodMemRM(
        space_ids=["<space-uuid>"],
        api_key="gm_...",
        base_url="https://localhost:8080",
    )
    dspy.configure(rm=rm)

    retrieve = dspy.Retrieve(k=3)
    passages = retrieve("What is the main finding?").passages
"""

from __future__ import annotations

from typing import Any

import dspy
from dspy.dsp.utils import dotdict
from dspy.utils.goodmem.client import GoodMemClient


class GoodMemRM(dspy.Retrieve):
    """A retrieval module that uses GoodMem for semantic memory retrieval.

    Queries one or more GoodMem spaces and returns the top-*k* matching
    text chunks as ``dotdict({"long_text": chunk_text})`` items, which is
    the format consumed by ``dspy.Retrieve.forward()`` and
    ``dspy.configure(rm=...)``.

    Args:
        space_ids: One or more GoodMem space UUIDs to search.
        api_key: GoodMem API key. Falls back to the ``GOODMEM_API_KEY``
            environment variable.
        base_url: GoodMem API base URL. Falls back to the
            ``GOODMEM_BASE_URL`` environment variable.
        k: Default number of top passages to retrieve.
        verify_ssl: Whether to verify TLS certificates.
        poll_timeout: Maximum seconds to wait for indexing results.
        poll_interval: Seconds between polling attempts.
        include_memory_definition: Include full memory metadata in the
            underlying API call.
        wait_for_indexing: Retry when no results are found.
    """

    def __init__(
        self,
        space_ids: list[str] | str,
        api_key: str | None = None,
        base_url: str | None = None,
        k: int = 3,
        *,
        verify_ssl: bool = True,
        poll_timeout: int = 10,
        poll_interval: int = 5,
        include_memory_definition: bool = True,
        wait_for_indexing: bool = True,
    ) -> None:
        import os

        resolved_api_key = api_key or os.environ.get("GOODMEM_API_KEY")
        resolved_base_url = base_url or os.environ.get("GOODMEM_BASE_URL")

        if not resolved_api_key:
            raise ValueError(
                "A GoodMem API key is required.  Pass api_key= or set the GOODMEM_API_KEY environment variable."
            )
        if not resolved_base_url:
            raise ValueError(
                "A GoodMem base URL is required.  Pass base_url= or set the GOODMEM_BASE_URL environment variable."
            )

        if isinstance(space_ids, str):
            space_ids = [sid.strip() for sid in space_ids.split(",") if sid.strip()]
        self.space_ids = space_ids

        self._client = GoodMemClient(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            verify_ssl=verify_ssl,
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
        )
        self.include_memory_definition = include_memory_definition
        self.wait_for_indexing = wait_for_indexing

        super().__init__(k=k)

    def forward(
        self,
        query_or_queries: str | list[str],
        k: int | None = None,
        **kwargs: Any,
    ) -> list[dotdict]:
        """Search GoodMem for the top-*k* passages matching the query.

        Args:
            query_or_queries: A single query string or a list of queries.
            k: Number of top passages to retrieve (defaults to ``self.k``).

        Returns:
            A list of ``dotdict({"long_text": chunk_text})`` objects,
            compatible with ``dspy.Retrieve.forward()`` which converts
            them into a ``dspy.Prediction(passages=[...])``.
        """
        k = k if k is not None else self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]

        passages: list[dotdict] = []
        for query in queries:
            result = self._client.retrieve_memories(
                query,
                self.space_ids,
                max_results=k,
                include_memory_definition=self.include_memory_definition,
                wait_for_indexing=self.wait_for_indexing,
            )
            for item in result.get("results", []):
                text = item.get("chunkText", "")
                if text:
                    passages.append(dotdict({"long_text": text}))

        return passages
