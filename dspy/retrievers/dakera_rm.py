"""Dakera retrieval module for DSPy.

Dakera is a self-hosted, decay-weighted vector memory server with a REST API.
Run it locally with:

    docker run -p 3300:3300 -e DAKERA_API_KEY=demo ghcr.io/dakera-ai/dakera:latest

Project: https://dakera.ai
"""

from __future__ import annotations

import os
from typing import Any

import requests

import dspy
from dspy.primitives.prediction import Prediction


class DakeraRM(dspy.Retrieve):
    """A retrieval module that queries a Dakera memory server.

    Dakera is a self-hosted REST API for persistent, decay-weighted vector memory
    across agent sessions.  This module wraps the ``POST /v1/memory/search`` endpoint
    and optionally exposes a ``store`` helper for writing new memories.

    Args:
        agent_id (str): Identifier for the agent whose memory namespace to query.
            Every search and store call is scoped to this agent.
        url (str, optional): Base URL of the Dakera server.
            Defaults to the ``DAKERA_URL`` environment variable, or
            ``"http://localhost:3300"`` if the variable is not set.
        api_key (str, optional): Bearer token for authentication.
            Defaults to the ``DAKERA_API_KEY`` environment variable.
        k (int, optional): Number of top memories to retrieve. Default is 5.
        session_id (str, optional): Constrain recall to a specific session.
            When ``None`` (default) all sessions for ``agent_id`` are searched.
        timeout (float, optional): Request timeout in seconds. Default is 10.

    Raises:
        ValueError: If ``api_key`` is not supplied and ``DAKERA_API_KEY`` is not set.

    Examples:
        Basic usage as the default DSPy retriever::

            import dspy
            from dspy.retrievers.dakera_rm import DakeraRM

            lm = dspy.LM("openai/gpt-4o-mini")
            rm = DakeraRM(agent_id="my-agent", api_key="demo")
            dspy.configure(lm=lm, rm=rm)

            retrieve = dspy.Retrieve(k=3)
            result = retrieve("What did the user say about deadlines?")
            print(result.passages)

        Inline usage inside a DSPy module::

            class MemoryQA(dspy.Module):
                def __init__(self):
                    self.memory = DakeraRM(agent_id="qa-agent", k=5)
                    self.generate = dspy.ChainOfThought("context, question -> answer")

                def forward(self, question):
                    context = self.memory(question).passages
                    return self.generate(context=context, question=question)

        Writing a memory then retrieving it::

            rm = DakeraRM(agent_id="my-agent", api_key="demo")
            rm.store("The project deadline is next Friday.", tags=["deadline"])
            result = rm("When is the project deadline?")
            print(result.passages)
    """

    def __init__(
        self,
        agent_id: str,
        url: str | None = None,
        api_key: str | None = None,
        k: int = 5,
        session_id: str | None = None,
        timeout: float = 10.0,
    ):
        super().__init__(k=k)

        self.agent_id = agent_id
        self.url = (url or os.environ.get("DAKERA_URL") or "http://localhost:3300").rstrip("/")
        self.session_id = session_id
        self.timeout = timeout

        resolved_key = api_key or os.environ.get("DAKERA_API_KEY")
        if not resolved_key:
            raise ValueError(
                "A Dakera API key is required. Pass api_key= to DakeraRM or set the "
                "DAKERA_API_KEY environment variable."
            )
        self._api_key = resolved_key

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a POST request to the Dakera server and return parsed JSON.

        Args:
            path: API path relative to the server URL, e.g. ``"/v1/memory/search"``.
            payload: JSON-serialisable request body.

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            requests.HTTPError: On non-2xx HTTP responses.
            requests.ConnectionError: When the server is unreachable.
            requests.Timeout: When the request exceeds ``self.timeout`` seconds.
        """
        response = requests.post(
            f"{self.url}{path}",
            json=payload,
            headers=self._headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def forward(
        self,
        query: str | list[str],
        k: int | None = None,
        session_id: str | None = None,
    ) -> Prediction:
        """Search Dakera for memories relevant to *query*.

        When *query* is a list of strings every query is issued individually
        and the results are concatenated and de-duplicated (preserving order).

        Args:
            query: A query string, or a list of query strings.
            k: Number of memories to return per query.  Overrides the value of
                ``self.k`` when provided.
            session_id: Restrict search to this session.  Overrides ``self.session_id``
                when provided.

        Returns:
            ``dspy.Prediction`` with a ``passages`` attribute — a list of strings,
            each being the ``content`` field of a retrieved memory.  Additional
            attributes ``memory_ids`` and ``scores`` parallel ``passages`` and carry
            the Dakera memory IDs and relevance scores respectively.

        Raises:
            requests.HTTPError: On non-2xx HTTP responses from the Dakera server.
        """
        k = k if k is not None else self.k
        sid = session_id if session_id is not None else self.session_id

        queries: list[str] = [query] if isinstance(query, str) else query

        seen_ids: set[str] = set()
        passages: list[str] = []
        memory_ids: list[str] = []
        scores: list[float] = []

        for q in queries:
            payload: dict[str, Any] = {
                "agent_id": self.agent_id,
                "query": q,
                "top_k": k,
            }
            if sid is not None:
                payload["session_id"] = sid

            result = self._post("/v1/memory/search", payload)

            for hit in result.get("memories", []):
                mem = hit.get("memory", {})
                mid = mem.get("id", "")
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)
                passages.append(mem.get("content", ""))
                memory_ids.append(mid)
                scores.append(float(hit.get("score", 0.0)))

        return Prediction(passages=passages, memory_ids=memory_ids, scores=scores)

    # ------------------------------------------------------------------
    # Write helper (not required by DSPy, but useful for pipelines that
    # both read and write to memory)
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        session_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write a new memory to the Dakera server.

        Args:
            content: The text content to store.
            session_id: Optional session identifier.  Falls back to ``self.session_id``.
            tags: Optional list of string tags for the memory.
            metadata: Optional arbitrary key-value metadata dict.

        Returns:
            The parsed JSON response from the Dakera server (typically contains
            the new memory's ``id`` and ``created_at`` timestamp).

        Raises:
            requests.HTTPError: On non-2xx HTTP responses from the Dakera server.

        Example::

            rm = DakeraRM(agent_id="my-agent", api_key="demo")
            resp = rm.store("The sprint ends on Friday.", tags=["deadline", "sprint"])
            print(resp["id"])  # e.g. "mem_abc123"
        """
        sid = session_id if session_id is not None else self.session_id

        payload: dict[str, Any] = {
            "content": content,
            "agent_id": self.agent_id,
        }
        if sid is not None:
            payload["session_id"] = sid
        if tags is not None:
            payload["tags"] = tags
        if metadata is not None:
            payload["metadata"] = metadata

        return self._post("/v1/memory/store", payload)

    def forget(self, memory_ids: list[str]) -> dict[str, Any]:
        """Delete specific memories from the Dakera server.

        Args:
            memory_ids: List of memory IDs to delete (as returned in ``Prediction.memory_ids``).

        Returns:
            The parsed JSON response from the Dakera server.

        Raises:
            requests.HTTPError: On non-2xx HTTP responses from the Dakera server.
        """
        payload: dict[str, Any] = {
            "agent_id": self.agent_id,
            "memory_ids": memory_ids,
        }
        return self._post("/v1/memory/forget", payload)

    # ------------------------------------------------------------------
    # State persistence (required for dspy.load / module.save)
    # ------------------------------------------------------------------

    def dump_state(self) -> dict[str, Any]:
        state = super().dump_state()
        state.update(
            {
                "agent_id": self.agent_id,
                "url": self.url,
                "session_id": self.session_id,
                "timeout": self.timeout,
            }
        )
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        self.agent_id = state["agent_id"]
        self.url = state["url"]
        self.session_id = state.get("session_id")
        self.timeout = state.get("timeout", 10.0)
        # api_key is NOT persisted in state to avoid leaking secrets.
        # Re-supply it via the constructor or DAKERA_API_KEY env var on load.
