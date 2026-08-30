from __future__ import annotations

import hashlib
from typing import Any, Callable

import dspy
from dspy.dsp.utils import dotdict

try:
    from rostam import Rostam, RostamError
except ImportError as err:
    raise ImportError(
        "The 'rostam' package is required to use RostamRM. Install it with `pip install rostam-client`",
    ) from err

Embedder = Callable[[str], list[float]]


def _content_id(text: str) -> str:
    """A stable string id derived from document content, used when the caller
    doesn't supply explicit ids to :meth:`RostamRM.index`."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def _to_point_id(external_id: str) -> int:
    """Map an external string id to the uint64 point id Rostam stores.

    A purely-numeric string that fits in 64 bits is used verbatim; anything
    else is hashed (BLAKE2b, 8 bytes) to a stable id, so repeated ``index()``
    calls for the same external id address the same point.
    """
    if external_id.isdigit():
        value = int(external_id)
        if value < (1 << 64):
            return value
    return int.from_bytes(hashlib.blake2b(external_id.encode("utf-8"), digest_size=8).digest(), "big")


def _scalar_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Keep only the metadata values Rostam can store as scalar/array payload
    fields (str, int, float, bool, or a homogeneous list of them)."""
    out: dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif (
            isinstance(v, (list, tuple))
            and v
            and all(isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in v)
        ):
            out[k] = list(v)
    return out


class RostamRM(dspy.Module):
    """A retrieval module that uses Rostam to return the top passages for a given query.

    Rostam (https://github.com/rostamlabs/rostam) is an open-source vector database.
    Unlike vendor-hosted search indexes, Rostam's search endpoint takes a query
    *vector* rather than a query string, so ``RostamRM`` is constructed with an
    ``embedder`` callable used to embed both the indexed documents and incoming
    queries.

    Args:
        rostam_collection_name (str): The name of the Rostam collection to search.
        rostam_client (Rostam, optional): An existing ``rostam.Rostam`` client
            instance. Exactly one of ``rostam_client`` or ``url`` must be given.
        url (str, optional): A Rostam server target (e.g. ``"http://localhost:8080"``
            or ``"tcp://localhost:7000"``) used to construct a client when
            ``rostam_client`` isn't supplied.
        embedder (Callable[[str], list[float]]): Embeds a query or document string
            into the vector Rostam indexes and searches over.
        k (int, optional): The default number of top passages to retrieve. Defaults to 3.
        auto_create (bool, optional): Create the collection on first ``index()`` call
            if it doesn't already exist. Defaults to True.
        metric (str, optional): Distance metric used when auto-creating the collection.
            Defaults to "cosine".
        filter (dict, optional): A default Rostam metadata filter applied to every
            search, overridable per call. Defaults to None.

    Examples:
        Below is a code snippet that shows how to use Rostam as the default retriever:
        ```python
        from rostam import Rostam

        rostam_client = Rostam("http://localhost:8080")
        retriever_model = RostamRM("my_collection_name", rostam_client=rostam_client, embedder=embed_fn)
        dspy.configure(lm=llm, rm=retriever_model)

        retrieve = dspy.Retrieve(k=1)
        topK_passages = retrieve("what are the stages in planning, sanctioning and execution of public works").passages
        ```

        Below is a code snippet that shows how to use Rostam in the forward() function of a module:
        ```python
        self.retrieve = RostamRM("my_collection_name", rostam_client=rostam_client, embedder=embed_fn, k=num_passages)
        ```
    """

    def __init__(
        self,
        rostam_collection_name: str,
        rostam_client: Rostam | None = None,
        url: str | None = None,
        *,
        embedder: Embedder,
        k: int = 3,
        auto_create: bool = True,
        metric: str = "cosine",
        filter: dict[str, Any] | None = None,
    ):
        if (rostam_client is None) == (url is None):
            raise ValueError("Exactly one of `rostam_client` or `url` must be specified.")

        super().__init__()
        self._collection = rostam_collection_name
        self._client = rostam_client if rostam_client is not None else Rostam(url)
        self._embedder = embedder
        self.k = k
        self._auto_create = auto_create
        self._metric = metric
        self._filter = filter
        self._created = False

    def forward(
        self,
        query_or_queries: str | list[str],
        k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[dotdict]:
        """Search Rostam for the top-k passages for query or queries.

        Args:
            query_or_queries (Union[str, list[str]]): The query or queries to search for.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.
            filter (Optional[dict]): A metadata filter overriding the one set at
                construction time.

        Returns:
            list[dotdict]: The retrieved passages, each a ``dotdict`` with a
            ``long_text`` field. This matches the retrieval-model contract
            ``dspy.Retrieve`` consumes (it reads ``.long_text`` off each item and
            wraps them into a ``Prediction(passages=[...])``), so ``RostamRM`` works
            both as the configured ``rm`` and when called directly.
        """
        k = k if k is not None else self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]
        active_filter = filter if filter is not None else self._filter

        passages: list[str] = []
        for query in queries:
            vector = self._embedder(query)
            hits = self._client.search_docs(self._collection, vector, k, filter=active_filter)
            passages.extend(hit.content for hit in hits)

        return [dotdict(long_text=content) for content in passages]

    def index(
        self,
        texts: list[str],
        *,
        embeddings: list[list[float]] | None = None,
        ids: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Load documents into the backing Rostam collection.

        Embeds via the configured ``embedder`` when ``embeddings`` aren't supplied.
        Returns the ids used (deterministically derived from content when ``ids``
        is omitted).
        """
        texts = list(texts)
        if not texts:
            return []

        # Validate every supplied parallel input against texts BEFORE touching the
        # server: otherwise a mismatch would create the collection and upsert the
        # first documents before the loop's length check raised, leaving the store
        # partially mutated by a call that reports failure.
        n = len(texts)
        for name, seq in (("embeddings", embeddings), ("ids", ids), ("metadatas", metadatas)):
            if seq is not None and len(seq) != n:
                raise ValueError(f"index(): {name} has length {len(seq)}, expected {n} (one per text)")

        vectors = list(embeddings) if embeddings is not None else [self._embedder(t) for t in texts]
        if vectors:
            self._ensure_collection(len(vectors[0]))

        if ids is None:
            ids = [_content_id(t) for t in texts]
        else:
            ids = list(ids)
        metadatas = list(metadatas) if metadatas is not None else [{} for _ in texts]

        for text, vector, metadata, doc_id in zip(texts, vectors, metadatas, ids, strict=True):
            self._client.upsert(
                self._collection, _to_point_id(doc_id), vector, content=text, metadata=_scalar_metadata(metadata)
            )
        return ids

    def _ensure_collection(self, dim: int) -> None:
        """Create the collection on first index() call (idempotent). No-op if
        ``auto_create`` is off or we already created it this session."""
        if not self._auto_create or self._created:
            return
        try:
            self._client.create_collection(self._collection, dim, metric=self._metric)
        except RostamError as e:
            # Already-exists is fine; anything else propagates.
            if "exist" not in (str(e) or "").lower():
                raise
        self._created = True
