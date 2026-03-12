from typing import Any

import requests

from dspy.clients.cache import request_cache
from dspy.dsp.utils import dotdict

# TODO: Ideally, this takes the name of the index and looks up its port.


class ColBERTv2:
    """Query a hosted ColBERTv2 retrieval server over HTTP.

    This wrapper talks to a ColBERTv2 endpoint that returns top-k retrieval results
    for a text query. Responses can be returned either as raw passage strings or as
    ``dotdict`` objects that preserve the metadata returned by the server.

    Args:
        url: Base URL for the retrieval service.
        port: Optional port appended to ``url``.
        post_requests: Whether to use POST requests instead of GET requests.

    Examples:
        ```python
        import dspy

        retriever = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
        results = retriever("Who wrote Pride and Prejudice?", k=3)
        print(results[0].long_text)
        ```
    """

    def __init__(
        self,
        url: str = "http://0.0.0.0",
        port: str | int | None = None,
        post_requests: bool = False,
    ):
        """Initialize a ColBERTv2 HTTP client.

        Args:
            url: Base URL for the retrieval server.
            port: Optional port appended to ``url``.
            post_requests: Whether retrieval requests should use POST instead of GET.
        """
        self.post_requests = post_requests
        self.url = f"{url}:{port}" if port else url

    def __call__(
        self,
        query: str,
        k: int = 10,
        simplify: bool = False,
    ) -> list[str] | list[dotdict]:
        """Retrieve top-k passages for a query.

        Args:
            query: Query string sent to the retrieval server.
            k: Maximum number of results to request.
            simplify: Whether to return only passage text instead of structured
                result objects.

        Returns:
            list[str] | list[dotdict]: Retrieved passages, either as strings or as
                ``dotdict`` objects containing the server response fields.
        """
        if self.post_requests:
            topk: list[dict[str, Any]] = colbertv2_post_request(self.url, query, k)
        else:
            topk: list[dict[str, Any]] = colbertv2_get_request(self.url, query, k)

        if simplify:
            return [psg["long_text"] for psg in topk]

        return [dotdict(psg) for psg in topk]


@request_cache()
def colbertv2_get_request_v2(url: str, query: str, k: int):
    """Send a GET request to a hosted ColBERTv2 server.

    The response is expected to contain a ``topk`` field whose items include a
    ``text`` value. Each returned item is normalized to also expose that text under
    ``long_text`` for compatibility with DSPy retrievers.

    Args:
        url: Retrieval endpoint URL.
        query: Query string to search for.
        k: Maximum number of results to return. Hosted ColBERTv2 currently supports
            ``k <= 100``.

    Returns:
        list[dict[str, Any]]: Top-k retrieval results with ``long_text`` populated.

    Raises:
        AssertionError: If ``k`` is greater than 100.
        requests.HTTPError: If the server responds with a non-success status.
        ValueError: If the server reports an error or omits the expected ``topk``
            field.
    """
    assert k <= 100, "Only k <= 100 is supported for the hosted ColBERTv2 server at the moment."

    payload = {"query": query, "k": k}
    res = requests.get(url, params=payload, timeout=10)
    res.raise_for_status()

    res_json = res.json()
    if res_json.get("error"):
        error_message = res_json.get("message", "Unknown error")
        raise ValueError(f"ColBERTv2 server returned an error: {error_message}")
    if "topk" not in res_json:
        raise ValueError(f"ColBERTv2 server returned an unexpected response: {res_json}")

    topk = res_json["topk"][:k]
    topk = [{**d, "long_text": d["text"]} for d in topk]
    return topk[:k]


@request_cache()
def colbertv2_get_request_v2_wrapped(*args, **kwargs):
    """Call :func:`colbertv2_get_request_v2` through the cached compatibility alias."""
    return colbertv2_get_request_v2(*args, **kwargs)


colbertv2_get_request = colbertv2_get_request_v2_wrapped


@request_cache()
def colbertv2_post_request_v2(url: str, query: str, k: int):
    """Send a POST request to a hosted ColBERTv2 server.

    Args:
        url: Retrieval endpoint URL.
        query: Query string to search for.
        k: Maximum number of results to return.

    Returns:
        list[dict[str, Any]]: Top-k retrieval results returned by the server.

    Raises:
        requests.HTTPError: If the server responds with a non-success status.
        ValueError: If the server reports an error or omits the expected ``topk``
            field.
    """
    headers = {"Content-Type": "application/json; charset=utf-8"}
    payload = {"query": query, "k": k}
    res = requests.post(url, json=payload, headers=headers, timeout=10)
    res.raise_for_status()

    res_json = res.json()
    if res_json.get("error"):
        error_message = res_json.get("message", "Unknown error")
        raise ValueError(f"ColBERTv2 server returned an error: {error_message}")
    if "topk" not in res_json:
        raise ValueError(f"ColBERTv2 server returned an unexpected response: {res_json}")

    return res_json["topk"][:k]


@request_cache()
def colbertv2_post_request_v2_wrapped(*args, **kwargs):
    """Call :func:`colbertv2_post_request_v2` through the cached compatibility alias."""
    return colbertv2_post_request_v2(*args, **kwargs)


colbertv2_post_request = colbertv2_post_request_v2_wrapped


class ColBERTv2RetrieverLocal:
    """Build or load a local ColBERT index for passage retrieval.

    This helper wraps the optional ``colbert-ai`` package to build a local index from
    in-memory passages, load an existing index, and run retrieval queries against it.
    """

    def __init__(self, passages: list[str], colbert_config=None, load_only: bool = False):
        """Initialize a local ColBERTv2 retriever.

        Args:
            passages: Corpus passages used to build or load the index.
            colbert_config: ColBERT configuration object used for indexing and search.
            load_only: Whether to skip index construction and only load an existing
                index.
        """
        assert (
            colbert_config is not None
        ), "Please pass a valid colbert_config, which you can import from colbert.infra.config import ColBERTConfig and modify it"
        self.colbert_config = colbert_config

        assert (
            self.colbert_config.checkpoint is not None
        ), "Please pass a valid checkpoint like colbert-ir/colbertv2.0, which you can modify in the ColBERTConfig with attribute name checkpoint"
        self.passages = passages

        assert (
            self.colbert_config.index_name is not None
        ), "Please pass a valid index_name, which you can modify in the ColBERTConfig with attribute name index_name"
        self.passages = passages

        if not load_only:
            print(
                f"Building the index for experiment {self.colbert_config.experiment} with index name "
                f"{self.colbert_config.index_name}"
            )
            self.build_index()

        print(
            f"Loading the index for experiment {self.colbert_config.experiment} with index name "
            f"{self.colbert_config.index_name}"
        )
        self.searcher = self.get_index()

    def build_index(self):
        """Build a local ColBERT index from ``self.passages``."""
        try:
            import colbert  # noqa: F401
        except ImportError:
            print(
                "Colbert not found. Please check your installation or install the module using pip install "
                "colbert-ai[faiss-gpu,torch]."
            )

        from colbert import Indexer
        from colbert.infra import Run, RunConfig

        with Run().context(RunConfig(nranks=self.colbert_config.nranks, experiment=self.colbert_config.experiment)):
            indexer = Indexer(checkpoint=self.colbert_config.checkpoint, config=self.colbert_config)
            indexer.index(name=self.colbert_config.index_name, collection=self.passages, overwrite=True)

    def get_index(self):
        """Load and return the configured local ColBERT search index."""
        try:
            import colbert  # noqa: F401
        except ImportError:
            print(
                "Colbert not found. Please check your installation or install the module using pip install "
                "colbert-ai[faiss-gpu,torch]."
            )

        from colbert import Searcher
        from colbert.infra import Run, RunConfig

        with Run().context(RunConfig(experiment=self.colbert_config.experiment)):
            searcher = Searcher(index=self.colbert_config.index_name, collection=self.passages)
        return searcher

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run retrieval by forwarding arguments to :meth:`forward`."""
        return self.forward(*args, **kwargs)

    def forward(self, query: str, k: int = 7, **kwargs):
        """Retrieve passages from the local ColBERT index.

        Args:
            query: Query string to search for.
            k: Maximum number of passages to return.
            **kwargs: Optional retrieval kwargs. ``filtered_pids`` can be provided to
                restrict results to a subset of passage ids.

        Returns:
            list[dotdict]: Retrieved passages with ``long_text``, ``score``, and
                ``pid`` fields.

        Raises:
            AssertionError: If ``filtered_pids`` is provided but is not a list of
                integers.
        """
        import torch

        if kwargs.get("filtered_pids"):
            filtered_pids = kwargs.get("filtered_pids")
            assert isinstance(filtered_pids, list) and all(isinstance(pid, int) for pid in filtered_pids), "The filtered pids should be a list of integers"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            searcher_results = self.searcher.search(
                query,
                # Number of passages to receive
                k=k,
                # Passing the filter function of relevant
                filter_fn=lambda pids: torch.tensor(
                    [pid for pid in pids if pid in filtered_pids], dtype=torch.int32
                ).to(device),
            )
        else:
            searcher_results = self.searcher.search(query, k=k)
        results = []
        for pid, rank, score in zip(*searcher_results, strict=False):  # noqa: B007
            results.append(dotdict({"long_text": self.searcher.collection[pid], "score": score, "pid": pid}))
        return results


class ColBERTv2RerankerLocal:
    """Score candidate passages for a query using a local ColBERT model."""

    def __init__(self, colbert_config=None, checkpoint: str = "bert-base-uncased"):
        """Initialize a local ColBERT reranker.

        Args:
            colbert_config: ColBERT configuration object used to tokenize and score
                passages.
            checkpoint: Model checkpoint used to instantiate the ColBERT model.
        """
        try:
            import colbert  # noqa: F401
        except ImportError:
            print(
                "Colbert not found. Please check your installation or install the module using pip install "
                "colbert-ai[faiss-gpu,torch]."
            )
        self.colbert_config = colbert_config
        self.checkpoint = checkpoint
        self.colbert_config.checkpoint = checkpoint

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run reranking by forwarding arguments to :meth:`forward`."""
        return self.forward(*args, **kwargs)

    def forward(self, query: str, passages: list[str] | None = None):
        """Compute ColBERT relevance scores for candidate passages.

        Args:
            query: Query string used to score passages.
            passages: Candidate passages to rerank.

        Returns:
            numpy.ndarray: One score per passage, in the same order as ``passages``.

        Raises:
            AssertionError: If ``passages`` is empty.
        """
        assert len(passages) > 0, "Passages should not be empty"

        import numpy as np
        from colbert.modeling.colbert import ColBERT
        from colbert.modeling.tokenization.doc_tokenization import DocTokenizer
        from colbert.modeling.tokenization.query_tokenization import QueryTokenizer

        passages = passages or []
        self.colbert_config.nway = len(passages)
        query_tokenizer = QueryTokenizer(self.colbert_config, verbose=1)
        doc_tokenizer = DocTokenizer(self.colbert_config)
        query_ids, query_masks = query_tokenizer.tensorize([query])
        doc_ids, doc_masks = doc_tokenizer.tensorize(passages)

        col = ColBERT(self.checkpoint, self.colbert_config)
        q = col.query(query_ids, query_masks)
        doc_ids, doc_masks = col.doc(doc_ids, doc_masks, keep_dims="return_mask")
        q_duplicated = q.repeat_interleave(len(passages), dim=0).contiguous()
        tensor_scores = col.score(q_duplicated, doc_ids, doc_masks)
        passage_score_arr = np.array([score.cpu().detach().numpy().tolist() for score in tensor_scores])
        return passage_score_arr
