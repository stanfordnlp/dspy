import os

import dspy
from dspy.primitives.prediction import Prediction

try:
    from exa_py import Exa
except ImportError as err:
    raise ImportError(
        "The 'exa' extra is required to use Exa retrievers and tools. Install it with `pip install dspy[exa]`",
    ) from err

INTEGRATION_HEADER = "exa-dspy"


def _make_client(api_key: str | None = None, base_url: str = "https://api.exa.ai") -> Exa:
    key = api_key or os.environ.get("EXA_API_KEY")
    if not key:
        raise ValueError("Exa API key must be provided or set in EXA_API_KEY environment variable")
    client = Exa(api_key=key, base_url=base_url)
    client.headers["x-exa-integration"] = INTEGRATION_HEADER
    return client


def _format_results(results: list, max_chars: int = 1000) -> str:
    parts = []
    for r in results:
        title = getattr(r, "title", None) or ""
        url = getattr(r, "url", None) or ""
        text = getattr(r, "text", None) or ""
        if text and len(text) > max_chars:
            text = text[:max_chars] + "..."
        summary = getattr(r, "summary", None) or ""
        highlights = getattr(r, "highlights", None)

        entry = f"Title: {title}\nURL: {url}"
        if summary:
            entry += f"\nSummary: {summary}"
        if text:
            entry += f"\nContent: {text}"
        if highlights:
            entry += f"\nHighlights: {' | '.join(highlights)}"
        parts.append(entry)
    return "\n\n---\n\n".join(parts)


class ExaRM(dspy.Retrieve):
    """A retrieval module that uses the Exa API to return the top passages for a given query.

    Unlike vector DB retrievers, ExaRM performs live web search, making it suitable for
    queries that require up-to-date information.

    Args:
        exa_api_key: Exa API key. Falls back to EXA_API_KEY env var.
        k: Number of results to retrieve. Defaults to 3.
        search_type: One of "auto", "neural", or "keyword". Defaults to "auto".
        include_domains: Restrict results to these domains.
        exclude_domains: Exclude results from these domains.
        start_published_date: Filter for content published after this date (YYYY-MM-DD).
        end_published_date: Filter for content published before this date (YYYY-MM-DD).

    Examples:
        Below is a code snippet that shows how to use Exa as the default retriever:
        ```python
        import dspy
        from dspy.retrievers.exa import ExaRM

        lm = dspy.LM("openai/gpt-4o-mini")
        retriever = ExaRM(k=3)
        dspy.configure(lm=lm, rm=retriever)

        retrieve = dspy.Retrieve(k=3)
        results = retrieve("what happened in AI this week").passages
        ```
    """

    def __init__(
        self,
        exa_api_key: str | None = None,
        k: int = 3,
        search_type: str = "auto",
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        start_published_date: str | None = None,
        end_published_date: str | None = None,
    ):
        super().__init__(k=k)
        self._client = _make_client(exa_api_key)
        self._search_type = search_type
        self._include_domains = include_domains
        self._exclude_domains = exclude_domains
        self._start_published_date = start_published_date
        self._end_published_date = end_published_date

    def forward(self, query_or_queries: str | list[str], k: int | None = None, **kwargs) -> Prediction:
        """Search the web with Exa for top passages matching the query.

        Args:
            query_or_queries: The query or queries to search for.
            k: Number of results to retrieve. Defaults to self.k.

        Returns:
            dspy.Prediction with `passages` list of strings.
        """
        k = k if k is not None else self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]

        all_passages = []
        all_urls = []
        for query in queries:
            search_kwargs: dict = {
                "num_results": k,
                "type": self._search_type,
                "contents": {"text": True},
            }
            if self._include_domains:
                search_kwargs["include_domains"] = self._include_domains
            if self._exclude_domains:
                search_kwargs["exclude_domains"] = self._exclude_domains
            if self._start_published_date:
                search_kwargs["start_published_date"] = self._start_published_date
            if self._end_published_date:
                search_kwargs["end_published_date"] = self._end_published_date
            search_kwargs.update(kwargs)

            response = self._client.search(query, **search_kwargs)
            for r in response.results:
                text = getattr(r, "text", None) or ""
                all_passages.append(text)
                all_urls.append(getattr(r, "url", None) or "")

        return Prediction(passages=all_passages, urls=all_urls)


class ExaSearchTool:
    """Web search tool for DSPy ReAct agents using Exa's search API.

    Returns formatted text results that ReAct agents can reason over.

    Args:
        api_key: Exa API key. Falls back to EXA_API_KEY env var.
        num_results: Number of results per search. Defaults to 5.
        search_type: One of "auto", "neural", or "keyword". Defaults to "auto".
        include_domains: Restrict results to these domains.
        exclude_domains: Exclude results from these domains.
        start_published_date: Filter for content published after this date (YYYY-MM-DD).
        end_published_date: Filter for content published before this date (YYYY-MM-DD).
        max_chars_per_result: Truncate each result's text to this many characters.

    Examples:
        ```python
        import dspy
        from dspy.retrievers.exa import ExaSearchTool

        tool = ExaSearchTool(num_results=5)
        agent = dspy.ReAct("question -> answer", tools=[tool])
        ```
    """

    def __init__(
        self,
        api_key: str | None = None,
        num_results: int = 5,
        search_type: str = "auto",
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        start_published_date: str | None = None,
        end_published_date: str | None = None,
        max_chars_per_result: int = 1000,
    ):
        self._client = _make_client(api_key)
        self._num_results = num_results
        self._search_type = search_type
        self._include_domains = include_domains
        self._exclude_domains = exclude_domains
        self._start_published_date = start_published_date
        self._end_published_date = end_published_date
        self._max_chars_per_result = max_chars_per_result

    def __call__(self, query: str) -> str:
        """Search the web using Exa. Returns titles, URLs, and content for the most relevant pages."""
        kwargs: dict = {
            "num_results": self._num_results,
            "type": self._search_type,
            "contents": {"text": {"max_characters": self._max_chars_per_result}},
        }
        if self._include_domains:
            kwargs["include_domains"] = self._include_domains
        if self._exclude_domains:
            kwargs["exclude_domains"] = self._exclude_domains
        if self._start_published_date:
            kwargs["start_published_date"] = self._start_published_date
        if self._end_published_date:
            kwargs["end_published_date"] = self._end_published_date

        response = self._client.search(query, **kwargs)
        return _format_results(response.results, self._max_chars_per_result)


class ExaContentsTool:
    """Content fetching tool for DSPy ReAct agents using Exa's contents API.

    Fetches and extracts clean text from a URL. Useful when a ReAct agent
    has a URL and needs to read the page content.

    Args:
        api_key: Exa API key. Falls back to EXA_API_KEY env var.
        max_characters: Max characters to extract per page. Defaults to 5000.

    Examples:
        ```python
        import dspy
        from dspy.retrievers.exa import ExaSearchTool, ExaContentsTool

        agent = dspy.ReAct("question -> answer", tools=[ExaSearchTool(), ExaContentsTool()])
        ```
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_characters: int = 5000,
    ):
        self._client = _make_client(api_key)
        self._max_characters = max_characters

    def __call__(self, url: str) -> str:
        """Fetch and extract clean text content from a URL using Exa."""
        response = self._client.get_contents([url], text={"max_characters": self._max_characters})
        if not response.results:
            return f"No content found for URL: {url}"
        return _format_results(response.results, self._max_characters)


class ExaFindSimilarTool:
    """Find similar pages tool for DSPy ReAct agents using Exa's find_similar API.

    Given a URL, finds other pages with similar content.

    Args:
        api_key: Exa API key. Falls back to EXA_API_KEY env var.
        num_results: Number of similar pages to return. Defaults to 5.
        max_chars_per_result: Truncate each result's text to this many characters.

    Examples:
        ```python
        import dspy
        from dspy.retrievers.exa import ExaFindSimilarTool

        tool = ExaFindSimilarTool()
        agent = dspy.ReAct("question -> answer", tools=[tool])
        ```
    """

    def __init__(
        self,
        api_key: str | None = None,
        num_results: int = 5,
        max_chars_per_result: int = 1000,
    ):
        self._client = _make_client(api_key)
        self._num_results = num_results
        self._max_chars_per_result = max_chars_per_result

    def __call__(self, url: str) -> str:
        """Find web pages similar to a given URL using Exa."""
        response = self._client.find_similar(
            url,
            num_results=self._num_results,
            contents={"text": {"max_characters": self._max_chars_per_result}},
        )
        return _format_results(response.results, self._max_chars_per_result)
