---
sidebar_position: 12
---

# retrieve.YouRM

### Constructor

Initialize an instance of the `YouRM` class that calls the [You.com APIs](https://documentation.you.com/api-reference/search) for web-based document retrieval. Options available are the "Search" and "News" APIs.

```python
YouRM(
    ydc_api_key: Optional[str] = None,
    k: int = 3,
    endpoint: Literal["search", "news"] = "search",
    num_web_results: Optional[int] = None,
    safesearch: Optional[Literal["off", "moderate", "strict"]] = None,
    country: Optional[str] = None,
    search_lang: Optional[str] = None,
    ui_lang: Optional[str] = None,
    spellcheck: Optional[bool] = None,
)
```

**Parameters:**

- `ydc_api_key` (Optional[str]): you.com API key, if `YDC_API_KEY` is not set in the environment
- `k` (int): If `endpoint="search"`, the max snippets to return per search hit.
  If `endpoint="news"`, the max articles to return.
- `endpoint` (Literal["search", "news"]): you.com endpoints
- `num_web_results` (Optional[int]): The max number of web results to return, must be under 20
- `safesearch` (Optional[Literal["off", "moderate", "strict"]]): Safesearch settings, one of "off", "moderate", "strict", defaults to moderate
- `country` (Optional[str]): Country code, ex: 'US' for United States, see API reference for more info
- `search_lang` (Optional[str]): (News API) Language codes, ex: 'en' for English, see API reference for more info
- `ui_lang` (Optional[str]): (News API) User interface language for the response, ex: 'en' for English.
  See API reference for more info
- `spellcheck` (Optional[bool]): (News API) Whether to spell check query or not, defaults to True

### Methods

#### `forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction`

If `endpoint="search"`, search the web for the top `k` snippets matching the given query or queries.

If `endpoint="news"`, search the web for the top `k` articles matching the given query or queries.

**Parameters:**

- `query_or_queries` (_Union[str, List[str]]_): The query or list of queries to search for.
- `k` (_Optional[int]_, _optional_): The number of results to retrieve. If not specified, defaults to the value set during initialization.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with schema `[{"long_text": str}]`

### Quickstart

Obtain a You.com API key from [https://api.you.com/](https://api.you.com/).

Export this key to an environment variable `YDC_API_KEY`.

```python
from dspy.retrieve.you_rm import YouRM
import os

# The retriever obtains the API key from the `YDC_API_KEY` env var
retriever_model = YouRM(endpoint="search")

results = retriever_model("Tell me about national parks in the US", k=5)

for result in results:
    print("Document:", result.long_text, "\n")
```
