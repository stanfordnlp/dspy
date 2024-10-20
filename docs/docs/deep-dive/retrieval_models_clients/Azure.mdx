import AuthorDetails from '@site/src/components/AuthorDetails';

# AzureAISearch

A retrieval module that utilizes Azure AI Search to retrieve top passages for a given query.

## Prerequisites

```bash
pip install azure-search-documents
```

## Setting up the AzureAISearchRM Client

The constructor initializes an instance of the `AzureAISearchRM` class and sets up parameters for sending queries and retrieving results with the Azure AI Search server.

- `search_service_name` (str): The name of the Azure AI Search service.
- `search_api_key` (str): The API key for accessing the Azure AI Search service.
- `search_index_name` (str): The name of the search index in the Azure AI Search service.
- `field_text` (str): The name of the field containing text content in the search index. This field will be mapped to the "content" field in the dsp framework.
- `field_vector` (Optional[str]): The name of the field containing vector content in the search index.
- `k` (int, optional): The default number of top passages to retrieve. Defaults to 3.
- `azure_openai_client` (Optional[openai.AzureOpenAI]): An instance of the AzureOpenAI client. Either openai_client or embedding_func must be provided. Defaults to None.
- `openai_embed_model` (Optional[str]): The name of the OpenAI embedding model. Defaults to "text-embedding-ada-002".
- `embedding_func` (Optional[Callable]): A function for generating embeddings. Either openai_client or embedding_func must be provided. Defaults to None.
- `semantic_ranker` (bool, optional): Whether to use semantic ranking. Defaults to False.
- `filter` (str, optional): Additional filter query. Defaults to None.
- `query_language` (str, optional): The language of the query. Defaults to "en-Us".
- `query_speller` (str, optional): The speller mode. Defaults to "lexicon".
- `use_semantic_captions` (bool, optional): Whether to use semantic captions. Defaults to False.
- `query_type` (Optional[QueryType], optional): The type of query. Defaults to QueryType.FULL.
- `semantic_configuration_name` (str, optional): The name of the semantic configuration. Defaults to None.
- `is_vector_search` (Optional[bool]): Whether to enable vector search. Defaults to False.
- `is_hybrid_search` (Optional[bool]): Whether to enable hybrid search. Defaults to False.
- `is_fulltext_search` (Optional[bool]): Whether to enable fulltext search. Defaults to True.
- `vector_filter_mode` (Optional[VectorFilterMode]): The vector filter mode. Defaults to None.


**Available Query Types:**

- SIMPLE
    """Uses the simple query syntax for searches. Search text is interpreted using a simple query
    #: language that allows for symbols such as +, * and "". Queries are evaluated across all
    #: searchable fields by default, unless the searchFields parameter is specified."""
- FULL
    """Uses the full Lucene query syntax for searches. Search text is interpreted using the Lucene
    #: query language which allows field-specific and weighted searches, as well as other advanced
    #: features."""
- SEMANTIC
    """Best suited for queries expressed in natural language as opposed to keywords. Improves
    #: precision of search results by re-ranking the top search results using a ranking model trained
    #: on the Web corpus.""

    More Details: https://learn.microsoft.com/en-us/azure/search/search-query-overview

**Available Vector Filter Mode:**

- POST_FILTER = "postFilter"
    """The filter will be applied after the candidate set of vector results is returned. Depending on
    #: the filter selectivity, this can result in fewer results than requested by the parameter 'k'."""

- PRE_FILTER = "preFilter"
    """The filter will be applied before the search query."""

    More Details: https://learn.microsoft.com/en-us/azure/search/vector-search-filters

**Note**

- The `AzureAISearchRM` client allows you to perform Vector search, Hybrid search, or Full text search.
- By default, the `AzureAISearchRM` client uses the Azure OpenAI Client for generating embeddings. If you want to use something else, you can provide your custom embedding_func, but either the openai_client or embedding_func must be provided.
- If you need to enable semantic search, either with vector, hybrid, or full text search, then set the `semantic_ranker` flag to True.
- If `semantic_ranker` is True, always set the `query_type` to QueryType.SEMANTIC and always provide the `semantic_configuration_name`.

Example of the AzureAISearchRM constructor:

```python
AzureAISearchRM(
    search_service_name: str,
    search_api_key: str,
    search_index_name: str,
    field_text: str,
    field_vector: Optional[str] = None,
    k: int = 3,
    azure_openai_client: Optional[openai.AzureOpenAI] = None,
    openai_embed_model: Optional[str] = "text-embedding-ada-002",
    embedding_func: Optional[Callable] = None,
    semantic_ranker: bool = False,
    filter: str = None,
    query_language: str = "en-Us",
    query_speller: str = "lexicon",
    use_semantic_captions: bool = False,
    query_type: Optional[QueryType] = QueryType.FULL,
    semantic_configuration_name: str = None,
    is_vector_search: Optional[bool] = False,
    is_hybrid_search: Optional[bool] = False,
    is_fulltext_search: Optional[bool] = True,
    vector_filter_mode: Optional[VectorFilterMode.PRE_FILTER] = None
)
```

## Under the Hood

### `forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction`

**Parameters:**

- `query_or_queries` (Union[str, List[str]]): The query or queries to search for.
- `k` (_Optional[int]_, _optional_): The number of results to retrieve. If not specified, defaults to the value set during initialization.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with a `long_text` attribute.

Internally, the method handles the specifics of preparing the request query to the Azure AI Search service and corresponding payload to obtain the response.

The function handles the retrieval of the top-k passages based on the provided query.

## Sending Retrieval Requests via AzureAISearchRM Client

1. _**Recommended**_ Configure default RM using `dspy.configure`.

This allows you to define programs in DSPy and have DSPy internally conduct retrieval using `dsp.retrieve` on the query on the configured RM.

```python
import dspy
from dspy.retrieve.azureaisearch_rm import AzureAISearchRM

azure_search = AzureAISearchRM(
    "search_service_name",
    "search_api_key",
    "search_index_name",
    "field_text",
    "k"=3
)

dspy.settings.configure(rm=azure_search)
retrieve = dspy.Retrieve(k=3)
retrieval_response = retrieve("What is Thermodynamics").passages

for result in retrieval_response:
    print("Text:", result, "\n")
```

2. Generate responses using the client directly.

```python
from dspy.retrieve.azureaisearch_rm import AzureAISearchRM

azure_search = AzureAISearchRM(
    "search_service_name",
    "search_api_key",
    "search_index_name",
    "field_text",
    "k"=3
)

retrieval_response = azure_search("What is Thermodynamics", k=3)
for result in retrieval_response:
    print("Text:", result.long_text, "\n")
```

3. Example of Semantic Hybrid Search.

```python
from dspy.retrieve.azureaisearch_rm import AzureAISearchRM

azure_search = AzureAISearchRM(
    search_service_name="search_service_name",
    search_api_key="search_api_key",
    search_index_name="search_index_name",
    field_text="field_text",
    field_vector="field_vector",
    k=3,
    azure_openai_client="azure_openai_client",
    openai_embed_model="text-embedding-ada-002"
    semantic_ranker=True,
    query_type=QueryType.SEMANTIC,
    semantic_configuration_name="semantic_configuration_name",
    is_hybrid_search=True,
)

retrieval_response = azure_search("What is Thermodynamics", k=3)
for result in retrieval_response:
    print("Text:", result.long_text, "\n")
```

***

<AuthorDetails name="Prajapati Harishkumar Kishorkumar"/>