# Retriever Modules Documentation

This documentation provides an overview of the DSPy Retrieval Model Clients.

## Supported LM Clients

| LM Client | Jump To |
| --- | --- |
| ColBERTv2 | [ColBERTv2 Section](#ColBERTv2) |
| AzureCognitiveSearch | [AzureCognitiveSearch Section](#AzureCognitiveSearch) |

## ColBERTv2

### Quickstart

```python
import dspy

colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

retrieval_response = colbertv2_wiki17_abstracts('When was the first FIFA World Cup held?', k=5)

for result in retrieval_response:
    print("Text:", result['text'], "\n")
```


### Constructor

The constructor initializes the `ColBERTv2` class instance and sets up the request parameters for interacting with the ColBERTv2 server.

```python
class ColBERTv2:
    def __init__(
        self,
        url: str = "http://0.0.0.0",
        port: Optional[Union[str, int]] = None,
        post_requests: bool = False,
    ):
```

**Parameters:**
- `url` (_str_): URL for ColBERTv2 server.
- `port` (_Union[str, int]_, _Optional_): Port endpoint for ColBERTv2 server. Defaults to `None`.
- `post_requests` (_bool_, _Optional_): Flag for using HTTP POST requests. Defaults to `False`.

### Methods

#### `__call__(self, query: str, k: int = 10, simplify: bool = False) -> Union[list[str], list[dotdict]]`

Enables making queries to the ColBERTv2 server for retrieval. Internally, the method handles the specifics of preparing the request prompt and corresponding payload to obtain the response. The function handles the retrieval of the top-k passages based on the provided query.

**Parameters:**
- `query` (_str_): Query string used for retrieval.
- `k` (_int_, _optional_): Number of passages to retrieve. Defaults to 10.
- `simplify` (_bool_, _optional_): Flag for simplifying output to a list of strings. Defaults to False.

**Returns:**
- `Union[list[str], list[dotdict]]`: Depending on `simplify` flag, either a list of strings representing the passage content (`True`) or a list of `dotdict` instances containing passage details (`False`).

## AzureCognitiveSearch

### Quickstart

#TODO

### Constructor

The constructor initializes an instance of the `AzureCognitiveSearch` class and sets up parameters for sending queries and retreiving results  with the Azure Cognitive Search server.

```python
class AzureCognitiveSearch:
    def __init__(
        self,
        search_service_name: str,
        search_api_key: str,
        search_index_name: str,
        field_text: str,
        field_score: str, # required field to map with "score" field in dsp framework
    ):
```

**Parameters:**
- `search_service_name` (_str_): Name of Azure Cognitive Search server.
- `search_api_key` (_str_): API Authentication token for accessing Azure Cognitive Search server.
- `search_index_name` (_str_): Name of search index in the Azure Cognitive Search server.
- `field_text` (_str_): Field name that maps to DSP "content" field.
- `field_score` (_str_): Field name that maps to DSP "score" field.

### Methods

Refer to [ColBERTv2](#ColBERTv2) documentation. Keep in mind there is no `simplify` flag for AzureCognitiveSearch.

AzureCognitiveSearch supports sending queries and processing the received results, mapping content and scores to a correct format for the Azure Cognitive Search server.