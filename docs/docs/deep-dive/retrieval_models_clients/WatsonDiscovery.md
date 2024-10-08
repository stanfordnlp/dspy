---
sidebar_position: 11
---

# retrieve.WatsonDiscoveryRM

### Constructor

The constructor initializes the `WatsonDiscoveryRM` class instance and sets up the request parameters for interacting with Watson Discovery service at IBM Cloud.

```python
class WatsonDiscoveryRM:
    def __init__(
        self,
        apikey: str,
        url:str,
        version:str,
        project_id:str,
        collection_ids:list=[],
        k: int = 7,
    ):
```

**Parameters:**

- `apikey` (str): apikey for authentication purposes,
- `url` (str): endpoint URL that includes the service instance ID
- `version` (str): Release date of the version of the API you want to use. Specify dates in YYYY-MM-DD format.
- `project_id` (str): The Universally Unique Identifier (UUID) of the project.
- `collection_ids` (list): An array containing the collections on which the search will be executed.
- `k` (int, optional): The number of top passages to retrieve. Defaults to 7.

### Methods

#### `forward(self, query_or_queries: Union[str, list[str]], k: Optional[int]= None) -> dspy.Prediction:`

Search the Watson Discovery collection for the top `k` passages matching the given query or queries.

**Parameters:**

- `query_or_queries` (_Union[str, list[str]]_): The query or list of queries to search for.
- `k` (_Optional[int]_, _optional_): The number of results to retrieve. If not specified, defaults to the value set during initialization.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with schema `[{"title":str, "long_text": str, "passage_score": float, "document_id": str, "collection_id": str, "start_offset": int, "end_offset": int, "field": str}]`

### Quickstart

```python
import dspy

retriever_model = WatsonDiscoveryRM(
    apikey = "Your API Key",
    url = "URL of the Watson Discovery Service",
    version = "2023-03-31",
    project_id = "Project Id",
    collection_ids = ["Collection ID"],
    k = 5
)

retrieval_response = retriever_model("Explore the significance of quantum computing",k=5)

for result in retrieval_response:
    print("Document:", result.long_text, "\n")
```
