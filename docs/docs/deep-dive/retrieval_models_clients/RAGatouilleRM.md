---
sidebar_position: 9
---

# retrieve.RAGatouilleRM

### Constructor

The constructor initializes the `RAGatouille` class instance and sets up the required parameters for interacting with the index created using [RAGatouille library](https://github.com/bclavie/RAGatouille).

```python
class RAGatouilleRM(dspy.Retrieve):
    def __init__(
        self,
        index_root: str,
        index_name: str,
        k: int = 3,
    ):
```

**Parameters:**

- `index_root` (_str_): Folder path where your index is stored.
- `index_name` (_str_): Name of the index you want to retrieve from.
- `k` (_int_): The default number of passages to retrieve. Defaults to `3`.

### Methods

#### `forward(self, query_or_queries: Union[str, List[str]], k:Optional[int]) -> dspy.Prediction`

Enables making queries to the RAGatouille-made index for retrieval. Internally, the method handles the specifics of preparing the query to obtain the response. The function handles the retrieval of the top-k passages based on the provided query.

**Parameters:**

- `query_or_queries` (Union[str, List[str]]): Query string used for retrieval.
- `k` (_int_, _optional_): Number of passages to retrieve. Defaults to 3.

**Returns:**

- `dspy.Prediction`: List of k passages
