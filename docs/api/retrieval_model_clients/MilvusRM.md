---
sidebar_position: 6
---

# retrieve.MilvusRM

### Constructor

Initialize an instance of the `MilvusRM` class, with the option to use OpenAI's `text-embedding-3-small` embeddings or any customized embedding function.

```python
MilvusRM(
    collection_name: str,
    uri: Optional[str] = "http://localhost:19530",
    token: Optional[str] = None,
    db_name: Optional[str] = "default",
    embedding_function: Optional[Callable] = None,
    k: int = 3,
)
```

**Parameters:**

- `collection_name (str)`: The name of the Milvus collection to query against.
- `uri (str, optional)`: The Milvus connection uri. Defaults to "http://localhost:19530".
- `token (str, optional)`: The Milvus connection token. Defaults to None.
- `db_name (str, optional)`: The Milvus database name. Defaults to "default".
- `embedding_function (callable, optional)`: The function to convert a list of text to embeddings.
  The embedding function should take a list of text strings as input and output a list of embeddings.
  Defaults to None. By default, it will get OpenAI client by the environment variable OPENAI_API_KEY and use OpenAI's embedding model "text-embedding-3-small" with the default dimension.
- `k (int, optional)`: The number of top passages to retrieve. Defaults to 3.

### Methods

#### `forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction`

Search the Milvus collection for the top `k` passages matching the given query or queries, using embeddings generated via the default OpenAI embedding or the specified `embedding_function`.

**Parameters:**

- `query_or_queries` (_Union[str, List[str]]_): The query or list of queries to search for.
- `k` (_Optional[int]_, _optional_): The number of results to retrieve. If not specified, defaults to the value set during initialization.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with schema `[{"id": str, "score": float, "long_text": str, "metadatas": dict }]`

### Quickstart

To support passage retrieval, it assumes that a Milvus collection has been created and populated with the following field:

- `text`: The text of the passage

MilvusRM uses OpenAI's `text-embedding-3-small` embedding by default or any customized embedding function.
While different options are available, the examples below demonstrate how to utilize the default OpenAI embeddings and a customized embedding function using the BGE model.

#### Default OpenAI Embeddings

```python
from dspy.retrieve.milvus_rm import MilvusRM
import os

os.envrion["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"

retriever_model = MilvusRM(
    collection_name="<YOUR_COLLECTION_NAME>",
    uri="<YOUR_MILVUS_URI>",
    token="<YOUR_MILVUS_TOKEN>"  # ignore this if no token is required for Milvus connection
    )

results = retriever_model("Explore the significance of quantum computing", k=5)

for result in results:
    print("Document:", result.long_text, "\n")
```

#### Customized Embedding Function

```python
from dspy.retrieve.milvus_rm import MilvusRM
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-base-en-v1.5')

def bge_embedding_function(texts: List[str]):
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings

retriever_model = MilvusRM(
    collection_name="<YOUR_COLLECTION_NAME>",
    uri="<YOUR_MILVUS_URI>",
    token="<YOUR_MILVUS_TOKEN>",  # ignore this if no token is required for Milvus connection
    embedding_function=bge_embedding_function
    )

results = retriever_model("Explore the significance of quantum computing", k=5)

for result in results:
    print("Document:", result.long_text, "\n")
```
