---
sidebar_position: 1
---

# dspy.ColBERTv2

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

### Quickstart

```python
import dspy

colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

retrieval_response = colbertv2_wiki17_abstracts('When was the first FIFA World Cup held?', k=5)

for result in retrieval_response:
    print("Text:", result['text'], "\n")
```

# dspy.ColBERTv2RetrieverLocal

This is taken from the official documentation of [Colbertv2](https://github.com/stanford-futuredata/ColBERT/tree/main) following the [paper](https://arxiv.org/abs/2112.01488).

You can install Colbertv2 by the following instructions from [here](https://github.com/stanford-futuredata/ColBERT?tab=readme-ov-file#installation)

### Constructor
The constructor initializes the ColBERTv2 as a local retriever object. You can initialize a server instance from your ColBERTv2 local instance using the code snippet from [here](https://github.com/stanford-futuredata/ColBERT/blob/main/server.py)

```python
class ColBERTv2RetrieverLocal:
    def __init__(
        self,
        passages:List[str],
        colbert_config=None,
        load_only:bool=False):
```

**Parameters**
- `passages` (_List[str]_): List of passages to be indexed
- `colbert_config` (_ColBERTConfig_, _Optional_): colbert config for building and searching. Defaults to None.
- `load_only` (_Boolean_): whether to load the index or build and then load. Defaults to False.

The `colbert_config` object is required for ColBERTv2, and it can be imported from `from colbert.infra.config import ColBERTConfig`. You can find the descriptions of config attributes from [here](https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/infra/config/settings.py)

### Methods

#### `forward(self, query:str, k:int, **kwargs) -> Union[list[str], list[dotdict]]`

It retrieves relevant passages from the index based on the query. If you already have a local index, then you can pass the `load_only` flag as `True` and change the `index` attribute of ColBERTConfig to the local path. Also, make sure to change the `checkpoint` attribute of ColBERTConfig to the embedding model that you used to build the index.

**Parameters:**
- `query` (_str_): Query string used for retrieval.
- `k` (_int_, _optional_): Number of passages to retrieve. Defaults to 7

It returns a `Prediction` object for each query

```python
Prediction(
    pid=[33, 6, 47, 74, 48],
    passages=['No pain, no gain.', 'The best things in life are free.', 'Out of sight, out of mind.', 'To be or not to be, that is the question.', 'Patience is a virtue.']
)
```
# dspy.ColBERTv2RerankerLocal

You can also use ColBERTv2 as a reranker in DSPy.

### Constructor

```python
class ColBERTv2RerankerLocal:
    
    def __init__(
        self,
        colbert_config=None,
        checkpoint:str='bert-base-uncased'):
```

**Parameters**
- `colbert_config` (_ColBERTConfig_, _Optional_): colbert config for building and searching. Defaults to None.
- `checkpoint` (_str_): Embedding model for embeddings the documents and query

### Methods
#### `forward(self,query:str,passages:List[str])`

Based on a query and list of passages, it reranks the passages and returns the scores along with the passages ordered in descending order based on the similarity scores.

**Parameters:**
- `query` (_str_): Query string used for reranking.
- `passages` (_List[str]_): List of passages to be reranked

It returns the similarity scores array and you can link it to the passages by

```python
for idx in np.argsort(scores_arr)[::-1]:
    print(f"Passage = {passages[idx]} --> Score = {scores_arr[idx]}")
```