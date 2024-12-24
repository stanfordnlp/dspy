# Needle Retrieval Model

[Needle](https://needle-ai.com/) is a managed RAG API that abstracts away tedious details in your RAG pipeline, such as chunking, vectorization, indexing, search etc. and directly expose the search results to you. Needle uses a combination of different techniques such as vector search, text search, contextualized chunks, reranking, and more to get the most relevant results and is available with a forever free tier.

Tired of searching for the best retrieval configuration for your RAG pipeline? We were there too, that's why we built Needle for you. For more details, check out the [Needle Documentation](https://docs.needle-ai.com/). If you have any questions, please reach out to us at [support@needle-ai.com](mailto:support@needle-ai.com) or at our [Discord server](https://discord.com/invite/XSHaP5pPHT).

## Prerequisites

To use the `NeedleRM`, you must to install dspy extra package via `pip install dspy[needle]`.

## Getting Started

You can construct the `NeedleRM` by providing the `collection_id`.

- `collection_id`: The ID of the collection to search. Find it in the [Needle Dashboard](https://needle-ai.com/dashboard/collections).
- `api_key`: (Optional) Used in authenticating to Needle API. If not provided, it's read from `NEEDLE_API_KEY` environment variable.
- `k`: (Optional) The number of top passages to retrieve. Defaults to `10`.

```python
rm = NeedleRM(
    collection_id: str,
    api_key: Optional[str] = None,
    k: Optional[int] = 10
)
```

Retrieve the top `k` passages for a given query.

```python
passages = rm("What is Needle?")
```

## Under the Hood

```python
forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None, **kwargs) -> list[dict[str, str]]
```

Internally, the method handles the specifics of preparing the request query to the Needle API and corresponding payload to obtain the response.
The function handles the retrieval of the top-k passages based on the provided query.

**Parameters:**

- `query_or_queries` (Union[str, List[str]]): The query or queries to search for.
- `k` (_Optional[int]_, _optional_): The number of results to retrieve. If not specified, defaults to the value set during initialization.

**Returns:**

- `list[dict[str, str]]`: Contains the retrieved passages, each represented as a `dotdict` with a `long_text` attribute.


## Usage Method 1: Configure as default retriever

1. _**Recommended**_ Configure default RM using `dspy.configure`.

This allows you to define programs in DSPy and have DSPy internally conduct retrieval using `dsp.retrieve` on the query on the configured RM.

```python
import dspy
from dspy.retrieve.needle_rm import NeedleRM

openai_lm = dspy.LM('openai/gpt-4o-mini', api_key='openai_api_key')
needle_rm = NeedleRM(collection_id="collection_id", api_key="needle_api_key")

dspy.settings.configure(
    lm=openai_lm, 
    rm=needle_rm
)

rm = dspy.Retrieve()
rm("What is Needle?")
```

## Usage Method 2: Direct invocation

```python
from dspy.retrieve.needle_rm import NeedleRM

rm = NeedleRM(collection_id="collection_id", api_key="api_key")

needle_rm("What is PQL?")
```

## Example: RAG

```python
dspy.configure(lm=openai_lm)

rag = dspy.ChainOfThought('context, question -> response')

question = "What is Needle?"
context = needle_rm(question)
rag(question=question, context=context)
```

Outputs

```
Prediction(
    reasoning='Needle is a powerful tool designed to bring AI intelligence to your data. It allows users to add files either through a web interface or an API, which are then indexed for semantic search. This enables users to retrieve the most relevant text chunks from their files, facilitating the construction of human-readable answers or integration into applications. The documentation provides guides, tutorials, and API references to help users get started with building retrieval-augmented generation (RAG) applications.',
    response='Needle is a tool that enables users to programmatically interact with the Needle platform, allowing for the creation, updating, deletion, and searching of collections and files. It is designed to facilitate semantic search on indexed data, returning relevant text chunks that can be used to generate human-readable answers or integrated into applications. The platform supports both a web UI and an API for file management and search functionalities.'
)
```
