---
sidebar_position: 5
---

# retrieve.FaissRM

### Constructor

Initialize an instance of FaissRM by providing it with a vectorizer and a list of strings

```python
FaissRM(
    document_chunks: List[str],
    vectorizer: dsp.modules.sentence_vectorizer.BaseSentenceVectorizer,
    k: int = 3
)
```

**Parameters:**

- `document_chunks` (_List[str]_): a list of strings that comprises the corpus to search. You cannot add/insert/upsert to this list after creating this FaissRM object.
- `vectorizer` (_dsp.modules.sentence_vectorizer.BaseSentenceVectorizer_, _optional_): If not provided, a dsp.modules.sentence_vectorizer.SentenceTransformersVectorizer object is created and used.
- `k` (_int_, _optional_): The number of top passages to retrieve. Defaults to 3.

### Methods

#### `forward(self, query_or_queries: Union[str, List[str]]) -> dspy.Prediction`

Search the FaissRM vector database for the top `k` passages matching the given query or queries, using embeddings generated via the vectorizer specified at FaissRM construction time

**Parameters:**

- `query_or_queries` (_Union[str, List[str]]_): The query or list of queries to search for.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with a `long_text` attribute and an `index` attribute. The `index` attribute is the index in the document_chunks array provided to this FaissRM object at construction time.

### Quickstart with the default vectorizer

The **FaissRM** module provides a retriever that uses an in-memory Faiss vector database. This module does not include a vectorizer; instead it supports any subclass of **dsp.modules.sentence_vectorizer.BaseSentenceVectorizer**. If a vectorizer is not provided, an instance of **dsp.modules.sentence_vectorizer.SentenceTransformersVectorizer** is created and used by **FaissRM**. Note that the default embedding model for **SentenceTransformersVectorizer** is **all-MiniLM-L6-v2**

```python
import dspy
from dspy.retrieve.faiss_rm import FaissRM

document_chunks = [
    "The superbowl this year was played between the San Francisco 49ers and the Kanasas City Chiefs",
    "Pop corn is often served in a bowl",
    "The Rice Bowl is a Chinese Restaurant located in the city of Tucson, Arizona",
    "Mars is the fourth planet in the Solar System",
    "An aquarium is a place where children can learn about marine life",
    "The capital of the United States is Washington, D.C",
    "Rock and Roll musicians are honored by being inducted in the Rock and Roll Hall of Fame",
    "Music albums were published on Long Play Records in the 70s and 80s",
    "Sichuan cuisine is a spicy cuisine from central China",
    "The interest rates for mortgages is considered to be very high in 2024",
]

frm = FaissRM(document_chunks)
turbo = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=turbo, rm=frm)
print(frm(["I am in the mood for Chinese food"]))
```
