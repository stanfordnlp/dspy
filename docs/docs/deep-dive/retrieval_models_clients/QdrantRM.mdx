# QdrantRM

[Qdrant](https://qdrant.tech/) is an open-source, high-performance vector search engine/database written in Rust. It can be used to retrieve semantically relevant passages to pass as context to your language model.

You can refer to the [end-to-end example](https://github.com/stanfordnlp/dspy/blob/main/examples/integrations/qdrant/qdrant_retriever_example.ipynb) demonstrating the use of DSPy and Qdrant.

## Setting Up QdrantRM

`QdrantRM` can be instantiated with any custom vectorizer and configured to return any payload field.

- `qdrant_collection_name (str)`: The name of the Qdrant collection.
- `qdrant_client (QdrantClient)`: An instance of `qdrant_client.QdrantClient`.
- `k (int, optional)`: The default number of top passages to retrieve. Default: 3.
- `document_field (str, optional)`: The key in the Qdrant payload with the content. Default: `"document"`.
- `vectorizer (BaseSentenceVectorizer, optional)`: An implementation `sentence_vectorizer.BaseSentenceVectorizer`. Default: `sentence_vectorizer.FastEmbedVectorizer`.
- `vector_name (str, optional)`: Name of the vector in the collection. Default: The first available vector name.

## Under the Hood

### `forward(self, query_or_queries: Union[str, list[str]], k: Optional[int] = None, filter: Optional[models.Filter] = None) -> dspy.Prediction`

**Parameters:**

- `query_or_queries (Union[str, List[str]])`: The query or queries to search for.
- `k (Optional[int])`: The number of top passages to retrieve. Defaults to `self.k`.
- `filter (Optional[qdrant_client.models.Filter])`: "Only include points satisfying the [filter conditions](https://qdrant.tech/documentation/concepts/filtering/)". Default: `None`.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with a `long_text` attribute.

## Example Usage

```python
import os

from qdrant_client import QdrantClient

import dspy
from dsp.modules.sentence_vectorizer import OpenAIVectorizer
from dspy.retrieve.qdrant_rm import QdrantRM

os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"

client = QdrantClient(url="http://localhost:6333/")
vectorizer = OpenAIVectorizer(model="text-embedding-3-small")

qdrant_retriever = QdrantRM(
    qdrant_client=client,
    qdrant_collection_name="{collection_name}",
    vectorizer=vectorizer,
    document_field="text",
)

dspy.settings.configure(rm=qdrant_retriever)
retrieve = dspy.Retrieve()

retrieve("Some computer programs.")
```
