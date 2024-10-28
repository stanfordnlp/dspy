# LancedbRM

[LanceDB](http://lancedb.com/) is a developer-friendly, open source database for AI. From hyper scalable vector search and advanced retrieval for RAG, to streaming training data and interactive exploration of large scale AI datasets, LanceDB is the best foundation for your AI application.

## Setting Up LancedbRM

`LancedbRM` can be instantiated with any custom vectorizer and configured to return any payload field.

- `table_name (str)`: The name of the table to query against.
- `persist_directory (str)`: directory where database is stored.
- `k (int, optional)`: The number of top passages to retrieve. Defaults to 3.

## Under the Hood

### `forward(self, query_or_queries: Union[str, list[str]], k: Optional[int] = None) -> dspy.Prediction`

**Parameters:**

- `query_or_queries (Union[str, List[str]])`: The query or queries to search for.
- `k (Optional[int])`: The number of top passages to retrieve. Defaults to `self.k`.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with a `long_text` attribute.

## Example Usage

```python
import os
import pandas as pd

import dspy
from dspy.retrieve.lancedb import LancedbRM

from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

from datasets import load_dataset

# load_dataset
ds = load_dataset("fancyzhx/dbpedia_14")
df = pd.DataFrame(ds['train'])

os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"

uri = 'tmp/db'
table_name = 'passages'
db = lancedb.connect(uri)
model = get_registry().get("sentence-transformers").create(name="BAAI/bge-small-en-v1.5", device="cpu")

class Passages(LanceModel):
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()

table = db.create_table(table_name, schema=Passages)
table.add(df['content'])


lancedb_retriever = LancedbRM(
    table_name=table_name,
    persist_directory=uri,
)

dspy.settings.configure(rm=lancedb_retriever)
retrieve = dspy.Retrieve()

retrieve("Integrated Circuits")
```
