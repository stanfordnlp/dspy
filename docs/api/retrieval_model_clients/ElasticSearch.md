
# retrieve.elastic_rm

### Constructor

Initialize an instance of the `elastic_rm` class, .

```python
elastic_rm(
    es_client: str,
    es_index: str,
    es_field: str,
    k: int = 3,
)
```

**Parameters:**
- `es_client` (_str_): The Elastic Search Client previously created and initialized (Ref. 1)
- `es_index` (_str_): Path to the directory where chromadb data is persisted.
- `es_field` (_str): The function used for embedding documents and queries. Defaults to `DefaultEmbeddingFunction()` if not specified.
- `k` (_int_, _optional_): The number of top passages to retrieve. Defaults to 3.

Ref. 1 - Connecting to Elastic Cloud -
https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/connecting.html

### Methods

#### `forward(self, query: [str], k: Optional[int] = None) -> dspy.Prediction`

Search the chromadb collection for the top `k` passages matching the given query or queries, using embeddings generated via the specified `embedding_function`.

**Parameters:**
- `query` (str_): The query.
- `k` (_Optional[int]_, _optional_): The number of results to retrieve. If not specified, defaults to the value set during initialization.

**Returns:**
- `dspy.Prediction`: Contains the retrieved passages as a list of string with the prediction signature.

ex:
```python
Prediction(
    passages=['Passage 1 Lorem Ipsum awesome', 'Passage 2 Lorem Ipsum Youppidoo', 'Passage 3 Lorem Ipsum Yassssss']
)
```

### Quick Example how to use Elastic Search in a local environment. 

Please refer to official doc if your instance is in the cloud. See (Ref. 1) above.

```python
from dspy.retrieve import elastic_rm
import os
from elasticsearch import Elasticsearch


ELASTIC_PASSWORD = os.getenv('ELASTIC_PASSWORD')

# Create the client instance
es = Elasticsearch(
    "https://localhost:9200",
    ca_certs="http_ca.crt", #Make sure you specifi the path to the certificate, generate one if you don't have.
    basic_auth=("elastic", ELASTIC_PASSWORD)
)

# Check your connection
if es.ping():
    print("Connected to Elasticsearch cluster")
else:
    print("Could not connect to Elasticsearch")

# Index name you want to search
index_name = "wiki-summary"

retriever_model = elastic_rm(
   'es_client',
   'es_index',
    es_field=embedding_function,
    k=3
)

results = retriever_model("Explore the significance of quantum computing", k=3)

for passage in results.passages:
    print("Document:", result, "\n")
```
