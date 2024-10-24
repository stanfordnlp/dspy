---
sidebar_position: 8
---

# retrieve.neo4j_rm

### Constructor

Initialize an instance of the `Neo4jRM` class.

```python
Neo4jRM(
    index_name: str,
    text_node_property: str,
    k: int = 5,
    retrieval_query: str = None,
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-ada-002",
)
```

**Environment Variables:**

You need to define the credentials as environment variables:

- `NEO4J_USERNAME` (_str_): Specifies the username required for authenticating with the Neo4j database. This is a crucial security measure to ensure that only authorized users can access the database.

- `NEO4J_PASSWORD` (_str_): Defines the password associated with the `NEO4J_USERNAME` for authentication purposes. This password should be kept secure to prevent unauthorized access to the database.

- `NEO4J_URI` (_str_): Indicates the Uniform Resource Identifier (URI) used to connect to the Neo4j database. This URI typically includes the protocol, hostname, and port, providing the necessary information to establish a connection to the database.

- `NEO4J_DATABASE` (_str_, optional): Specifies the name of the database to connect to within the Neo4j instance. If not set, the system defaults to using `"neo4j"` as the database name. This allows for flexibility in connecting to different databases within a single Neo4j server.

- `OPENAI_API_KEY` (_str_): Specifies the API key required for authenticiating with OpenAI's services.

**Parameters:**

- `index_name` (_str_): Specifies the name of the vector index to be used within Neo4j for organizing and querying data.
- `text_node_property` (_str_, _optional_): Defines the specific property of nodes that will be returned.
- `k` (_int_, _optional_): The number of top results to return from the retrieval operation. It defaults to 5 if not explicitly specified.
- `retrieval_query` (_str_, _optional_): A custom query string provided for retrieving data. If not provided, a default query tailored to the `text_node_property` will be used.
- `embedding_provider` (_str_, _optional_): The name of the service provider for generating embeddings. Defaults to "openai" if not specified.
- `embedding_model` (_str_, _optional_): The specific embedding model to use from the provider. By default, it uses the "text-embedding-ada-002" model from OpenAI.

### Methods

#### `forward(self, query: [str], k: Optional[int] = None) -> dspy.Prediction`

Search the neo4j vector index for the top `k` passages matching the given query or queries, using embeddings generated via the specified `embedding_model`.

**Parameters:**

- `query` (str\_): The query.
- `k` (_Optional[int]_, _optional_): The number of results to retrieve. If not specified, defaults to the value set during initialization.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages as a list of string with the prediction signature.

ex:

```python
Prediction(
    passages=['Passage 1 Lorem Ipsum awesome', 'Passage 2 Lorem Ipsum Youppidoo', 'Passage 3 Lorem Ipsum Yassssss']
)
```

### Quick Example how to use Neo4j in a local environment.

```python
from dspy.retrieve.neo4j_rm import Neo4jRM
import os

os.environ["NEO4J_URI"] = 'bolt://localhost:7687'
os.environ["NEO4J_USERNAME"] = 'neo4j'
os.environ["NEO4J_PASSWORD"] = 'password'
os.environ["OPENAI_API_KEY"] = 'sk-'

retriever_model = Neo4jRM(
    index_name="vector",
    text_node_property="text"
)

results = retriever_model("Explore the significance of quantum computing", k=3)

for passage in results:
    print("Document:", passage, "\n")
```
