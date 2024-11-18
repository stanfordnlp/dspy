# FalkordbRM

### Constructor

Initialize an instance of the `FalkordbRM` class.

```python
FalkordbRM(
    node_label: str,
    text_node_property: str,
    embedding_node_property: str,
    k: int = 5,
    retrieval_query: str,
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-ada-002",
)
```

**Environment Variables:**

You need to define the credentials as environment variables

- `FALKORDB_HOST` (_str_): Specifies the host required for connecting with the Falkordb database. If not provided, the system will default to `localhost`

- `FALKORDB_PORT` (_int_): Specifies the port required for connecting with the Falkordb database. If not provided, the system will default to `6379`

- `FALKORDB_USERNAME` (_str_, optional): Specifies the username required for authenticating with a [Falkordb Cloud](https://app.falkordb.cloud/signin) database.

- `FALKORDB_PASSWORD` (_str_, optional): Specifies the password required for authenticating with a [Falkordb Cloud](https://app.falkordb.cloud/signin) database.

- `FALKORDB_DATABASE` (_str_, optional): Specifies the name of the database to connect to within the Falkordb instance. If not provided, the systems defaults to using a randomly generated four ascii_letters character string e.g "tari".

- `OPENAI_API_KEY` (_str_): Specifies the API key required for authenticating with OpenAI's services.

**Parameters:**

- `node_label` (_str_): Specifies the label of the node to be used within Falkordb for organizing and querying data.
- `text_node_property` (_str_, _optional_): Defines the specific text property of the node that will be returned.
- `embedding_node_property` (_str_): Defines the specific embedding property of the node that will be used within Falkordb for querying data.
- `k` (_int_, _optional_): The number of top results to return from the retrieval operation. It defaults to 5 if not explicitly specified.
- `retrieval_query` (_str_, _optional_): A custom query string provided for retrieving data. If not provided, a default query tailored to the `text_node_property` will be used.
- `embedding_provider` (_str_, _optional_): The name of the service provider for generating embeddings. Only "openai" is supported.
- `embedding_model` (_str_, _optional_): The specific embedding model to use from the provider. By default, it uses the "text-embedding-ada-002" model from OpenAI.


### Methods

#### `forward(self, query: [str], k: Optional[int] = None) -> dspy.Prediction`

Search the Falkordb vector index for the top `k` passages matching the given query or queries, using embeddings generated via the specified `embedding_model`.

**Parameters:**

- `query` (str\_): The query.
- `k` (_Optional[int]_, _optional_): The number of results to retrieve. If not specified, defaults to the value set during initialization.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages as a list of string with the prediction signature.

ex:

```python
Prediction(
    passages=['Passage 1 Lorem Ipsum awesom', 'Passage 2 Lorem Ipsum Youppidoo', 'Passage 3 Lorem Ipsum Yassssss']
)
```

### Quick Example how to use Falkordb in a local environment.

```python
from dspy.retrieve.falkordb_rm import FalkordbRM
import os


os.environ["FALKORDB_HOST"] = 'localhost'
os.environ["FALKORDB_PORT"] = 6379
os.environ["OPENAI_API_KEY"] = 'sk-'

retriever_model = FalkordbRM(
    node_label="myIndex",
    text_node_property="text",
    embedding_node_property="embedding"
)

results = retriever_model("Explore the significance of quantum computing", k=3)

for passage in results:
    print("Document:", passage, "\n")
```