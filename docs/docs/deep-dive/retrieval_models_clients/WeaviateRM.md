# Weaviate Retrieval Model
[Weaviate](https://weaviate.io/) is an open-source vector database that can be used to retrieve relevant passages before passing it to the language model. Weaviate supports a variety of [embedding models](https://weaviate.io/developers/weaviate/model-providers) from OpenAI, Cohere, Google and more! Before building your DSPy program, you will need a Weaviate cluster running with data. You can follow this [notebook](https://github.com/weaviate/recipes/blob/main/integrations/llm-frameworks/dspy/Weaviate-Import.ipynb) as an example. 


## Configuring the Weaviate Client 
Weaviate is available via a hosted service ([WCD](https://console.weaviate.cloud/)) or as a self managed instance. You can learn about the different installation methods [here](https://weaviate.io/developers/weaviate/installation#installation-methods). 

* `weaviate_collection_name` (str): The name of the Weaviate collection
* `weaviate_client` (WeaviateClient): An instance of the Weaviate client
* `k` (int, optional): The number of top passages to retrieve. The default is set to `3`

An example of the WeaviateRM constructor: 

```python
WeaviateRM(
    weaviate_collection_name: str
    weaviate_client: str,
    k: int = 5
)
```

## Using Multitenancy
Multi-tenancy allows a collection to efficiently serve isolated groups of data. Each "tenant" in a multi-tenant collection can only access its own data, while sharing the same data structure and settings.

If your Weaviate instance is tenant-aware, you can provide a tenant_id in the WeaviateRM constructor or as a keyword argument:

```python
retriever_model = WeaviateRM(
    weaviate_collection_name="<WEAVIATE_COLLECTION>",
    weaviate_client=weaviate_client,
    tenant_id="tenant123"
)

results = retriever_model("Your query here", tenant_id="tenantXYZ")
```
When tenant_id is specified, this will scope all retrieval requests to the tenant ID provided.

## Under the Hood

`forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None, **kwargs) -> dspy.Prediction`

### Parameters
* `query_or_queries` (Union[str, List[str]]): The query or queries to search for
* `k (Optional[int]): The number of top passages to retrieve. It defaults to `self.k`
*  `**kwargs`: Additional keyword arguments like `rerank` for example

### Returns
* `dspy.Prediction`: An object containing the retrieved passages


## Sending Retrieval Requests via the WeaviateRM Client

Here is an example of the Weaviate constructor using embedded:

```python
import weaviate
import dspy
from dspy.retrieve.weaviate_rm import WeaviateRM

weaviate_client = weaviate.connect_to_embedded() # you can also use local or WCD

retriever_model = WeaviateRM(
    weaviate_collection_name="<WEAVIATE_COLLECTION>",
    weaviate_client=weaviate_client 
)

results = retriever_model("Explore the significance of quantum computing", k=5)

for result in results:
    print("Document:", result.long_text, "\n")
```

You can follow along with more DSPy and Weaviate examples [here](https://weaviate.io/developers/integrations/llm-frameworks/dspy)!
