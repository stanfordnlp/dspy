# ClarifaiRM

[Clarifai](https://clarifai.com/) is a powerful AI platform that provides vector search capabilities through its search API. DSPy has integrated ClarifaiRM to support efficient text search and retrieval through its specialized indexing and ability to handle large-scale document collections.

To support passage retrieval, ClarifaiRM assumes that documents have been properly ingested into a Clarifai application with the following:
- Text data properly indexed and stored
- Appropriate search configurations set up in the Clarifai platform
- Valid authentication credentials (PAT key) with appropriate permissions

The ClarifaiRM module requires the `clarifai` Python package. If not already installed, you can install it using:

```bash
pip install clarifai
```

**Note:** 

Before using ClarifaiRM, ensure you have:

1. Created a Clarifai account and application
2. Ingested your documents into the application
3. Obtained your User ID, App ID, and Personal Access Token (PAT)

## Setting up the ClarifaiRM Client

The constructor initializes an instance of the `ClarifaiRM` class, which requires authentication credentials and configuration to connect to your Clarifai application.

- `clarifai_user_id` (_str_): Your unique Clarifai user identifier.
- `clarifai_app_id` (_str_): The ID of your Clarifai application where documents are stored.
- `clarifai_pat` (_Optional[str]_): Your Clarifai Personal Access Token (PAT). It will look for `CLARIFAI_PAT` in environment variables if not provided.
- `k` (_int_, _optional_): The number of top passages to retrieve. Defaults to 3.

Example of the ClarifaiRM constructor:

```python
ClarifaiRM(
    clarifai_user_id: str,
    clarifai_app_id: str,
    clarifai_pat: Optional[str] = None,
    k: int = 3,
)
```

**Note:** 

The PAT can be provided either directly to the constructor or through the `CLARIFAI_PAT` environment variable. For security best practices, using environment variables is recommended.

## Under the Hood

### `retrieve_hits(self, hits)`

**Parameters:**
- `hits` (_ClarifaiHit_): A hit object from Clarifai's search response.

**Returns:**
- `str`: The retrieved text content.

Internal method that retrieves text content from the hit's URL using authenticated requests.

### `forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None, **kwargs) -> dspy.Prediction`

**Parameters:**
- `query_or_queries` (_Union[str, List[str]]_): The query or list of queries to search for.
- `k` (_Optional[int]_, _optional_): The number of results to retrieve. If not specified, defaults to the value set during initialization.
- `**kwargs`: Additional keyword arguments passed to Clarifai's search function.

**Returns:**
- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with a `long_text` attribute.

Search the Clarifai application for the top `k` passages matching the given query or queries. Uses parallel processing with ThreadPoolExecutor to efficiently retrieve multiple results.

## Examples

### Basic Usage
```python
import os
from dspy.retrieve.clarifai_rm import ClarifaiRM
import dspy

os.environ["CLARIFAI_PAT"] = "your_pat_key"

retriever_model = ClarifaiRM(
    clarifai_user_id="your_user_id",
    clarifai_app_id="your_app_id",
    k=5
)

turbo = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=turbo, rm=retriever_model)

results = retriever_model("Explore the significance of quantum computing")
```

### Multiple Queries
```python
queries = [
    "What is machine learning?",
    "How does deep learning work?",
    "Explain neural networks"
]

results = retriever_model(queries, k=3)
```

### Using with DSPy Retrieve Module
```python
from dspy import Retrieve

retrieve = Retrieve(k=5)

class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = Retrieve(k=3)
    
    def forward(self, query):
        passages = self.retrieve(query)
        return passages

rag = RAG()
result = rag("What are the latest developments in AI?")
```

### Handling Results
```python
results = retriever_model("quantum computing advances", k=5)

for i, result in enumerate(results, 1):
    print(f"Result {i}:")
    print(result.long_text)
    print("-" * 50)

first_passage = results[0].long_text

num_results = len(results)
```

### Integration with Other DSPy Components
```python
from dspy import ChainOfThought, Predict, Retrieve 

# Create a simple QA chain
class QAChain(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = Retrieve(k=3)
        self.generate_answer = ChainOfThought("question, context -> answer")
    
    def forward(self, question):
        context = self.retrieve(question)
        answer = self.generate_answer(question=question, context=context)
        return answer

qa = QAChain()
answer = qa("What are the main applications of quantum computing?")
```

### Error Handling Example
```python
try:
    retriever_model = ClarifaiRM(
        clarifai_user_id="your_user_id",
        clarifai_app_id="your_app_id",
        clarifai_pat="invalid_pat"
    )
    results = retriever_model("test query")
except Exception as e:
    print(f"Error occurred: {e}")
```

**Note:** 

These examples assume you have:

- A properly configured Clarifai application
- Valid authentication credentials
- Documents already ingested into your Clarifai app
- The necessary environment variables set up