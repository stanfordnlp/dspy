# VectaraRM

[Vectara](https://vectara.com/) is a modern vector database and search platform that provides advanced retrieval capabilities through its search API. DSPy has integrated VectaraRM to enable efficient semantic search and retrieval with features like contextual snippets, multi-corpus search, and lexical interpolation.

To support passage retrieval, VectaraRM assumes that documents have been properly ingested into a Vectara corpus with the following:

- Text data properly indexed and stored
- Appropriate corpus configurations set up in the Vectara platform
- Valid authentication credentials with appropriate permissions
- Proper sentence segmentation for contextual retrieval

The VectaraRM module requires the following:

- Python's built-in `json` and `requests` packages for API communication
- A Vectara account with API access
- At least one configured corpus with indexed documents

```bash
pip install requests  # if not already installed
```

**Note:**

Before using VectaraRM, ensure you have:

1. Created a Vectara account
2. Created and configured at least one corpus
3. Obtained your:
   - Customer ID (vectara_customer_id)
   - Corpus ID (vectara_corpus_id)
   - API Key (vectara_api_key)
4. Indexed your documents in the Vectara corpus

## Setting up the VectaraRM Client

The constructor initializes an instance of the `VectaraRM` class, which requires authentication credentials and configuration to connect to your Vectara corpus.

- `vectara_customer_id` (_Optional[str]_): Your unique Vectara customer identifier. If not provided, defaults to `VECTARA_CUSTOMER_ID` environment variable.
- `vectara_corpus_id` (_Optional[str]_): The ID of your Vectara corpus where documents are stored. Can be a single ID or comma-separated list for multiple corpora. If not provided, defaults to `VECTARA_CORPUS_ID` environment variable.
- `vectara_api_key` (_Optional[str]_): Your Vectara API key. If not provided, defaults to `VECTARA_API_KEY` environment variable.
- `k` (_int_, _optional_): The number of top passages to retrieve. Defaults to 5.

Example of the VectaraRM constructor:

```python
VectaraRM(
    vectara_customer_id: Optional[str] = None,
    vectara_corpus_id: Optional[str] = None,
    vectara_api_key: Optional[str] = None,
    k: int = 5,
)
```

### Environment Variable Setup

You can set up authentication using environment variables:

```bash
export VECTARA_CUSTOMER_ID="your_customer_id"
export VECTARA_CORPUS_ID="your_corpus_id"
export VECTARA_API_KEY="your_api_key"
```

### Example Initialization

```python

retriever = VectaraRM(k=5)


retriever = VectaraRM(
    vectara_customer_id="your_customer_id",
    vectara_corpus_id="your_corpus_id",
    vectara_api_key="your_api_key",
    k=5
)


retriever = VectaraRM(
    vectara_corpus_id="corpus_id_1,corpus_id_2",
    k=5
)
```

**Note:**

- For security best practices, using environment variables is recommended
- The client configures default parameters for context retrieval (2 sentences before and after the matched text)
- API timeout is set to 120 seconds by default


## Under the Hood

### `_vectara_query(self, query: str, limit: int) -> List[str]`

**Parameters:**

- `query` (_str_): The search query string.
- `limit` (_int_): Maximum number of results to retrieve.

**Returns:**

- `List[str]`: List of dictionaries containing retrieved passages and their scores. Each dictionary has:
  - `text`: The retrieved passage text
  - `score`: The relevance score for the passage

Internal method that queries the Vectara API to retrieve relevant passages. This method:

- Supports single or multiple corpus search
- Configures contextual snippet retrieval
- Applies lexical interpolation for improved search quality
- Handles all API communication with Vectara

The method includes these key configurations:

- Contextual retrieval of 2 sentences before and after the matched text
- Lexical interpolation lambda of 0.025 for balancing semantic and keyword matching
- Custom start and end tags for highlighting matched text
- Request timeout of 120 seconds

Example of the internal API request structure:
```python
{
    "query": [{
        "query": query,
        "start": 0,
        "numResults": limit,
        "contextConfig": {
            "sentencesBefore": 2,
            "sentencesAfter": 2,
            "startTag": "<%START%>",
            "endTag": "<%END%>"
        },
        "corpusKey": [
            {
                "customerId": customer_id,
                "corpusId": corpus_id,
                "lexicalInterpolationConfig": {"lambda": 0.025}
            }
        ]
    }]
}
```

### `forward(self, query_or_queries: Union[str, List[str]], k: Optional[int]) -> dspy.Prediction`

**Parameters:**

- `query_or_queries` (_Union[str, List[str]]_): The query or list of queries to search for.
- `k` (_Optional[int]_, _optional_): The number of results to retrieve. If not specified, defaults to the value set during initialization.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with a `long_text` attribute.

Search the Vectara corpus for the top `k` passages matching the given query or queries. The method includes several key features:

- Handles both single queries and multiple queries efficiently
- For multiple queries, increases the internal limit to 3*k for better coverage
- Aggregates scores across multiple queries for the same passage
- Returns the top k passages sorted by combined relevance scores

The method processes queries through these steps:

1. Normalizes input to handle both single and multiple queries
2. Filters out any empty queries
3. For each query:
   - Calls `_vectara_query` with appropriate limit
   - Aggregates results and scores
4. Sorts passages by combined scores
5. Returns top k passages in DSPy's expected format


## Examples

### Basic Usage
```python
from dspy.retrieve.vectara_rm import VectaraRM
import dspy
import os


os.environ["VECTARA_CUSTOMER_ID"] = "your_customer_id"
os.environ["VECTARA_CORPUS_ID"] = "your_corpus_id"
os.environ["VECTARA_API_KEY"] = "your_api_key"


retriever_model = VectaraRM(k=5)


turbo = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=turbo, rm=retriever_model)


results = retriever_model("What are the advantages of quantum computing?")


for result in results:
    print(f"Retrieved text: {result.long_text}\n")
```

### Multiple Queries Support
```python

queries = [
    "quantum computing applications",
    "quantum computing advantages",
    "quantum computing use cases"
]


results = retriever_model(queries, k=5)


for i, result in enumerate(results, 1):
    print(f"Top Result {i}:")
    print(result.long_text)
    print("-" * 50)
```

### Multiple Corpora Usage
```python

retriever = VectaraRM(
    vectara_corpus_id="corpus_id_1,corpus_id_2",
    vectara_api_key="your_api_key",
    vectara_customer_id="your_customer_id",
    k=5
)

results = retriever("quantum computing research")
```

### Integration with DSPy Modules
```python
from dspy import Retrieve, ChainOfThought

class AdvancedRAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = VectaraRM(k=num_passages)
        self.generate_answer = ChainOfThought("question, context -> answer")
    
    def forward(self, query):
        
        context = self.retrieve(query)
        
        
        answer = self.generate_answer(
            question=query,
            context=context
        )
        return answer


rag = AdvancedRAG(num_passages=5)
response = rag("Explain the impact of quantum computing on cryptography")
```

### Error Handling and Validation
```python
try:
    retriever = VectaraRM(
        vectara_customer_id="invalid_id",
        vectara_api_key="invalid_key",
        k=5
    )
    
    results = retriever("test query")
except Exception as e:
    print(f"Error occurred: {e}")
    


def validate_results(results):
    if not results:
        print("No results found")
        return False
    
    print(f"Retrieved {len(results)} passages")
    return True

results = retriever("quantum computing")
if validate_results(results):
    for result in results:
        print(result.long_text)
```

### Advanced Configuration Example
```python
from dspy import Retrieve, ChainOfThought
import os

def setup_vectara_retriever(
    customer_id=None, 
    corpus_ids=None, 
    api_key=None, 
    num_results=5
):
    
    customer_id = customer_id or os.environ.get("VECTARA_CUSTOMER_ID")
    corpus_ids = corpus_ids or os.environ.get("VECTARA_CORPUS_ID")
    api_key = api_key or os.environ.get("VECTARA_API_KEY")
    
    
    retriever = VectaraRM(
        vectara_customer_id=customer_id,
        vectara_corpus_id=corpus_ids,
        vectara_api_key=api_key,
        k=num_results
    )
    
    return retriever


class VectaraSearchPipeline:
    def __init__(self):
        self.retriever = setup_vectara_retriever(num_results=5)
        self.qa_chain = ChainOfThought("question, context -> answer")
    
    def search(self, query):
        try:
            results = self.retriever(query)
            if not results:
                return "No relevant information found."
            
            context = "\n".join([r.long_text for r in results])
            answer = self.qa_chain(question=query, context=context)
            return answer
        except Exception as e:
            return f"Error during search: {str(e)}"


pipeline = VectaraSearchPipeline()
result = pipeline.search("What are the latest advances in quantum computing?")
```

**Note:**

- Ensure proper error handling in production environments
- Monitor API usage and response times
- Consider implementing retry logic for failed requests
- Validate results before processing
- Use environment variables for sensitive credentials




### Multiple Queries Support

The VectaraRM supports efficient handling of multiple queries with intelligent score aggregation. Here's a detailed breakdown:

```python
from dspy.retrieve.vectara_rm import VectaraRM
import os


retriever = VectaraRM(
    vectara_customer_id=os.getenv("VECTARA_CUSTOMER_ID"),
    vectara_corpus_id=os.getenv("VECTARA_CORPUS_ID"),
    vectara_api_key=os.getenv("VECTARA_API_KEY"),
    k=5
)


queries = [
    "machine learning fundamentals",
    "deep learning basics",
    "neural networks introduction"
]

results = retriever(queries, k=3)
```

#### Score Aggregation Deep Dive
```python

class SearchExample:
    def __init__(self):
        self.retriever = VectaraRM(k=5)
    
    def search_with_variations(self, base_query):
        queries = [
            base_query,
            f"basics of {base_query}",
            f"{base_query} introduction"
        ]
        
        results = self.retriever(queries, k=5)
        
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Text: {result.long_text}")
            print("-" * 50)
        
        return results


searcher = SearchExample()
results = searcher.search_with_variations("quantum computing")
```

#### Advanced Usage with Score Analysis
```python
def analyze_multiple_queries(queries, k=5):
    retriever = VectaraRM(k=k)
    
    
    individual_results = {}
    for query in queries:
        results = retriever(query)
        individual_results[query] = results
    
    
    combined_results = retriever(queries)
    
    print("Individual Query Results:")
    for query, results in individual_results.items():
        print(f"\nQuery: {query}")
        for i, res in enumerate(results, 1):
            print(f"Result {i}: {res.long_text[:100]}...")
    
    print("\nCombined Results (with score aggregation):")
    for i, res in enumerate(combined_results, 1):
        print(f"Result {i}: {res.long_text[:100]}...")
    
    return individual_results, combined_results


queries = [
    "quantum computing applications",
    "quantum computing in cryptography",
    "quantum computing future impact"
]

individual, combined = analyze_multiple_queries(queries)
```

**Note:**

- When using multiple queries, VectaraRM automatically:
  - Increases the internal search limit to 3*k for better coverage
  - Aggregates scores for duplicate passages across queries
  - Sorts final results by combined relevance scores
- The final number of results will still be limited to the specified k value
- Empty queries are automatically filtered out
- Score aggregation helps surface passages that are relevant to multiple query variations

### Using Environment Variables

Setting up VectaraRM with environment variables provides a secure way to manage your credentials while keeping them separate from your code.

#### Basic Environment Setup
```bash

export VECTARA_CUSTOMER_ID="your_customer_id"
export VECTARA_CORPUS_ID="your_corpus_id"
export VECTARA_API_KEY="your_api_key"
```

#### Using Environment Variables in Code
```python
from dspy.retrieve.vectara_rm import VectaraRM
import os


retriever = VectaraRM(k=5)  


retriever = VectaraRM(
    vectara_customer_id=os.getenv("VECTARA_CUSTOMER_ID"),
    vectara_corpus_id=os.getenv("VECTARA_CORPUS_ID"),
    vectara_api_key=os.getenv("VECTARA_API_KEY"),
    k=5
)
```

#### Environment Variable Validation
```python
def validate_vectara_env():
    required_vars = [
        "VECTARA_CUSTOMER_ID",
        "VECTARA_CORPUS_ID",
        "VECTARA_API_KEY"
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


try:
    validate_vectara_env()
    retriever = VectaraRM()
except ValueError as e:
    print(f"Environment setup error: {e}")
```

**Note:**

- Store sensitive credentials in environment variables, not in code
- Use appropriate environment management for different deployment environments
- Consider using a `.env` file for local development
- Never commit credentials to version control
- Validate environment variables before initialization

### Multiple Corpora Usage

VectaraRM supports searching across multiple corpora simultaneously by providing comma-separated corpus IDs. This allows you to search across different document collections while maintaining efficient retrieval and score aggregation.

#### Basic Multiple Corpora Setup
```python
from dspy.retrieve.vectara_rm import VectaraRM


retriever = VectaraRM(
    vectara_corpus_id="corpus_id_1,corpus_id_2,corpus_id_3",
    k=5
)


results = retriever("quantum computing")
```

#### Advanced Corpora Configuration
```python
class MultiCorpusSearch:
    def __init__(self, corpus_ids: list):
        corpus_string = ",".join(corpus_ids)
        
        self.retriever = VectaraRM(
            vectara_corpus_id=corpus_string,
            k=5
        )
    
    def search(self, query: str):
        return self.retriever(query)


corpus_ids = ["corpus1", "corpus2", "corpus3"]
searcher = MultiCorpusSearch(corpus_ids)
results = searcher.search("machine learning applications")
```

**Note:**

- Each corpus in the list must be accessible with your credentials
- Results are automatically aggregated across all specified corpora
- The lexical interpolation configuration (lambda=0.025) applies to all corpora
- The same context configuration (sentences before/after) applies across all corpora

### Integration with DSPy

VectaraRM seamlessly integrates with other DSPy components to create powerful retrieval-augmented applications. Here are examples of common integration patterns:

#### Basic RAG Integration
```python
from dspy import Retrieve, ChainOfThought
import dspy

class BasicRAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = VectaraRM(k=num_passages)
        self.generate_answer = ChainOfThought("question, context -> answer")
    
    def forward(self, question):
        context = self.retrieve(question)
        answer = self.generate_answer(
            question=question,
            context=context
        )
        return answer


rag = BasicRAG()
response = rag("What is quantum cryptography?")
```

#### Advanced DSPy Integration
```python
from dspy import Predict, Retrieve

class EnhancedRetriever(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = VectaraRM(k=5)
        self.predict = Predict("query, context -> answer, evidence")
    
    def forward(self, query):
        
        passages = self.retrieve(query)
        
        
        response = self.predict(
            query=query,
            context=[p.long_text for p in passages]
        )
        
        return {
            'answer': response.answer,
            'evidence': response.evidence,
            'passages': passages
        }


turbo = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=turbo)

retriever = EnhancedRetriever()
result = retriever("Explain the impact of quantum computing on cybersecurity")
```

**Note:**


- VectaraRM can be used as a drop-in replacement for any DSPy retriever
- The retrieved passages are automatically formatted for DSPy compatibility
- Consider adjusting the number of passages (k) based on your specific use case
- The retriever can be used in any DSPy pipeline or module


### Error Handling

VectaraRM includes comprehensive error handling to manage common issues with API requests, authentication, and result processing. Here are patterns for robust error handling:

#### Basic Error Handling
```python
from dspy.retrieve.vectara_rm import VectaraRM
import requests
import json

def initialize_safe_retriever():
    try:
        retriever = VectaraRM(k=5)
        
        test_result = retriever("test query")
        return retriever
    except requests.exceptions.RequestException as e:
        print(f"API Connection Error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid API Response: {e}")
        return None
    except Exception as e:
        print(f"Initialization Error: {e}")
        return None
```

#### Robust Query Implementation
```python
class RobustVectaraRetriever:
    def __init__(self, max_retries=3):
        self.retriever = VectaraRM()
        self.max_retries = max_retries
    
    def safe_retrieve(self, query):
        for attempt in range(self.max_retries):
            try:
                results = self.retriever(query)
                if results:
                    return results
                print(f"No results found (Attempt {attempt + 1}/{self.max_retries})")
            except requests.exceptions.Timeout:
                print(f"Request timed out (Attempt {attempt + 1}/{self.max_retries})")
                continue
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e} (Attempt {attempt + 1}/{self.max_retries})")
                continue
        
        return []  


retriever = RobustVectaraRetriever()
results = retriever.safe_retrieve("quantum computing")
```

#### Common Error Scenarios
```python
def handle_vectara_errors():
    try:
        retriever = VectaraRM(vectara_api_key="invalid_key")
        results = retriever("test")
    except Exception as e:
        print(f"Authentication failed: {e}")
    
    
    try:
        retriever = VectaraRM(vectara_corpus_id="invalid_corpus")
        results = retriever("test")
    except Exception as e:
        print(f"Invalid corpus ID: {e}")
    
    
    try:
        results = retriever("test", timeout=1)  
    except requests.exceptions.Timeout:
        print("Request timed out")
```

**Note:**

- Always implement proper error handling in production environments
- Consider implementing retry logic for transient failures
- Validate API responses before processing
- Handle timeout scenarios appropriately
- Log errors for monitoring and debugging
- Check response status codes for API-specific error information

Let's wrap up with a comprehensive section showing all the advanced configurations and best practices:

### Advanced Usage and Best Practices

#### Complete Configuration Example
```python
from dspy.retrieve.vectara_rm import VectaraRM
import os
from typing import List, Dict

class VectaraSearchManager:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'customer_id': os.getenv('VECTARA_CUSTOMER_ID'),
            'corpus_ids': os.getenv('VECTARA_CORPUS_ID').split(','),
            'api_key': os.getenv('VECTARA_API_KEY'),
            'default_k': 5
        }
        
        self.retriever = self._initialize_retriever()
    
    def _initialize_retriever(self):
        return VectaraRM(
            vectara_customer_id=self.config['customer_id'],
            vectara_corpus_id=','.join(self.config['corpus_ids']),
            vectara_api_key=self.config['api_key'],
            k=self.config['default_k']
        )
    
    def search_with_metadata(self, query: str, k: int = None):
        results = self.retriever(query, k=k)
        return {
            'query': query,
            'num_results': len(results),
            'passages': [r.long_text for r in results]
        }
```

#### Performance Optimization
```python
class OptimizedVectaraSearch:
    def __init__(self):
        self.retriever = VectaraRM(k=5)
        self._cache = {}  
    
    def batch_search(self, queries: List[str]):
        """Efficient handling of multiple queries"""
        return self.retriever(queries)
    
    def cached_search(self, query: str):
        """Caching frequent queries"""
        if query in self._cache:
            return self._cache[query]
            
        results = self.retriever(query)
        self._cache[query] = results
        return results
```

#### Best Practices Implementation
```python
def vectara_best_practices():
    
    required_vars = ["VECTARA_CUSTOMER_ID", "VECTARA_CORPUS_ID", "VECTARA_API_KEY"]
    if not all(os.getenv(var) for var in required_vars):
        raise EnvironmentError("Missing required environment variables")
    
    
    retriever = VectaraRM(k=5)
    
    
    try:
        results = retriever("test query")
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return None
    
    # 4. Results validation
    if not results:
        print("No results found")
        return []
    
    return results
```

**Note:**

- Always validate environment setup before initialization
- Consider implementing caching for frequent queries
- Use batch processing when possible
- Monitor and log API usage
- Handle rate limits appropriately
- Keep authentication credentials secure
- Regularly validate corpus configurations
- Implement appropriate error handling and logging

