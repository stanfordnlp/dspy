# Retrieve

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"

## Background
DSPy supports retrieval through the Retrieve module that serves to process user queries and output relevant passages from retrieval corpuses. This module ties in with the DSPy-supported Retrieval Model Clients which are retrieval servers that users can utilize to retrieve information for information retrieval tasks.

## Instantiating Retrieve

Retrieve is simply instantiate by a user-defined `k` number of retrieval passages to return for a query.

```python
class Retrieve(Parameter):
    def __init__(self, k=3):
        self.stage = random.randbytes(8).hex()
        self.k = k
```

Additionally, configuring a Retrieval model client or server through `dspy.configure` allows for user retrieval in DSPy programs through internal calls from Retrieve. 

```python
#Example Usage

#Define a retrieval model server to send retrieval requests to
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

#Configure retrieval server internally
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)

#Define Retrieve Module
retriever = dspy.Retrieve(k=3)
```

### Under the Hood

Retrieve makes use of the internally configured retriever to send a single query or list of multiple queries to determine the top-k relevant passages. The module queries the retriever for each provided query, accumulating scores (or probabilities if the `by_prob` arg is set) for each passage and returns the passages sorted by their cumulative scores or probabilities. 

The Retrieve module can also integrate a reranker if this is configured, in which case, the reranker re-scores the retrieved passages based on their relevance to the quer, improving accuracy of the results. 

### Tying it All Together

We can now call the Retrieve module on a sample query or list of queries and observe the top-K relevant passages.

```python
query='When was the first FIFA World Cup held?'

# Call the retriever on a particular query.
topK_passages = retriever(query).passages

print(f"Top {retriever.k} passages for question: {query} \n", '-' * 30, '\n')

for idx, passage in enumerate(topK_passages):
    print(f'{idx+1}]', passage, '\n')
```

```
Top 3 passages for question: When was the first FIFA World Cup held? 
 ------------------------------ 

1] History of the FIFA World Cup | The FIFA World Cup was first held in 1930, when FIFA president Jules Rimet [...]. 

2] 1950 FIFA World Cup | The 1950 FIFA World Cup, held in Brazil from 24 June to 16 July 1950, [...]. 

3] 1970 FIFA World Cup | The 1970 FIFA World Cup was the ninth FIFA World Cup, the quadrennial [...].
```
