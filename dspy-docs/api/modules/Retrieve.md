# dspy.Retrieve

### Constructor

The constructor initializes the `Retrieve` class and sets up its attributes, taking in `k` number of retrieval passages to return for a query.

```python
class Retrieve(Parameter):
    def __init__(self, k=3):
        self.stage = random.randbytes(8).hex()
        self.k = k
```

**Parameters:**
- `k` (_Any_): Number of retrieval responses

### Method

#### `__call__(self, *args, **kwargs):`

This method serves as a wrapper for the `forward` method. It allows making retrievals on an input query using the `Retrieve` class.

**Parameters:**
- `**args`: Arguments required for retrieval.
- `**kwargs`: Keyword arguments required for retrieval.

**Returns:**
- The result of the `forward` method.

### Examples

```python
query='When was the first FIFA World Cup held?'

# Call the retriever on a particular query.
retrieve = dspy.Retrieve(k=3)
topK_passages = retrieve(query).passages

print(f"Top {retrieve.k} passages for question: {query} \n", '-' * 30, '\n')

for idx, passage in enumerate(topK_passages):
    print(f'{idx+1}]', passage, '\n')
```
