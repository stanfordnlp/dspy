import AuthorDetails from '@site/src/components/AuthorDetails';

# ColBERTv2

## Setting up the ColBERTv2 Client

The constructor initializes the `ColBERTv2` class instance and sets up the request parameters for interacting with the ColBERTv2 retrieval server. This server is hosted remotely at `'http://20.102.90.50:2017/wiki17_abstracts`. 

- `url` (_str_): URL for ColBERTv2 server. Defaults to `"http://0.0.0.0"`
- `port` (_Union[str, int]_, _Optional_): Port endpoint for ColBERTv2 server. Defaults to `None`.
- `post_requests` (_bool_, _Optional_): Flag for using HTTP POST requests. Defaults to `False`.

Example of the ColBERTv2 constructor:

```python
class ColBERTv2:
    def __init__(
        self,
        url: str = "http://0.0.0.0",
        port: Optional[Union[str, int]] = None,
        post_requests: bool = False,
    ):
```

## Under the Hood

### `__call__(self, query: str, k: int = 10, simplify: bool = False) -> Union[list[str], list[dotdict]]`

**Parameters:**
- `query` (_str_): Search query string used for retrieval sent to ColBERTv2 server.
- `k` (_int_, _optional_): Number of passages to retrieve. Defaults to 10.
- `simplify` (_bool_, _optional_): Flag for simplifying output to a list of strings. Defaults to False.

**Returns:**
- `Union[list[str], list[dotdict]]`: Depending on `simplify` flag, either a list of strings representing the passage content (`True`) or a list of `dotdict` instances containing passage details (`False`).

Internally, the method handles the specifics of preparing the request query to the ColBERTv2 server and corresponding payload to obtain the response. 

The function handles the retrieval of the top-k passages based on the provided query.

If `post_requests` is set, the method sends a query to the server via a POST request else via a GET request.

It then processes and returns the top-k passages from the response with the list of retrieved passages dependent on the `simplify` flag return condition above.


## Sending Retrieval Requests via ColBERTv2 Client
1) _**Recommended**_ Configure default RM using `dspy.configure`.

This allows you to define programs in DSPy and have DSPy internally conduct retrieval using `dsp.retrieve` on the query on the configured RM.

```python
import dspy
import dsp

dspy.settings.configure(rm=colbertv2_wiki17_abstracts)
retrieval_response = dsp.retrieve("When was the first FIFA World Cup held?", k=5)

for result in retrieval_response:
    print("Text:", result, "\n")
```


2) Generate responses using the client directly.
```python
import dspy

retrieval_response = colbertv2_wiki17_abstracts('When was the first FIFA World Cup held?', k=5)

for result in retrieval_response:
    print("Text:", result['text'], "\n")
```

***

<AuthorDetails name="Arnav Singhvi"/>