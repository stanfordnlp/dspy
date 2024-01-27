---
sidebar_position: 3
---

# Creating Custom RM Client

DSPy provides support for various retreival modules out of the box like `Colbertv2`, `AzureCognitiveSearch`, `Pinecone`, `Weaviate`, etc. Unlike LM module, creating a custom RM module is much more simple and flexible. As on now, DSPy has 2 ways to create a custom RM: the Pythonic way and the DSPythonic way. We'll take a look at both and underastand why both of them are actually doing the same thing and how you can implement both. Let's see!

## I/O of RM Client

Before understanding the implementation let's understand the idea and I/O in RM module. The input to an RM module is going to either be a single query or a list of queries, and the output from it should be the `topk` passages per query that we receive from a retrieval model or a vector store or a search client. 

![I/O in RM Module](./img/io_rm_module.png)

Conventionally we simply call the RM module object as is with query/queries as argument of the call, that means we need to implement `__call__` method in the custom RM module class. As for output we need to return a list of string in some form or manner. We'll see how this I/O is essentially same in both methods of implementation but differs in the medium of delivery.

## The Pythonic Way

If the input is list of queries or a single query, and the output is the list of topk passages then we basically want to create a class that takes queries as input and outputs list of topk passages when called. The means at minimum we want to write the retrieval logic in just the `__call__` method. That means in this method we need to implement 2 methods in class: `__init__` and `__call__`. So our custom RM module should be like:

```python
from typing import List, Union

class PythonicRMClient:
    def __init__(self):
        pass

    def __call__(self, query: Union[str, List[str]], k:int) -> Union[List[str], List[List[str]]]:
        pass
```

:::info
If you got intimidated by the type-hinting above don't worry it's nothing too crazy. `typing` is a package that provides various types to use to define argument and output type of function parameter and function output respectively.

`Union` provides use to define all possible types of the argument/output. So:
* `Union[str, List[str]]`: Assigned to `query` means it can either be a string or a list of string i.e. a single query or a list of queries.
* `Union[List[str], List[List[str]]]`: Assigned to the output of `__call__` means it can either be a string or a list of string i.e. a single query or a list of queries.
:::

So let's start by implementing `PythonicRMClient` for a local retreival model that is hosted on a API with only endpoint being `/`. We'll start by implementing the `__init__` method, which does nothing except initializing the class attributes.

```python
def __init__(self, url: str, port:int = None):
    self.url = f`{url}:{port}` if port else url
```

Pretty simple, attach port to url if present else use the url as is. Now it's time to write the retreival logic in `__call__` method:

```python
def __call__(self, query:str, k:int) -> List[str]:
    params = {"query": query, "k": k}
    response = requests.get(self.url, params=params)

    response = response.json()["retreived_passages"]    # List of top k passages
    return response
```

That's all pretty simple all we doing is calling the API making a request and getting back our list of **top k passages** as the response whick we return as is. Let's bring it all together and see how our RM class looks like:

```python
from typing import List

class PythonicRMClient:
    def __init__(self, url: str, port:int = None):
        self.url = f`{url}:{port}` if port else url

    def __call__(self, query:str, k:int) -> List[str]:
        # Only accept single query input, feel free to modify it to support 

        params = {"query": query, "k": k}
        response = requests.get(self.url, params=params)

        response = response.json()["retreived_passages"]    # List of top k passages
        return response
```

That's all!! This is the most basic way to implement a RM model and this is how RM models like `Colbertv2` and `AzureCognitiveSearch` were implemented in DSP v1. Things get a bit more streamlined in DSPy but the essence is still the same. Let's see how!

## The DSPythonic Way

In essence DSPythonic way is not much different than Pythonic way. The input remains the same but the output would now be a object of `dspy.Prediction` class, which is the standard format of output for any DSPy module as we saw in previous docs. Aside from that the class would now inherit `dspy.Retrieve` class, which will help us out with state management nothing that we should be concerned abotu for now. 

So for the DSPythonic way we need to implement `__init__` and `forward` method, we'll still call the object the same way as Pythonice but `__call__` method will call `forward` method as is. So you only need to implement the `forward` method. So our custom RM module should be like:

```python
import dspy
from typing import List, Union, Optional

class DSPythonicRMClient(dspy.Retrieve):
    def __init__(self, k:int):
        pass

    def forward(self, query: Union[str, List[str]], k:Optional[str]) -> dspy.Prediction:
        pass
```

Unlike `PythonicRMClient`, we initialize `k` as part of the initialization call and `forward` method will take query/queries as arguments with `k` being optional argument. This is because in `dspy.Retrieve` the `__init__` method needs `k` as an argument, this happens when we call `super().__init__()`.

We'll be implementing `DSPythonicRMClient` for the same local retreival model API we used above. We'll start by implementing the `__init__` method, which is pretty much same as the the `PythonicRMClient`.

```python
def __init__(self, url: str, port:int = None, k:int = 3):
    super().__init__(k=k)

    self.url = f`{url}:{port}` if port else url
```

We'll now implement the `forward` method which is the same except now we'll be return the output as `dspy.Prediction` object under the `passage` attribute which is standard among all the RM modules.

```python
def forward(self, query:str, k:Optional[int]) -> dspy.Prediction:
    params = {"query": query, "k": k if k else self.k}
    response = requests.get(self.url, params=params)

    response = response.json()["retreived_passages"]    # List of top k passages
    return dspy.Prediction(
        passages=response
    )
```

If you pass `k` as argument during call then it'll use that else it'll use `self.k`. Let's bring it all together and see how our RM class looks like:

```python
import dspy
from typing import List, Union, Optional

class DSPythonicRMClient(dspy.Retrieve):
    def __init__(self, url: str, port:int = None, k:int = 3):
        super().__init__(k=k)

        self.url = f`{url}:{port}` if port else url

    def forward(self, query_or_queries:str, k:Optional[int]) -> dspy.Prediction:
        params = {"query": query_or_queries, "k": k if k else self.k}
        response = requests.get(self.url, params=params)

        response = response.json()["retreived_passages"]    # List of top k passages
        return dspy.Prediction(
            passages=response
        )
```

That's all!! This is the way to implement a RM model the way DSPy intends and this is how more recent RM models like `QdrantRM`, `WeaviateRM`, etc. are implemented in DSPy. How do we use these retriever though?

## Using Custom RM Models

The work of the RM client is to return top k passages for a give query, as long as we understand that fact we can use it in any way made possible via Python and DSPy. Based on that we have two ways to use our custom rm client: Direct Method and using `dspy.Retreive`.

### Direct Method

The most straight forward way to use custom model is by directly using there object in the DSPy Pipeline. So let's take the following psuedo code of a DSPy Pipeline as example.

```python
class DSPyPipeline(dspy.Module):
    def __init__(self):
        super().__init__()

        url = "http://0.0.0.0"
        port = 3000

        self.pythonic_rm = PythonicRMClient(url=url, port=port)
        self.dspythonic_rm = DSPythonicRMClient(url=url, port=port, k=3)

        ...

    def forward(self, *args):
        ...

        passages_from_pythonic_rm = self.pythonic_rm(query)
        passages_from_dspythonic_rm = self.dspythonic_rm(query).passages

        ...
```

As long as you are getting the list of passages you should be good to go!!

### Using `dspy.Retrieve`

This is the way that is more experimental in essence, through which you can keep the pipeline the same and still experiment with different RM model. How? By configuring it!

```python
import dspy

lm = ...
url = "http://0.0.0.0"
port = 3000

# pythonic_rm = PythonicRMClient(url=url, port=port)
dspythonic_rm = DSPythonicRMClient(url=url, port=port, k=3)

dspy.settings.configure(lm=lm, rm=dspythonic_rm)
```

Now, in the pipeline you just need to use `dspy.Retrieve` which'll use this `rm` client to get the topk passage for a given query!

```python
class DSPyPipeline(dspy.Module):
    def __init__(self):
        super().__init__()

        url = "http://0.0.0.0"
        port = 3000

        self.rm = dspy.Retrieve(k=3)
        ...

    def forward(self, *args):
        ...

        passages = self.rm(query)

        ...
```

Now in case you need to use a different rm you just need to update the `rm` parameter via `dspy.settings.configure`.

:::info[How `dspy.Retrieve` uses `rm`]
When we call `dspy.Retrieve` the `__call__` method will execute `forward` method as is. In `forward` method, the topk passages are received by the `dsp.retrieveEnsemble` method in [search.py](https://github.com/stanfordnlp/dspy/blob/main/dsp/primitives/search.py).

Inside `retrieveEnsemble` method is `rm` is not initialized in `dsp.settings` an error would be raised but if it is initialized, it'll call the `retrieve` method which uses `dsp.settings.rm` that we configured to get the top k passages.
:::