---
sidebar_position: 5
---

# dsp.PremAI

[PremAI](https://app.premai.io)  is an all-in-one platform that simplifies the process of creating robust, production-ready applications powered by Generative AI. By streamlining the development process, PremAI allows you to concentrate on enhancing user experience and driving overall growth for your application.

### Prerequisites

Refer to the [quick start](https://docs.premai.io/introduction) guide to getting started with the PremAI platform, create your first project and grab your API key.

### Usage

Please make sure you have premai python sdk installed. Otherwise you can do it using this command:

```bash
pip install -U premai
```

Here is a quick example on how to use premai python sdk with dspy

```python
from dspy import PremAI

llm = PremAI(model='mistral-tiny', project_id=123, api_key="your-premai-api-key")
print(llm("what is a large language model"))
```

> Please note: Project ID 123 is just an example. You can find your project ID inside our platform under which you created your project.

### Constructor

The constructor initializes the base class `LM` and verifies the `api_key` provided or defined through the `PREMAI_API_KEY` environment variable.

```python
class PremAI(LM):
    def __init__(
        self,
        model: str,
        project_id: int,
        api_key: str,
        base_url: Optional[str] = None,
        session_id: Optional[int] = None,
        **kwargs,
    ) -> None:
```

**Parameters:**

- `model` (_str_): Models supported by PremAI. Example: `mistral-tiny`. We recommend using the model selected in [project launchpad](https://docs.premai.io/get-started/launchpad).
- `project_id` (_int_): The [project id](https://docs.premai.io/get-started/projects) which contains the model of choice.
- `api_key` (_Optional[str]_, _optional_): API provider from PremAI. Defaults to None.
- `**kwargs`: Additional language model arguments will be passed to the API provider.

### Methods

#### `__call__(self, prompt: str, **kwargs) -> List[Dict[str, Any]]`

Retrieves completions from PremAI by calling `request`.

Internally, the method handles the specifics of preparing the request prompt and corresponding payload to obtain the response.

**Parameters:**

- `prompt` (_str_): Prompt to send to PremAI.
- `**kwargs`: Additional keyword arguments for completion request. Example: parameters like `temperature`, `max_tokens` etc. You can find all the additional kwargs [here](https://docs.premai.io/get-started/sdk#optional-parameters).

### Native RAG Support

Prem Repositories which allows users to upload documents (.txt, .pdf etc) and connect those repositories to the LLMs. You can think Prem repositories as native RAG, where each repository can be considered as a vector database. You can connect multiple repositories. You can learn more about repositories [here](https://docs.premai.io/get-started/repositories).

Repositories are also supported in dspy premai. Here is how you can do it. 

```python
query = "what is the diameter of individual Galaxy"
repository_ids = [1991, ]
repositories = dict(
    ids=repository_ids,
    similarity_threshold=0.3,
    limit=3
)
```

First we start by defining our repository with some repository ids. Make sure that the ids are valid repository ids. You can learn more about how to get the repository id [here](https://docs.premai.io/get-started/repositories). 

> Please note: Similar like `model` when you invoke the argument `repositories`, then you are potentially overriding the repositories connected in the launchpad. 

Now, we connect the repository with our chat object to invoke RAG based generations. 

```python 
response = llm(query, max_tokens=100, repositories=repositories)

print(response)
print("---")
print(json.dumps(llm.history, indent=4))
```

Here is how an example generation would look like with Prem Repositories. 

```bash
'The diameters of individual galaxies range from 80,000-150,000 light-years.'
---
[
    {
        "prompt": "what is the diameter of individual Galaxy",
        "response": "The diameters of individual galaxies range from 80,000-150,000 light-years.",
        "document_chunks": [
            {
                "repository_id": 1991,
                "document_id": 1307,
                "chunk_id": 173926,
                "document_name": "Kegy 202 Chapter 2",
                "similarity_score": 0.586126983165741,
                "content": "n thousands\n                                                                                                                                               of           light-years. The diameters of individual\n                                                                                                                                               galaxies range from 80,000-150,000 light\n                                                                                                                       "
            },
            {
                "repository_id": 1991,
                "document_id": 1307,
                "chunk_id": 173925,
                "document_name": "Kegy 202 Chapter 2",
                "similarity_score": 0.4815782308578491,
                "content": "                                                for development of galaxies. A galaxy contains\n                                                                                                                                               a large number of stars. Galaxies spread over\n                                                                                                                                               vast distances that are measured in thousands\n                                       "
            },
            {
                "repository_id": 1991,
                "document_id": 1307,
                "chunk_id": 173916,
                "document_name": "Kegy 202 Chapter 2",
                "similarity_score": 0.38112708926200867,
                "content": " was separated from the               from each other as the balloon expands.\n  solar surface. As the passing star moved away,             Similarly, the distance between the galaxies is\n  the material separated from the solar surface\n  continued to revolve around the sun and it\n  slowly condensed into planets. Sir James Jeans\n  and later Sir Harold Jeffrey supported thisnot to be republishedalso found to be increasing and thereby, the\n                                                             universe is"
            }
        ],
        "kwargs": {
            "max_tokens": 100,
            "repositories": {
                "ids": [
                    1991
                ],
                "similarity_threshold": 0.3,
                "limit": 3
            }
        },
        "raw_kwargs": {
            "max_tokens": 100,
            "repositories": {
                "ids": [
                    1991
                ],
                "similarity_threshold": 0.3,
                "limit": 3
            }
        }
    }
]
```

So, this also means that you do not need to make your own RAG pipeline when using the Prem Platform. Prem uses it's own RAG technology to deliver best in class performance for Retrieval Augmented Generations.

> Ideally, you do not need to connect Repository IDs here to get Retrieval Augmented Generations. You can still get the same result if you have connected the repositories in prem platform. 