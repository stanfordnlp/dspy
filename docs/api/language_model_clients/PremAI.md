---
sidebar_position: 5
---

# dsp.PremAI

[PremAI](https://app.premai.io) is an all-in-one platform that simplifies the process of creating robust, production-ready applications powered by Generative AI. By streamlining the development process, PremAI allows you to concentrate on enhancing user experience and driving overall growth for your application.

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

### Prem Templates

Writing Prompt Templates can be super messy. Prompt templates are long, hard to manage, and must be continuously tweaked to improve and keep the same throughout the application.

With **Prem**, writing and managing prompts can be super easy. The **_Templates_** tab inside the [launchpad](https://docs.premai.io/get-started/launchpad) helps you write as many prompts you need and use it inside the SDK to make your application running using those prompts. You can read more about Prompt Templates [here](https://docs.premai.io/get-started/prem-templates).

Using templates in DsPY is quite easy. First, here is an example of a prompt template which you store / re-iterate inside Prem.

```text
Say hello to my name and say a feel-good quote
from my age. My name is: {name} and age is {age}
```

Assuming this prompt template is stored under a template with id: `78069ce8-xxxxx-xxxxx-xxxx-xxx` then this is how you will be using it.

```python
from dspy import PremAI

client = PremAI(project_id=1234)
template_id = "78069ce8-xxxxx-xxxxx-xxxx-xxx"

response = c
```

### Native RAG Support

PremAI Repositories allow users to upload documents (.txt, .pdf, etc.) and connect those repositories to the LLMs to serve as vector databases and support native RAG. You can learn more about PremAI repositories [here](https://docs.premai.io/get-started/repositories).

Repositories are also supported through the dspy-premai integration. Here is how you can use this workflow:

```python
query = "what is the diameter of individual Galaxy"
repository_ids = [1991, ]
repositories = dict(
    ids=repository_ids,
    similarity_threshold=0.3,
    limit=3
)
```

First, we start by defining our repository with some valid repository ids. You can learn more about how to get the repository id [here](https://docs.premai.io/get-started/repositories).

> Note: This is similar to LM integrations where now you are overriding the repositories connected in the launchpad when you invoke the argument' repositories'.

Now, we connect the repository with our chat object to invoke RAG-based generations.

```python
response = llm(query, max_tokens=100, repositories=repositories)

print(response)
print("---")
print(json.dumps(llm.history, indent=4))
```

Here is what an example generation would look like with PremAI Repositories.

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
            },`
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

So this also means that you do not need to create your own RAG pipeline when using the PremAI Platform and can instead take advantage of its local RAG technology to deliver best-in-class performance for Retrieval Augmented Generations.

> Ideally, you do not need to connect Repository IDs here to get Retrieval Augmented Generations. You can still get the same result if you have connected the repositories in PremAI platform.
