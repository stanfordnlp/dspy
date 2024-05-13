---
sidebar_position: 5
---

# dsp.PremAI

[PremAI](https://app.premai.io) is a unified platform that lets you build powerful production-ready GenAI-powered applications with the least effort so that you can focus more on user experience and overall growth. With dspy, you can connect with several [best-in-class LLMs](https://models.premai.io/) of your choice with a single interface.

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
- `session_id` (_Optional[int]_, _optional_): The ID of the session to use. It helps to track the chat history.
- `**kwargs`: Additional language model arguments will be passed to the API provider.

### Methods

#### `__call__(self, prompt: str, **kwargs) -> List[Dict[str, Any]]`

Retrieves completions from PremAI by calling `request`.

Internally, the method handles the specifics of preparing the request prompt and corresponding payload to obtain the response.

After generation, the completions are post-processed based on the `model_type` parameter.

**Parameters:**

- `prompt` (_str_): Prompt to send to PremAI.
- `**kwargs`: Additional keyword arguments for completion request. Example: parameters like `temperature`, `max_tokens` etc. You can find all the additional kwargs [here](https://docs.premai.io/get-started/sdk#optional-parameters).
