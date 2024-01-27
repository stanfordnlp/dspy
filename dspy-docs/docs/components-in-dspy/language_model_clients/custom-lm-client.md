---
sidebar_position: 3
---

# Creating Custom LM Client

DSPy provides you with multiple LM clients that you can use to excute any of your pipeline. However, in case you have an API or local model that is not able to be executed by any of the existing client then you can create one yourself!! It's not too difficult so let's see how!!

## Format of LM Client

An LM client needs to implement 3 methods at minimum: `__init__`, `basic_request` and `__call__`. So your custom LM client should follow the template below:

```python
from dsp import LM


class CustomLMClient(LM):
    def __init__(self):
        self.provider = "default"

        self.history = []

    def basic_request(self, prompt, **kwargs):
        pass

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        pass
```

While for the most part you are free to add and customize the client in anyway you see fit, but some components need to be present and work in a certain way so you can utilize every feature of DSPy without interruption. On of these feture being viewing history of calls made to LLM via `inspect_history`. To elaborate:

* `__init__`: Should contain the `self.provider="default` and `self.history=[]`. `self.history` will contain the prompt-completion pair created via LLM call since the object was initialized. `self.provider` is used in `inspect_history` method andfor most part you can leave it as **"default"**.
* `basic_request`: This function makes the call to LLM and get the completion for the given prompt over the given `kwargs` which usually have parameters like `temperature`, `max_tokens`, etc. After you get the completion is received from the LLM you must update the `self.history` list by appending the dictionary `{"prompt": prompt, "response": response}` feel free to add anything else but these 2 are mandatory.
* `__call__`: This function should return the list of completions model returned, in the model basic case this is just a list of string with string being the completion. However it can be a tuple too like `Cohere` LM client returns the list of tuple where the tuple is a pair of completion and it's likelihood. Aside from the you must receive the completion via `request` call which unless modified just calls `basic_request` as is. This is to keep the history updated and that's all.

By now you must've realized the reason we have these rules is mainly for making the history inpection and modules work without breaking.

:::info
You can mitigate the history issue by updating the history in the `__call__` itself. So if you can take care of history updates in `__call__` itself you just need to implement `__init__` and `__call__`.

Or if you are up for ir rewrite `inspect_history` method in the class.
:::

## Implementing our Custom LM

Based on whatever we learned until now let's implement our custom LLM that calls the Claude's API. In Claude we need to initialize 2 things to make a successful call: `API_KEY` and `BASE_URL`. The base URL based on the [**docs**](https://docs.anthropic.com/claude/reference/messages_post) is `https://api.anthropic.com/v1/messages`. Let's write our `__init__` method:

```python
def __init__(model, api_key):
    self.model = model
    self.api_key = api_key
    self.provider = "default"

    self.base_url = "https://api.anthropic.com/v1/messages"
```

Based on the above implementation you might've realized that we need to pass 2 parameters when we'll initialize this LM client i.e. the `model` like `claude-2` and the `api_key` which you'll get in Claude's API Console. Now it's time to define the `basic_request` method where we'll make the call to `self.base_url` and get the completion:

```python
def basic_request(self, prompt: str, **kwargs):
    headers = {
        "x-api-key": self.api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "messages-2023-12-15",
        "content-type": "application/json"
    }

    data = {
        **kwargs,
        "model": self.model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(self.base_url, headers=headers, json=data)
    response = response.json()

    self.history.append({
        "prompt": prompt,
        "response": response,
        "kwargs": kwargs,
    })
    return response
```

Now it's time to define `__call__` method that'll bring it all together:

```python
def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
    response = self.request(prompt, **kwargs)

    completions = [result["text"] for result in response["content"]]

    return completions
```

To write it all in a single class we'll get:

```python
from dsp import LM

class Claude(LM):
    def __init__(model, api_key):
        self.model = model
        self.api_key = api_key
        self.provider = "default"

        self.base_url = "https://api.anthropic.com/v1/messages"

    def basic_request(self, prompt: str, **kwargs):
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "messages-2023-12-15",
            "content-type": "application/json"
        }

        data = {
            **kwargs,
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(self.base_url, headers=headers, json=data)
        response = response.json()

        self.history.append({
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
        })
        return response

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        response = self.request(prompt, **kwargs)

        completions = [result["text"] for result in response["content"]]

        return completions
```

That's it! Now we can configure this as an `lm` in DSPy and use it in the pipeline like any other LM Client:

```python
import dspy

claude = Claude(model='claude-2')

dspy.settings.configure(lm=claude)
```