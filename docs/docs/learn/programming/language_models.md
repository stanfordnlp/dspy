---
sidebar_position: 2
---

# Language Models

The first step in any DSPy code is to set up your language model. For example, you can configure OpenAI's GPT-4o-mini as your default LM as follows.

```python linenums="1"
# Authenticate via `OPENAI_API_KEY` env: import os; os.environ['OPENAI_API_KEY'] = 'here'
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)
```

## Automatic routing vs. explicit backend control

!!! warning "Normalized LM routing is experimental"
    The normalized `BaseLM` router is enabled with `dspy.configure(experimental=True)` or `dspy.context(experimental=True)`. In the current stable path, `dspy.LM(...)` returns the legacy LiteLLM-backed LM. The normalized router is planned to become the default in a future DSPy release.

With normalized routing enabled, use `dspy.LM(...)` when you want DSPy to make its best routing decision from the model name and provider metadata.

Use a concrete normalized LM class when you need exact control over the endpoint family:

```python linenums="1"
# OpenAI-compatible Chat Completions endpoint.
lm = dspy.OpenAIChatLM(
    "llama3.2",
    api_base="http://localhost:11434/v1",
    api_key="ollama",
)

# OpenAI-compatible Responses endpoint.
lm = dspy.OpenAIResponsesLM(
    "openai/gpt-oss-120b",
    api_base="https://api.groq.com/openai/v1",
    api_key="YOUR_GROQ_API_KEY",
)
```

Think of the split this way:

- `dspy.LM(...)`: automatic routing and sensible defaults.
- `dspy.OpenAIChatLM(...)`: force an OpenAI-compatible `/chat/completions` endpoint.
- `dspy.OpenAITextLM(...)`: force an OpenAI-compatible legacy `/completions` endpoint.
- `dspy.OpenAIResponsesLM(...)`: force an OpenAI-compatible `/responses` endpoint.
- `dspy.AnthropicLM(...)` and `dspy.GenAILM(...)`: force native Anthropic or Google GenAI backends.
- Custom `dspy.BaseLM` subclasses: use only when the provider is not OpenAI-compatible and needs a different request or response shape.

With normalized routing enabled, common OpenAI-compatible providers can be selected by prefix:

```python linenums="1"
with dspy.context(experimental=True):
    groq_lm = dspy.LM("groq/llama-3.3-70b-versatile")
    groq_responses_lm = dspy.LM("groq/openai/gpt-oss-120b")
    ollama_lm = dspy.LM("ollama/llama3.2")
```

DSPy routes those to the appropriate OpenAI-compatible backend and fills in the standard base URL. Set the corresponding environment variable, such as `GROQ_API_KEY`, or pass `api_key` explicitly.

### Registering custom LM routes

If your organization uses a provider prefix that DSPy does not know yet, register a lightweight route instead of writing a new LM class. A route receives the same arguments as `dspy.LM(...)`. It should return a concrete `BaseLM` when it owns the model, or `None` to let DSPy try the next route.

```python linenums="1"
import os
import dspy

@dspy.register_lm_backend
def route_my_gateway(model: str, *args, **kwargs):
    if not model.startswith("mygateway/"):
        return None

    provider_model = model.removeprefix("mygateway/")
    kwargs.setdefault("api_base", "https://gateway.example.com/openai/v1")
    kwargs.setdefault("api_key", os.environ.get("MYGATEWAY_API_KEY"))
    return dspy.OpenAIChatLM(provider_model, *args, **kwargs)

with dspy.context(experimental=True):
    lm = dspy.LM("mygateway/llama-3.3-70b-versatile")
```

You can also register a full `BaseLM` subclass by prefix:

```python linenums="1"
class AcmeLM(dspy.BaseLM):
    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        ...

dspy.register_lm_backend(AcmeLM, prefix="acme")

with dspy.context(experimental=True):
    lm = dspy.LM("acme/small")
```

You usually do **not** need a new subclass for Groq, Ollama, SGLang, vLLM, LM Studio, OpenRouter, Fireworks, Together, or other OpenAI-compatible services. Route them to `OpenAIChatLM`, `OpenAITextLM`, or `OpenAIResponsesLM`. Write a subclass only when the provider has a different request shape, response shape, streaming format, or authentication flow.

### Unusual endpoint URLs

Most OpenAI-compatible services accept a base URL such as `https://provider.example.com/v1`, and DSPy appends `/chat/completions` or `/responses` for the concrete backend you choose. If your proxy uses a non-standard full endpoint path, pass `endpoint_url` instead of `api_base`:

```python linenums="1"
import dspy

lm = dspy.OpenAIChatLM(
    "my-model",
    endpoint_url="https://proxy.example.com/custom/path/invoke-chat",
    api_key="PROXY_API_KEY",
)
```

Use the same pattern with `OpenAIResponsesLM(..., endpoint_url="https://proxy.example.com/custom/path/invoke-responses")` for a non-standard Responses endpoint. If your proxy does not use an OpenAI-shaped JSON request and response at all, pass a custom callable with `OpenAIChatLM(..., completions=callable)` or `OpenAIResponsesLM(..., responses=callable)`, or write a small `BaseLM` subclass.

!!! info "A few different LMs"

    === "OpenAI"
        You can authenticate by setting the `OPENAI_API_KEY` env variable or passing `api_key` below.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_OPENAI_API_KEY')
        dspy.configure(lm=lm)
        ```

    === "Gemini (AI Studio)"
        You can authenticate by setting the GEMINI_API_KEY env variable or passing `api_key` below.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('gemini/gemini-2.5-pro-preview-03-25', api_key='GEMINI_API_KEY')
        dspy.configure(lm=lm)
        ```

    === "Anthropic"
        You can authenticate by setting the ANTHROPIC_API_KEY env variable or passing `api_key` below.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('anthropic/claude-sonnet-4-5-20250929', api_key='YOUR_ANTHROPIC_API_KEY')
        dspy.configure(lm=lm)
        ```

    === "Vertex AI (GCP)"
        For Google Cloud's Vertex AI, authenticate with a service account JSON key or Application Default Credentials. You can pass credentials directly in code or set environment variables.

        ```python linenums="1"
        import dspy
        import json

        # Load the service account JSON and convert to a string
        with open("service_account.json") as f:
            credentials = json.dumps(json.load(f))

        lm = dspy.LM(
            "vertex_ai/gemini-2.0-flash",
            vertex_credentials=credentials,
            vertex_project="your-gcp-project-id",
            vertex_location="us-central1",
        )
        dspy.configure(lm=lm)
        ```

        Alternatively, set environment variables and skip the kwargs:

        ```bash
        export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service_account.json"
        export VERTEXAI_PROJECT="your-gcp-project-id"
        export VERTEXAI_LOCATION="us-central1"
        ```

        ```python linenums="1"
        import dspy
        lm = dspy.LM("vertex_ai/gemini-2.0-flash")
        dspy.configure(lm=lm)
        ```

        !!! warning "Common pitfalls"
            - Use the `vertex_ai/` model prefix, not `gemini/`. The `gemini/` prefix routes to the Gemini API which requires an API key instead of GCP credentials.
            - Use `vertex_project` and `vertex_location`, not `project` or `location`. Parameters without the `vertex_` prefix are silently ignored and LiteLLM falls back to defaults, which may cause requests to land in an unintended region.

    === "Databricks"
        If you're on the Databricks platform, authentication is automatic via their SDK. If not, you can set the env variables `DATABRICKS_API_KEY` and `DATABRICKS_API_BASE`, or pass `api_key` and `api_base` below.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('databricks/databricks-meta-llama-3-1-70b-instruct')
        dspy.configure(lm=lm)
        ```

    === "Groq"
        You can authenticate by setting `GROQ_API_KEY` or passing `api_key` below. Groq exposes OpenAI-compatible endpoints.

        For most Groq chat models, use Chat Completions:

        ```python linenums="1"
        import os
        import dspy

        lm = dspy.OpenAIChatLM(
            "llama-3.3-70b-versatile",
            api_base="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
        )
        dspy.configure(lm=lm)
        ```

        For Groq models exposed through the Responses API, use `OpenAIResponsesLM`:

        ```python linenums="1"
        import os
        import dspy

        lm = dspy.OpenAIResponsesLM(
            "openai/gpt-oss-120b",
            api_base="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
        )
        dspy.configure(lm=lm)
        ```

    === "Local LMs on a GPU server"
          First, install [SGLang](https://sgl-project.github.io/start/install.html) and launch its server with your LM.

          ```bash
          > pip install "sglang[all]"
          > pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ 

          > CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --port 7501 --model-path meta-llama/Meta-Llama-3-8B-Instruct
          ```

          Then, connect to it from your DSPy code as an OpenAI-compatible endpoint.

          ```python linenums="1"
          import dspy

          lm = dspy.OpenAIChatLM(
              "meta-llama/Meta-Llama-3-8B-Instruct",
              api_base="http://localhost:7501/v1",  # ensure this points to your port
              api_key="local",
          )
          dspy.configure(lm=lm)
          ```

    === "Local LMs on your laptop"
          First, install [Ollama](https://github.com/ollama/ollama) and launch its server with your LM.

          ```bash
          > curl -fsSL https://ollama.ai/install.sh | sh
          > ollama run llama3.2:1b
          ```

          Then, connect to it from your DSPy code as an OpenAI-compatible endpoint.

        ```python linenums="1"
        import dspy

        lm = dspy.OpenAIChatLM(
            "llama3.2",
            api_base="http://localhost:11434/v1",
            api_key="ollama",
        )
        dspy.configure(lm=lm)
        ```

    === "Other providers"
        In DSPy, you can use any of the dozens of [LLM providers supported by LiteLLM](https://docs.litellm.ai/docs/providers). Simply follow their instructions for which `{PROVIDER}_API_KEY` to set and how to pass the `{provider_name}/{model_name}` to the constructor.

        Some examples:

        - `anyscale/mistralai/Mistral-7B-Instruct-v0.1`, with `ANYSCALE_API_KEY`
        - `together_ai/togethercomputer/llama-2-70b-chat`, with `TOGETHERAI_API_KEY`
        - `sagemaker/<your-endpoint-name>`, with `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION_NAME`
        - `azure/<your_deployment_name>`, with `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION`, and the optional `AZURE_AD_TOKEN` and `AZURE_API_TYPE` as environment variables. If you are initiating external models without setting environment variables, use the following:
        `lm = dspy.LM('azure/<your_deployment_name>', api_key='AZURE_API_KEY', api_base='AZURE_API_BASE', api_version='AZURE_API_VERSION')`

        If your provider offers an OpenAI-compatible endpoint, use the matching concrete backend. Most OpenAI-compatible providers support Chat Completions:

        ```python linenums="1"
        import dspy

        lm = dspy.OpenAIChatLM(
            "your-model-name",
            api_key="PROVIDER_API_KEY",
            api_base="https://provider.example.com/v1",
        )
        dspy.configure(lm=lm)
        ```

        If the provider supports the Responses API, use `OpenAIResponsesLM` instead:

        ```python linenums="1"
        import dspy

        lm = dspy.OpenAIResponsesLM(
            "your-model-name",
            api_key="PROVIDER_API_KEY",
            api_base="https://provider.example.com/v1",
        )
        dspy.configure(lm=lm)
        ```
If you run into errors, please refer to the [LiteLLM Docs](https://docs.litellm.ai/docs/providers) to verify if you are using the same variable names/following the right procedure.

## Calling the LM directly.

It's easy to call the `lm` you configured above directly. This gives you a unified API and lets you benefit from utilities like automatic caching.

```python linenums="1"       
lm("Say this is a test!", temperature=0.7)
lm(messages=[{"role": "user", "content": "Say this is a test!"}])
```

Legacy LiteLLM-backed LMs return a list of outputs, for example `['This is a test!']`. Normalized `BaseLM` backends return an `LMResponse`; use `response.text`, `response.outputs`, or `response.to_legacy_outputs()` depending on the level of detail you need.

### Message-style calls

!!! warning "Experimental LM API"
    This section describes the normalized `BaseLM` API introduced experimentally
    in DSPy 3.3 and enabled with `dspy.configure(experimental=True)`. In the current
    stable path, `dspy.LM(...)` returns the legacy LiteLLM-backed LM. The normalized
    LM path is planned to become the default in DSPy 3.5 and may change before then.

For multi-turn or multimodal calls, use DSPy message constructors. The canonical explicit form is `messages=[...]`:

```python linenums="1"
response = lm(
    messages=[
        dspy.System("Be concise."),
        dspy.User("What is DSPy?"),
    ],
    temperature=0.2,
)
```

For hand-written conversations, you can also pass messages positionally:

```python linenums="1"
response = lm(
    dspy.System("Be concise."),
    dspy.User("What is DSPy?"),
    temperature=0.2,
)
```

If you have already built a list of messages, pass it as the single positional argument:

```python linenums="1"
messages = [
    dspy.System("Be concise."),
    dspy.User("What is DSPy?"),
]

response = lm(messages)
```

This is one conversation, not a batch. `lm(...)` always represents one LM request.

### Multimodal message rules

Bare content parts create one implicit user message:

```python linenums="1"
response = lm(
    "Describe this image.",
    dspy.Image("https://example.com/dog.png"),
)
```

For multi-turn calls, put content parts inside the message they belong to:

```python linenums="1"
response = lm(
    dspy.System("Be concise."),
    dspy.User(
        "Describe this image.",
        dspy.Image("https://example.com/dog.png"),
    ),
)
```

Do not mix explicit messages with bare content parts:

```python linenums="1"
# Not allowed: the image has no explicit message turn.
lm(
    dspy.System("Be concise."),
    dspy.User("Describe this image."),
    dspy.Image("https://example.com/dog.png"),
)
```

Instead, place the image, audio, file, tool call, citation, or other content part inside `dspy.User(...)`, `dspy.Assistant(...)`, or `dspy.ToolResult(...)`.

## Using the LM with DSPy modules.

Idiomatic DSPy involves using _modules_, which we discuss in the next guide.

```python linenums="1" 
# Define a module (ChainOfThought) and assign it a signature (return an answer, given a question).
qa = dspy.ChainOfThought('question -> answer')

# Run with the default LM configured with `dspy.configure` above.
response = qa(question="How many floors are in the castle David Gregory inherited?")
print(response.answer)
```
**Possible Output:**
```text
The castle David Gregory inherited has 7 floors.
```

## Using multiple LMs.

You can change the default LM globally with `dspy.configure` or change it inside a block of code with `dspy.context`.

!!! tip
    Using `dspy.configure` and `dspy.context` is thread-safe!


```python linenums="1" 
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))
response = qa(question="How many floors are in the castle David Gregory inherited?")
print('GPT-4o-mini:', response.answer)

with dspy.context(lm=dspy.LM('openai/gpt-3.5-turbo')):
    response = qa(question="How many floors are in the castle David Gregory inherited?")
    print('GPT-3.5-turbo:', response.answer)
```
**Possible Output:**
```text
GPT-4o-mini: The number of floors in the castle David Gregory inherited cannot be determined with the information provided.
GPT-3.5-turbo: The castle David Gregory inherited has 7 floors.
```

## Configuring LM generation.

For any LM, you can configure any of the following attributes at initialization or in each subsequent call.

```python linenums="1" 
gpt_4o_mini = dspy.LM('openai/gpt-4o-mini', temperature=0.9, max_tokens=3000, stop=None, cache=False)
```

By default LMs in DSPy are cached. If you repeat the same call, you will get the same outputs. But you can turn off caching by setting `cache=False`.

If you want to keep caching enabled but force a new request (for example, to obtain diverse outputs),
pass a unique `rollout_id` and set a non-zero `temperature` in your call. DSPy hashes both the inputs
and the `rollout_id` when looking up a cache entry, so different values force a new LM request while
still caching future calls with the same inputs and `rollout_id`. The ID is also recorded in
`lm.history`, which makes it easy to track or compare different rollouts during experiments. Changing
only the `rollout_id` while keeping `temperature=0` will not affect the LM's output.

```python linenums="1"
lm("Say this is a test!", rollout_id=1, temperature=1.0)
```

You can pass these LM kwargs directly to DSPy modules as well. Supplying them at
initialization sets the defaults for every call:

```python linenums="1"
predict = dspy.Predict("question -> answer", rollout_id=1, temperature=1.0)
```

To override them for a single invocation, provide a ``config`` dictionary when
calling the module:

```python linenums="1"
predict = dspy.Predict("question -> answer")
predict(question="What is 1 + 52?", config={"rollout_id": 5, "temperature": 1.0})
```

In both cases, ``rollout_id`` is forwarded to the underlying LM, affects
its caching behavior, and is stored alongside each response so you can
replay or analyze specific rollouts later.


## Inspecting output and usage metadata.

Every LM object maintains the history of its interactions, including inputs, outputs, token usage (and $$$ cost), and metadata.

```python linenums="1" 
len(lm.history)  # e.g., 3 calls to the LM

lm.history[-1].keys()  # access the last call to the LM, with all metadata
```

**Output:**
```text
dict_keys(['prompt', 'messages', 'kwargs', 'response', 'outputs', 'usage', 'cost', 'timestamp', 'uuid', 'model', 'response_model', 'model_type])
```

## Using the Responses API

OpenAI recommends the [Responses API](https://platform.openai.com/docs/api-reference/responses) for new OpenAI projects, especially for reasoning models, built-in tools, multimodal inputs, and agentic workflows. Chat Completions remains broadly supported and is still the most common OpenAI-compatible shape across third-party providers.

Use the concrete Responses backend when you want to force `/responses`:

```python linenums="1"
import dspy

lm = dspy.OpenAIResponsesLM(
    "openai/gpt-5-mini",
    temperature=1.0,
    max_tokens=16000,
)
dspy.configure(lm=lm)
```

For an OpenAI-compatible provider that exposes `/responses`, keep the provider's base URL and choose `OpenAIResponsesLM`:

```python linenums="1"
import os
import dspy

lm = dspy.OpenAIResponsesLM(
    "openai/gpt-oss-120b",
    api_base="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)
dspy.configure(lm=lm)
```

Use `OpenAIChatLM` when the provider supports `/chat/completions` but not `/responses`, or when maximum compatibility matters. Use `OpenAITextLM` for legacy `/completions` models.

!!! note "Legacy `model_type`"
    The legacy LiteLLM-backed `dspy.LM` path accepts `model_type="responses"`. In the normalized LM path, prefer `dspy.LM(...)` for automatic routing or instantiate `OpenAIResponsesLM` directly for exact control.


## Advanced: Building custom LMs and writing your own Adapters.

Though rarely needed, new custom LMs should inherit from `dspy.BaseLM` and implement `forward(request: dspy.LMRequest) -> dspy.LMResponse`. This gives your LM the normalized request, response, usage, cost, history, caching, and streaming contracts used by DSPy's built-in normalized backends. Inherit from `dspy.BaseLM` only when maintaining an existing legacy prompt/messages LM. Another advanced layer in the DSPy ecosystem is that of _adapters_, which sit between DSPy signatures and LMs. A future version of this guide will discuss these advanced features, though you likely don't need them.

