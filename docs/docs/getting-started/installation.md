# Setting up DSPy

Install DSPy with `pip install dspy` or run `uv add dspy` to add DSPy to a virtual environment. DSPy works in Python 3.10+ environments.

## Connecting to a language model

DSPy connects to language models with the `dspy.LM` class. To set up a language model, we provide a `"provider/model"` format string and an API key:

```py
import dspy

# Pass the key explicitly...
lm = dspy.LM("openai/gpt-5-nano", api_key="YOUR_OPENAI_API_KEY")

dspy.configure(lm=lm)
```

Behind the scenes, DSPy uses its bundled native lm15 engines to translate a common request/response contract into provider protocols. No separate lm15 installation is needed. Both `engine="auto"` (the default) and `engine="lm15"` use this native backend, without a compatibility fallback.

You can replace the model and credentials with a supported native provider such as Anthropic or Gemini. Model features and authentication differ by provider; not every historical provider prefix is supported. See [other native providers](#other-native-providers) below and [custom providers and engines](../tutorials/custom_lm_engines/index.md) for other backends. If you are upgrading, review the [breaking changes](../community/normalized-lm-api-migration.md#migrating-from-litellm-backed-dspy).

Once we have an `LM`, calling `dspy.configure(lm=lm)` sets our `LM` as the default provider for every DSPy program in the process. This sets our `LM` globally, but we can selectively override this with [`dspy.context`](../diving-deeper/settings-and-context.md) when more granular control is needed.

Let’s ensure everything works by manually calling the model:

```py
print(lm("What is the capital of France?"))
```

Which returns:

```
['The capital of France is Paris.']
```

With an `LM` configured, we can proceed to writing our first program.

## Other native providers

Set the appropriate credentials before constructing your LM:

| Service | Example model | Configuration |
| --- | --- | --- |
| Anthropic | `anthropic/claude-sonnet-4-5-20250929` | `ANTHROPIC_API_KEY` or `api_key=` |
| Google AI Studio | `gemini/gemini-2.5-pro` | `GEMINI_API_KEY` or `api_key=` |
| Azure OpenAI | `azure/your-deployment-name` | `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` (resource root) |
| Vertex AI Gemini | `vertex:gemini-2.0-flash` | Application Default Credentials, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_LOCATION` |
| Local Ollama | `ollama_chat/llama3.2:1b` | Start Ollama and pull the model; the native route supplies its local address and placeholder key |

For example, after setting `ANTHROPIC_API_KEY`:

```python
dspy.configure(lm=dspy.LM("anthropic/claude-sonnet-4-5-20250929"))
```

Vertex uses its own `vertex:` route, not the old `vertex_ai/` prefix or
`vertex_project` / `vertex_location` kwargs. A service-account file can be selected
with `GOOGLE_APPLICATION_CREDENTIALS`. Set `GOOGLE_CLOUD_LOCATION` explicitly when
region matters; its default is `global`. Azure LM calls use the native v1 route,
not the old `api_version` configuration. Use `model_type="responses"` for a native
Responses endpoint; `model_type="text"` is unsupported.

For an OpenAI-compatible Chat Completions server using bearer authentication:

```python
lm = dspy.LM(
    "openai/your-model-name",
    api_base="https://your-provider.example/v1",
    api_key="YOUR_PROVIDER_KEY",
)
```

An unlisted service can be [registered](../tutorials/custom_lm_engines/index.md#declaring-a-provider-instead-of-writing-an-engine)
when it uses a supported wire dialect. Other protocols or authentication flows
need a custom engine. Unsupported settings raise instead of invoking another
backend. Embeddings have a separate [OpenAI SDK or callable interface](../api/models/Embedder.md).

---

**Next:** [Your first program →](first-program.md)
