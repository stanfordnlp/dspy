# Language Model Backends

DSPy routes every LM call through a **backend module** — a plain Python
module (or object) that knows how to talk to a specific provider.

DSPy ships with built-in backends for **OpenAI-compatible APIs** (including
vLLM, Ollama, Together, etc.), **Anthropic**, **Google GenAI**, and a
**litellm** catch-all.  We don't plan to add more provider-specific
backends to the core library.  Instead, providers and community members
can publish their own backends as standalone packages on
[dspy-community](https://github.com/dspy-community) and register them
via entry points (see below).

## The backend protocol

A backend must expose seven attributes / functions:

| Attribute | Type | Purpose |
|---|---|---|
| `ContextWindowError` | Exception class | The exception DSPy should catch to detect context-window overflows. |
| `supports_function_calling(model)` | `(str) → bool` | Does this model support tool / function calling? |
| `supports_reasoning(model)` | `(str) → bool` | Does this model expose chain-of-thought / reasoning tokens? |
| `supports_response_schema(model)` | `(str) → bool` | Does this model support structured `response_format` schemas? |
| `supported_params(model)` | `(str) → set[str]` | The set of request-parameter names the provider understands. |
| `complete_request(request, model_type, num_retries)` | sync function | Execute a completion and return an OpenAI-shaped `ChatCompletion`. |
| `acomplete_request(request, model_type, num_retries)` | async function | Async version of the above. |

### Optional: streaming

If your backend also supports streaming, add:

| Attribute | Type | Purpose |
|---|---|---|
| `astream_complete(request, num_retries)` | async function | Return an async iterator of `StreamChunk` objects. After exhaustion, the iterator's `.assembled` attribute must hold the reassembled `ChatCompletion`. |

DSPy only uses the streaming path when `dspy.settings.send_stream` is set
**and** the backend has an `astream_complete` attribute.

---

## Example

The best references are the built-in backends that translate a
non-OpenAI wire format into the `ChatCompletion` objects DSPy expects:

- **[`dspy/clients/_anthropic.py`](https://github.com/stanfordnlp/dspy/blob/main/dspy/clients/_anthropic.py)** —
  Anthropic Messages API.
- **[`dspy/clients/_google.py`](https://github.com/stanfordnlp/dspy/blob/main/dspy/clients/_google.py)** —
  Google GenAI (Gemini) `generateContent` API.

It uses shared retry and streaming helpers from
[`dspy/clients/_request_utils.py`](https://github.com/stanfordnlp/dspy/blob/main/dspy/clients/_request_utils.py)
(`call_with_retries`, `acall_with_retries`, `StreamChunk`), which you're
free to reuse in your own backend.

---

## Passing the backend directly

The simplest way to use a custom backend is the `backend=` parameter:

```python
import dspy
import my_backend

lm = dspy.LM("my-model-name", backend=my_backend)
dspy.configure(lm=lm)

# Use DSPy as usual — every call goes through my_backend.
predict = dspy.Predict("question -> answer")
predict(question="What is DSPy?")
```

The model string is passed as-is to your backend in `request["model"]`
and to your capability functions (`supports_function_calling(model)`,
etc.).  Since `backend=` bypasses automatic resolution, no prefix is
required — you can use whatever model string your API expects.

If you *do* add a prefix (e.g. `"myprovider/my-model-name"`), it shows
up in logs and `dspy.inspect_history()` which can be helpful for
debugging, but your backend will need to strip it before sending it to the
API (see `strip_prefix` in [`dspy/clients/_request_utils.py`](https://github.com/stanfordnlp/dspy/blob/main/dspy/clients/_request_utils.py)).

When `backend=` is provided, DSPy skips all automatic resolution (entry
points and built-in prefix matching) and uses the object you gave it
directly.

---

## Registering via entry points (for packages)

If you're distributing your backend as an installable package, you can
register it as a `dspy.backends` entry point so users don't need to pass
`backend=` manually.

In your package's `pyproject.toml`:

```toml
[project.entry-points."dspy.backends"]
myprovider = "my_package.dspy_backend"
```

Then any user who installs your package can simply write:

```python
lm = dspy.LM("myprovider/my-model-name")
```

DSPy will discover the entry point, load your module, validate it against
the required protocol, and use it automatically.  Entry-point backends take
priority over DSPy's built-in backends, so you can even override the
`openai` or `anthropic` prefix if needed.

---

## Key things to know

- **Return type**: `complete_request` and `acomplete_request` must return
  an `openai.types.chat.ChatCompletion` (or a duck-typed equivalent with
  the same attribute structure).  DSPy's post-processing reads
  `.choices[0].message.content`, `.usage`, etc.

- **`model_type`**: DSPy passes `"chat"` (most common), `"text"` (legacy
  completions), or `"responses"` (OpenAI Responses API). Your backend can
  raise `ValueError` for types it doesn't support.

- **`ContextWindowError`**: DSPy catches this specific exception to raise
  its own `ContextWindowExceededError`.  Make sure your backend raises it
  when the provider reports a context-length overflow.

- **Retries**: DSPy passes `num_retries` to your backend — it's your
  responsibility to implement retry logic (or use the helpers from
  `dspy.clients._request_utils`).

- **Caching**: DSPy handles caching at a layer above the backend. Your
  `complete_request` / `acomplete_request` functions will only be called
  on cache misses — no need to implement caching yourself.
