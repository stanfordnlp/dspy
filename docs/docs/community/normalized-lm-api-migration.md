# The DSPy LM contract (3.5)

DSPy's LM layer speaks one language: the lm15 objects bundled inside DSPy.
Import them from `dspy.lm15`; no separate installation is needed. They are the
original lm15 classes, not DSPy wrappers or subclasses.

## What changed, in one table

**3.4 was the transition release; 3.5 removed the old LM integration interfaces.**

| Interface | DSPy 3.4 | DSPy 3.5 |
| --- | --- | --- |
| `lm("Hello")` | List-returning convenience | Kept: one user message with the LM's defaults, returns a list |
| `lm(Request(...))` / `await lm.acall(Request(...))` | Returns an lm15 `Response` | The contract for one call |
| `lm.generate(Request(...), n=3)` | — | New: `n` responses for one request (what adapters use) |
| OpenAI-style `lm(messages=[{"role": ..., "content": ...}])` | Deprecated | **Removed**; raises `TypeError` with this guide's link |
| Custom `BaseLM.forward()` / `aforward()` implementations | Deprecated | **Removed**; custom LMs are engines |
| `LegacyEngine` / `AsyncLegacyEngine`, `complete_legacy()` | Deprecated transition tools | **Removed** |
| `Adapter.format()` | Returned OpenAI message dictionaries | Returns a `Prompt` of lm15 messages |
| `dspy.Type.format()` | Returned OpenAI content blocks | Returns lm15 content parts (or a string) |
| `Type.parse_lm_response(...)` | Took an OpenAI-shaped output dictionary | Takes the lm15 `Response` |
| `Tool.format_as_litellm_function_call()` | OpenAI function-tool dictionary | Renamed `Tool.as_function_tool()`, returns an lm15 `FunctionTool` |

Nothing OpenAI-shaped exists inside DSPy anymore. Provider wire formats belong
to engines: the native lm15 engines and the LiteLLM compatibility engine, which
still follows the same request/response contract. LiteLLM itself is **not**
removed.

Signatures are untouched. `'context, image: dspy.Image, question -> reasoning: dspy.Reasoning, answer: str'`
means exactly what it did before.

## Ordinary programs

Keep using ordinary calls and DSPy modules:

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
outputs = lm("Hello")               # ['Hi there!']
program = dspy.ChainOfThought("question -> answer")
```

`lm("Hello", temperature=0.7, n=2)` still accepts generation options in the
OpenAI vocabulary (`temperature`, `max_tokens`, `stop`, `reasoning_effort`,
`response_format`, ...), exactly as `dspy.LM(...)` and `dspy.Predict(config=...)`
always have. Those options are read with lm15's own Chat Completions reader into
`Request.config`; DSPy keeps no second copy of that vocabulary.

Engine selection is unchanged:

- `engine="auto"` (default): prefer native lm15 for supported routes.
  Provider-specific options lm15 has no mapping for (for example `prediction`
  or `vertex_project`) ride in `Config.extensions` and select the LiteLLM
  compatibility engine **before execution**.
- `engine="lm15"`: require the native backend. Options it cannot carry raise.
- `engine="litellm"`: explicitly use the compatibility backend.

Authentication failures, timeouts and provider errors never cause a switch to
another backend.

### Caching, retries, usage and streaming

DSPy owns response caching, retries, candidate fan-out, callbacks and history.
Every call is cached by its lm15 `Request` (plus `rollout_id` and `n`). Cache
entries written by DSPy 3.4's ordinary calls used the old dictionary key and are
**not** read anymore: after upgrading, previously cached answers are generated
again once. Nothing errors; the cost is one regeneration per cached prompt.

`n` answers are `n` separate requests in sequence. `num_retries` counts
additional attempts with exponential delays capped at 60 seconds unless the
provider specifies `retry_after`. A stream is not retried once a chunk has been
sent to the caller.

`dspy.streamify` keeps its listener-facing chunks (`chunk.choices[0].delta`),
built from lm15 stream events. Streaming through the LiteLLM engine now uses the
same event mapping as native streaming, so every chunk has the same shape.

Token usage is recorded once per public call, and cache hits add no billed usage.
`lm.history[-1]` holds the `Request` under `"request"`, the `Response` (or tuple
of responses for `n > 1`) under `"response"`, and a display copy of the messages
under `"messages"`.

## Migrating OpenAI-style messages

This form was deprecated in 3.4 and no longer runs:

```python
outputs = lm(messages=[
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "What is DSPy?"},
])
```

Build a request instead. System instructions go in `Request.system`, turns are
lm15 `Message` objects, generation options live in `Config`:

```python
from dspy.lm15 import Config, Message, Request

request = Request(
    model=lm.model,
    system="Be concise.",
    messages=(Message.user("What is DSPy?"),),
    config=Config(max_tokens=200),
)
response = lm(request)
print(response.text)
# Async equivalent: response = await lm.acall(request)
```

The request's model must match the LM. `lm(request)` accepts only DSPy's
execution controls (`cache`, `rollout_id`) and client settings (`api_key`,
`api_base`, ...); generation options belong in `Config`. To continue a
conversation, add `response.message` to the next request's messages. For several
candidates, `lm.generate(request, n=3)` returns a list of `Response`s.

## Custom LMs are engines

A custom backend implements the small engine interface and is passed to
`dspy.LM`:

```python
from dspy.lm15 import Message, Response, Usage

class EchoEngine:
    def complete(self, request):
        return Response(
            id=None, model=request.model, message=Message.assistant("hello"),
            finish_reason="stop", usage=Usage(),
        )

lm = dspy.LM("custom/echo", engine=EchoEngine())
```

Implement `stream(request)` yielding canonical lm15 events to support streaming.
Supply `async_engine=` with `async complete(request)` and `stream(request)`
returning an async iterator for async calls. Engines must not add another DSPy
cache or retry loop. Custom engines are caller-owned and are not closed by DSPy.

An engine may declare `supports_function_calling`, `supports_reasoning`,
`supports_response_schema` and `supported_params`; adapters read those to decide
whether to use native tool calling, native reasoning, or structured outputs.

If you subclass `dspy.BaseLM` for extra persistent state, pass your engine to
`super().__init__(model, engine=..., async_engine=...)`. There is no `forward()`
to override; a `BaseLM` without an engine raises when called.

### Errors and retry ownership

**Engines report failures; DSPy owns retry and fallback policy.** Raise specific
errors from `dspy.lm15` (`RateLimitError`, `AuthError`, ...). At its engine
boundary DSPy translates them into its public error family (`dspy.LMError`,
`dspy.LMAuthError`, `dspy.ContextWindowExceededError`, ...). Unknown failures
become `LMUnexpectedError` with the original exception as `__cause__`; they are
never retried on the strength of their message text.

## Custom adapters and custom types

Adapters render a `dspy.adapters.Prompt`:

```python
from dspy.adapters import Prompt
from dspy.lm15 import Message

class MyAdapter(dspy.ChatAdapter):
    def format(self, signature, demos, inputs):
        prompt = super().format(signature, demos, inputs)
        return Prompt(system=prompt.system + "\nAnswer in French.", messages=prompt.messages)
```

`format_demos` and `format_conversation_history` return lists of lm15
`Message`s; `format_user_message_content` and `format_assistant_message_content`
still return text. `Adapter.__call__` turns the prompt and `lm_kwargs` into one
`Request` with `dspy.clients.requests.build_request`, calls
`lm.generate(request, n=...)`, and parses each `Response`.

A custom `dspy.Type` returns lm15 content parts from `format()`:

```python
from dspy.lm15 import ImagePart

class Thumbnail(dspy.Type):
    url: str

    def format(self):
        return [ImagePart(url=self.url, media_type="image/jpeg")]
```

Returning OpenAI-style content dictionaries raises
`dspy.adapters.types.base_type.TypeFormatError`, naming the type. `parse_lm_response(response)` receives the lm15 `Response`; read
`response.message.parts_of(ThinkingPart)`, `response.citations`, and so on.

Fine-tuning data is still written in OpenAI's chat format because that is the
file format providers accept: `ChatAdapter.format_finetune_data` uses lm15's own
Chat Completions writer through `dspy.adapters.base.prompt_to_openai_messages`.

### Behaviour changes worth knowing

- **`dspy.File.filename` is not sent.** A file becomes one lm15 `DocumentPart`
  (inline data or `file_id`); engines derive any file name a provider needs
  from the media type. The name stays on the Python object for display.
- **`dspy.experimental.Document` no longer enables Anthropic's native citations.**
  lm15's `DocumentPart` has no citation opt-in yet. Plain-text documents are
  sent as text framed by their title and context; PDFs as a document part. The
  `Citations` output field stays in the prompt and is parsed from the answer;
  when a provider returns `CitationPart`s anyway, those win. Native Anthropic
  document citations return once lm15 carries the opt-in.
- **`Citations.Citation` fields other than `cited_text` are optional.** lm15
  citations carry text, title and URL; character offsets appear only when a
  provider reports them.
- **Audio and files on Chat Completions endpoints.** The bundled lm15's Chat
  Completions writer carries text and images only, so `dspy.Audio` and
  `dspy.File` on a chat model (native or through LiteLLM) raise
  `LMUnsupportedFeatureError` before any request is sent. Use
  `model_type="responses"` on OpenAI, or an Anthropic or Gemini model, which
  carry them natively. Carrying OpenAI's documented `input_audio` and `file`
  chat blocks is pending in lm15.
- **Reasoning summaries.** When a Responses-API model runs with native
  reasoning, DSPy asks for `summary="auto"` so `dspy.Reasoning` fields can be
  filled from the response.

## Breaking replacement of the experimental 3.3 types

The old `dspy.core.types` import raises a migration error. The 3.3 names map to
lm15 objects:

| Old experimental API | New API |
| --- | --- |
| `dspy.LMRequest` | `dspy.lm15.Request` |
| `dspy.LMResponse` | `dspy.lm15.Response` |
| `dspy.LMMessage`, `dspy.LMConfig` | `dspy.lm15.Message`, `dspy.lm15.Config` |
| `dspy.System(text)` | `Request(system=text, ...)` |
| `dspy.User(text)`, `dspy.Assistant(text)`, `dspy.Developer(text)` | `Message.user(text)`, `Message.assistant(text)`, `Message.developer(text)` |
| `dspy.ToolCall(id=..., name=..., args=...)` | `ToolCallPart(id=..., name=..., input=...)` |
| `dspy.ToolResult(content, call_id=...)` | `Message.tool(call_id, content)` |
| `LMTextPart`, `LMImagePart`, etc. | `TextPart`, `ImagePart`, etc., from `dspy.lm15` |
| `response.outputs[0].parts` | `response.message.parts` |

DSPy's signature types (`Image`, `Audio`, `File`, `Tool`, `ToolCalls`, `History`,
`Reasoning`, etc.) and error classes are not removed.
