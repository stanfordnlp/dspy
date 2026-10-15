# The DSPy LM contract (3.5)

DSPy's LM layer speaks one language: the lm15 objects bundled inside DSPy.
Import them from `dspy.lm15`; no separate installation is needed. They are the
original lm15 classes, not DSPy wrappers or subclasses.

## What changed, in one table

**3.4 was the transition release; 3.5 removed the old LM integration interfaces.**

| Interface | DSPy 3.4 | DSPy 3.5 |
| --- | --- | --- |
| `lm("Hello")` | List-returning convenience | Kept: one user message with the LM's defaults, returns a list |
| `lm(Request(...))` / `await lm.acall(Request(...))` | Returns an lm15 `Response` | The adapter and integration contract; one call per candidate |
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
  Provider-specific options lm15 has no mapping for (for example
  `vertex_project`) ride in `Config.extensions` and select the LiteLLM
  compatibility engine **before execution**.
- `engine="lm15"`: require the native backend. Options it cannot carry raise.
- `engine="litellm"`: explicitly use the compatibility backend.

Authentication failures, timeouts and provider errors never cause a switch to
another backend. Text completions and client settings not implemented by the
native integration remain on LiteLLM. The `timeout` setting, a number of seconds
or an `httpx.Timeout`, stays native: it bounds the wait for the next byte of the
reply (headers first, then each streamed chunk), the send, and a free
connection; the native default is 600 seconds, the same as LiteLLM's. An
`httpx.Timeout` component set to `None` means "wait forever", which the native
engine cannot honour: under `engine="auto"` that call uses LiteLLM, and under
`engine="lm15"` it raises `LMUnsupportedFeatureError` (feature `timeout`)
rather than silently substituting a default.

Text that no provider can receive (a lone surrogate character) raises a plain
`ValueError` before any engine is chosen, on every engine setting, so it is
never reported as an unexpected engine failure or retried.

Typed Requests through the LiteLLM engine forward `top_k` as a LiteLLM
argument on the Chat Completions path (LiteLLM translates it for providers
that take it); on the Responses path, which has no such field, it is dropped
and recorded under `"adaptations"`.

A setting the native route cannot carry exactly as written is **adapted and
recorded**, rather than silently ignored or unnecessarily refused (lm15 MAP-13): `seed` on Anthropic is
left out, `temperature=1.5` on Anthropic becomes `1.0`, a thinking-summary
level the wire lacks becomes `auto`. Each history entry carries the record
under `"adaptations"` (`field`, `action`, `asked`, `applied`, `reason`); the
lm15 `Response` in the entry carries the same. Nothing is printed. What lm15
still refuses — an image inside a tool result on the Chat Completions wire, a
stored cache object that does not exist on the provider — is known before any
request is sent and raises `LMUnsupportedFeatureError`, naming the field in
`error.feature`. All engines take `Request`; `engine="auto"` does not fall back
to old dictionary content for an unsupported native request. DSPy implements
`n > 1` by making separate requests, not by asking the native engine for `n`.

For stop sequences on the Responses API, the engine matters:

- **Native lm15:** streams internally and closes at the stop. Final usage is
  unavailable after an early cut; closing does not guarantee billing stops.
- **Typed Requests through LiteLLM:** trims the completed reply using lm15's
  stop helper. Generation is not stopped early; full provider usage is retained.
  This is recorded in `response.adaptations` and history. Typed Responses
  streaming through LiteLLM remains unsupported; use native lm15 for it.

Both preserve scores for whole retained tokens. If a stop cuts through a token,
`response.logprobs_complete` is false: the retained partial token has no invented
score. The flag survives saved-response caching. Chat Completions still sends
stop sequences to the provider directly.

### Caching, retries, usage and streaming

DSPy owns response caching, retries, candidate fan-out, callbacks and history.
Every call is cached by its lm15 `Request` (plus `rollout_id` and `n`). Cache
entries written by DSPy 3.4's ordinary calls used the old dictionary key and are
**not** read anymore: after upgrading, previously cached answers are generated
again once. Nothing errors; the cost is one regeneration per cached prompt.

Adapters produce `n` answers with `n` separate `lm(request)` calls in sequence
(`await lm.acall(request)` for async). Each candidate has its own cache entry,
callbacks, history and usage accounting. A single candidate keeps the caller's
`rollout_id`; multiple candidates derive stable, separate identities from that
rollout and the candidate index. The user's sampling `seed` is unchanged.
Repeated predictions reuse the corresponding candidates, while adjacent caller
rollouts do not overlap. A later failure or cancellation leaves earlier completed
calls recorded and cached.

`num_retries` counts additional attempts per call, with exponential delays capped
at 60 seconds unless the provider specifies `retry_after`. A stream is not retried
once a chunk has been sent to the caller.

`dspy.streamify` keeps its listener-facing chunks (`chunk.choices[0].delta`),
built from lm15 stream events. Streaming through the LiteLLM engine now uses the
same event mapping as native streaming, so every chunk has the same shape.

Token usage is recorded once per public call, and cache hits add no billed usage.
For adapter calls, `lm.history[-1]` holds the `Request` under `"request"`, one
`Response` under `"response"`, and a display copy of the messages under
`"messages"`. The string convenience `lm("text", n=...)` retains its existing
grouped history and list-returning behavior.

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
candidates, use `dspy.Predict("question -> answer", n=3)` and read
`prediction.completions`. Direct `lm(request)` calls always return one response;
they do not accept `n` or `num_generations`, or inherit the LM's candidate count.

## Errors and retry ownership

**Engines report failures; DSPy owns retry and fallback policy.** New engines raise
specific errors from `dspy.lm15`, the same bundled vocabulary as their requests and
responses. Use these imports, not a separately installed lm15 package whose Python
classes have different identities.

```python
from dspy.lm15 import RateLimitError

# In an engine, after recognizing the backend's actual rate-limit response:
raise RateLimitError("Provider rate limit", provider="my-backend", retry_after=2.0)
```

Do not return an error dictionary or a fake successful `Response`. LiteLLM engines
translate their SDK's exception classes and documented provider codes into lm15
errors. Native engines propagate lm15 errors. Arbitrary custom exceptions are not
classified by words such as "network" or "timeout".

At its owned engine/capability boundaries, DSPy translates canonical errors into its
existing public error family. Applications can keep catching `dspy.LMError`,
`dspy.LMAuthError`, `dspy.ContextWindowExceededError`, and their siblings. These
public error names are **not** part of the 3.5 interface removal. Already-public
DSPy errors, including those from legacy plugins, are preserved by identity.

- `LMLockTimeoutError` identifies local credential-lock contention, not a provider
  timeout or bad credential. It is transient and eligible for managed retries.
- `LMStreamAssemblyError` identifies an incomplete or invalid stream. It is not
  automatically retried; `partial` may hold salvageable content, not a successful
  result. Unknown usage stays unknown.
- `LMCollectionLimitError` identifies a local collection budget being reached.
  It is not retryable. `partial_events`, `rejected_event`, `limit`, `maximum`,
  `retained_bytes`, and `retained_events` survive the engine boundary. Reading
  `partial` assembles the incomplete turn on demand; wrapping the error does
  not allocate combined text/audio. DSPy does not add a live-session API here.
- `LMUnsupportedFeatureError.feature` preserves the unsupported setting path
  reported by lm15. It also fills `features` when no explicit list was provided.
- Unknown engine failures become `LMUnexpectedError`, with the original exception
  as `__cause__`. The canonical cause retains its exact lm15 code and SDK cause;
  the public error retains useful request IDs, provider codes, retry hints, routing
  diagnostics, and partial responses where available.
- Wrong Python API arguments still raise `TypeError`/`ValueError` before execution.
  Missing dependencies retain `ImportError`. Cancellation, keyboard interrupts,
  and warnings promoted to errors are not converted into retryable LM failures.

Only an engine attempt is retryable. Pricing, cache writes, history, and callbacks
must not cause a completed generation to run again. Reported usage for completed
candidates is retained when later work fails. Invalid or non-finite retry hints use
normal backoff; valid provider hints are honored, including HTTP-date headers.
A network retry can still repeat a request the provider already processed or billed:
this is **not an exactly-once guarantee**. Use `num_retries=0` when replay is unsafe.

All engine streams pass through the same guard: one leading start, a final end,
and no events after completion. Raised errors and canonical `StreamErrorEvent`s
both fail the call. Incomplete results are never cached as success. Cleanup errors
do not replace an active failure or cancellation; secondary diagnostics are kept in
`cleanup_errors` where the exception supports them.

Adapter fallback is separate from generation retry. `ChatAdapter` may make one
additional JSON-format call for an `AdapterParseError`, but never after visible
stream output. Unexpected parser bugs and engine/setup failures propagate. JSON
schema fallback is decided **before** the model call, not after a failed execution.

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

A custom engine owns its connection: `api_key`, `api_base`, `timeout`,
`extra_headers` and the other client settings are refused on `dspy.LM(engine=...)`,
on `copy()` and on every call, rather than dropped. `copy(engine=...)` replaces the
engine pair, and each side must be its kind (a coroutine function on the async side).
An engine is saved by `dump_state()`/`load_state()` of its own, and loading one is
gated by `allow_unsafe_lm_state=True` like a custom LM class. For an HTTP provider
lm15 can speak to, declare it with `dspy.lm15.register_provider(...)` instead of
writing an engine: each `dspy.LM` binds the registrations present at construction,
and a fallback to LiteLLM keeps the declared address and credential. See the
[custom-engine tutorial](../tutorials/custom_lm_engines/index.md).

An engine may declare `supports_function_calling`, `supports_reasoning`,
`supports_response_schema` and `supported_params`; adapters read those to decide
whether to use native tool calling, native reasoning, or structured outputs.

If you subclass `dspy.BaseLM` for extra persistent state, pass your engine to
`super().__init__(model, engine=..., async_engine=...)`. There is no `forward()`
to override; a `BaseLM` without an engine raises when called.

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
`lm(request)` separately for each candidate, and parses each `Response`. Custom
LM wrappers continue to intercept adapter calls through `__call__` and `acall`.

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
  `model_type="responses"` on OpenAI, or a Gemini model, for audio and files;
  Anthropic supports documents but not audio. Carrying OpenAI's documented `input_audio` and `file`
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
