# DSPy 3.5 LM migration

DSPy's LM layer now uses the lm15 objects bundled inside DSPy. Import them from
`dspy.lm15`; no separate installation is needed. They are the original lm15
classes, not DSPy wrappers or subclasses.

## The 3.5 cutoff

**3.4 was the transition release; 3.5 removed the old LM integration interfaces.**

| Interface | DSPy 3.4 | DSPy 3.5 |
| --- | --- | --- |
| `lm("Hello")` | List-returning convenience | Kept as a convenience over the canonical engine path |
| `lm(Request(...))` / `await lm.acall(Request(...))` | Returns an lm15 `Response` | The adapter and integration contract |
| OpenAI-style `lm(messages=[{"role": ..., "content": ...}])` | Deprecated; use an explicit lm15 request | Removed; raises `TypeError` with this guide's link |
| Custom `BaseLM.forward()` / `aforward()` implementations | Deprecated; migrate to an engine | Removed as an integration interface |
| `LegacyEngine` / `AsyncLegacyEngine` wrappers | Deprecated transition tools | Removed |
| Custom-engine `complete_legacy()` shortcuts | Deprecated | Removed |
| `Adapter.format()` | Returned OpenAI message dictionaries | Returns a `Prompt` of lm15 messages |
| `dspy.Type.format()` | Returned OpenAI content blocks | Returns lm15 content parts (or a string) |
| `Type.parse_lm_response(...)` | Took an OpenAI-shaped output dictionary | Takes the lm15 `Response` |
| `Tool.format_as_litellm_function_call()` | OpenAI function-tool dictionary | Renamed `Tool.as_function_tool()`, returns an lm15 `FunctionTool` |

In 3.5, DSPy's adapters build lm15 requests and consume lm15 responses directly,
not the list-returning convenience path. Provider wire formats belong to engines.
DSPy's built-in LM engines prefer native lm15. LiteLLM is no longer a required
dependency, but its compatibility engine remains available with
`pip install "dspy[litellm]"`. Native imports and calls do not require LiteLLM;
attempting to use compatibility without it raises an actionable installation error.
The `Request`/`Response` contract and one-call-per-candidate execution remain unchanged.

### Migrating from LiteLLM-backed DSPy

- Install `dspy[litellm]` for compatibility routes. `engine="auto"` prefers native
  lm15 and may select LiteLLM for unsupported routes, settings or representations
  **before provider I/O**. `engine="lm15"` is strict native;
  `engine="litellm"` explicitly selects compatibility. Provider failures never
  trigger a backend switch.
- `model_type="text"` uses the optional LiteLLM compatibility engine, not native
  lm15. The public call still uses a string or canonical `Request`, not `messages=`.
- Provider prefixes, credentials and options supported only by LiteLLM do not
  become native integrations automatically. Use the optional compatibility engine
  where supported. For an unlisted compatible HTTP
  endpoint, [register a provider](../tutorials/custom_lm_engines/index.md#declaring-a-provider-instead-of-writing-an-engine).
  A custom engine remains an option for other protocols or authentication flows.
- For native calls, replace `cache_control_injection_points` with the
  [`prompt_cache` policy](../tutorials/cache/index.md#using-provider-side-prompt-caching).
  LiteLLM's provider-shaped caching options remain compatibility-engine options;
  they do not establish native support.
- `dspy.enable_litellm_logging()` and `dspy.disable_litellm_logging()` remain
  available for the optional compatibility integration. `dspy.enable_logging()` /
  `disable_logging()` control DSPy's own logging.
- Hosted `dspy.Embedder` calls now use the OpenAI SDK against OpenAI, Azure
  OpenAI, or an OpenAI-compatible embeddings endpoint. Other providers require a
  callable; their old LiteLLM model prefixes no longer route automatically.
  Installing `dspy[litellm]` does not restore embedding fallback.
  See [Embedder](../api/models/Embedder.md).

For OpenRouter cross-model fallback, move `models` and `route` out of the SDK's
`extra_body` wrapper into generation options:

```python
lm = dspy.LM(
    "openrouter/openai/gpt-4.1-mini",
    models=["openai/gpt-4.1-mini", "anthropic/claude-haiku-4.5"],
    route="fallback",
)
```

DSPy sends this policy to OpenRouter; it does not implement the fallback itself.
An invalid model ID can still be rejected by OpenRouter before fallback applies.
These options are accepted only for the OpenRouter route. For OpenRouter privacy
and provider restrictions, use predictor configuration, such as
`dspy.Predict("question -> answer", provider={"data_collection": "deny"})`.
The LM constructor's `provider=` argument selects a training/launch provider,
not OpenRouter's inference routing policy.

DSPy still uses **LiteLLM-derived model metadata** for capability and pricing
hints, without importing the LiteLLM library. On first use it attempts to fetch
the upstream catalog, falling back to the bundled snapshot. Set
`LITELLM_LOCAL_MODEL_COST_MAP=True` to use only the snapshot;
`LITELLM_MODEL_COST_MAP_URL` still overrides the catalog URL. These legacy
environment names remain supported. A catalog entry does not establish a
supported native route, and missing pricing is unknown, not zero cost.

When preparing an upgrade from 3.4, expose its migration warnings with:

```sh
python -W default::DeprecationWarning your_program.py
```

In 3.4, legacy custom LMs warn even through program calls; ordinary calls through
built-in adapters do not warn about DSPy's own internal dictionary boundary.

Signature syntax is untouched: `'context, image: dspy.Image, question -> reasoning: dspy.Reasoning, answer: str'`
still declares the same fields. See [behaviour changes](#behaviour-changes-worth-knowing)
for modality limitations.

## Ordinary programs

Keep using ordinary calls and DSPy modules:

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
outputs = lm("Hello")
```

Ordinary calls return lists of strings or dictionaries. `experimental=True` no
longer changes that return type; it still controls other experimental features.
For multi-turn calls, use [explicit requests](#explicit-requests) rather than
OpenAI-style message dictionaries. A string call sends one user message using
the LM's defaults. DSPy modules such as `dspy.ChainOfThought("question -> answer")`
keep their usual interface.

`lm("Hello", temperature=0.7, n=2)` still accepts generation options in the
OpenAI vocabulary (`temperature`, `max_tokens`, `stop`, `reasoning_effort`,
`response_format`, ...). Set predictor defaults with
`predict = dspy.Predict("question -> answer", temperature=0.7)`; override them
for one call with `predict(question="Why?", config={"temperature": 0.2})`.
Those options are read with lm15's own Chat Completions reader into
`Request.config`; DSPy keeps no second copy of that vocabulary.

Engine selection:

- `engine="auto"` (default): prefer native lm15 for supported routes and
  representable inputs. Unsupported routes and ordinary provider-specific inputs
  that cannot be represented faithfully may select optional LiteLLM **before
  provider I/O**. Unmapped options such as `vertex_project` travel in `Config.extensions`.
- `engine="lm15"`: require the native backend. Unsupported mappings raise.
- `engine="litellm"`: explicitly use the optional compatibility backend.

Authentication failures, timeouts and provider errors never cause a switch to
another backend. Compatibility use requires `pip install "dspy[litellm]"`;
native use does not.

The `timeout` setting, a number of seconds or an `httpx.Timeout`, bounds the wait
for the next byte of the reply (headers first, then each streamed chunk), the
send, and a free connection; the native default is 600 seconds. An
`httpx.Timeout` component set to `None` means "wait forever", which the native
engine cannot honour: under `engine="auto"` that call uses optional LiteLLM, and
under `engine="lm15"` it raises `LMUnsupportedFeatureError` (feature `timeout`)
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
still refuses is known before any request is sent. Under `engine="auto"`, a
representational refusal may select optional LiteLLM before inference, so Chat
audio and file inputs can use its SDK bridge. Explicit native selection,
declared-provider refusals and provider prompt-cache requests do not take this
fallback; unsupported inputs raise `LMUnsupportedFeatureError`. Compatibility is
not evidence of native modality parity. All engines still take `Request`,
including the compatibility engine. DSPy implements
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
Every call is cached by its lm15 `Request`, model type, `rollout_id` and `n`.
Sync and async calls keep separate cache entries, so one engine's answer cannot
replace its counterpart's answer in a custom engine pair. Cache
entries written by DSPy 3.4's ordinary calls used the old dictionary key and are
**not** read anymore: after upgrading, previously cached answers are generated
again once. Nothing errors; the cost is one regeneration per cached prompt.

Adapters produce `n` answers with `n` separate `lm(request)` calls in sequence
(`await lm.acall(request)` for async). Each candidate has its own cache entry,
callbacks, history and usage accounting. A single candidate keeps the caller's
`rollout_id`; multiple candidates derive stable, separate identities from that
rollout and the candidate index. For both `Predict` and convenience calls such as
`lm("Hello", n=3)`, multiple candidates use an explicit sampling `seed` as
`seed + candidate_index` (zero-based): the first seed is unchanged.
`None` remains `None`, and `n=1` leaves the seed unchanged. Provider determinism
is best effort; distinct seeds do not guarantee distinct outputs.
Repeated predictions reuse the corresponding candidates, while adjacent caller
rollouts do not overlap. A later failure or cancellation leaves earlier completed
calls recorded and cached.

Separate requests can take longer and bill input tokens more than once.

`num_retries` counts additional attempts, with exponential delays of 1, 2, 4…
seconds, capped at 60 seconds unless the provider specifies `retry_after`.
Backend retries are disabled on the managed LiteLLM path. A stream is not retried
once a chunk has been sent to the caller. Caching a completed stream does not
replay fake token timing on a later ordinary cache hit.

`dspy.streamify` keeps its listener-facing chunks (`chunk.choices[0].delta`),
built from lm15 stream events. LiteLLM Chat streaming uses the same event mapping
as native streaming. Predictions and direct calls to LiteLLM text and Responses
models inside a streamified module complete without token chunks. Other calls
in the module can still stream. Token streaming on these compatibility endpoints
remains unsupported; use native lm15 for Responses token streaming. Custom
engines must implement the canonical streaming interface to produce token chunks.

`streamify` is the **program-level** API: it selects predictor fields, emits tool
and module status messages, and delivers the final `Prediction`. lm15 events describe
**one model call**, so they do not replace those responsibilities.

Token usage is recorded once per public call, and cache hits add no billed usage.
Native response counters stay provider-verbatim in `Response.usage`; DSPy's
legacy prompt/completion counters are derived separately. A native cost estimate
is **unknown**, not zero, when no reliable pricing is available.
For adapter calls, `lm.history[-1]` holds the `Request` under `"request"`, one
`Response` under `"response"`, and a display copy of the messages under
`"messages"`. The string convenience `lm("text", n=...)` retains its existing
grouped history and list-returning behavior.

## Migrating OpenAI-style messages

The public `messages=` argument is removed, including provider SDK message
objects as well as dictionaries. This does not remove `Request.messages`, which
holds canonical lm15 `Message` objects. This form ran in 3.4 but no longer runs:

```python
outputs = lm(messages=[
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "What is DSPy?"},
])
```

Use the explicit request below instead. Put system instructions in `Request.system`,
conversation turns in lm15 `Message` objects, and generation options in `Config`.
The result is a `Response`, not a list: read `response.text`, `response.message`, or
`response.tool_calls` as appropriate. Async calls use the same types.

## Explicit requests

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

The request's model must match the LM. Generation options belong in its config;
LM generation defaults are not added. Client configuration still comes from the
LM. `lm(request, cache=False, rollout_id=...)` controls DSPy's response cache.
`Config.cache` instead controls provider-side prompt caching.

Only these execution controls and client settings (`api_key`, `api_base`, ...)
are accepted alongside a typed request. To continue a conversation, add
`response.message` to the next request's messages. For several
candidates, use `dspy.Predict("question -> answer", n=3)` and read
`prediction.completions`. Direct `lm(request)` calls always return one response;
they do not accept `n` or `num_generations`, or inherit the LM's candidate count.

## Errors and retry ownership

**Engines report failures; DSPy owns retries.** New engines raise
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
errors. Native engines propagate lm15 errors; custom engines should translate
their backend's documented failures into those same errors. Arbitrary custom exceptions are not
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

Protocols live in `dspy.clients.engines`; inheritance is not required:

- `Engine`: `complete(request: Request) -> Response`.
- `AsyncEngine`: `async def complete(request: Request) -> Response`.
- `StreamingEngine`: `Engine` plus `stream(request) -> Iterator[StreamEvent]`.
- `AsyncStreamingEngine`: `AsyncEngine` plus `stream(request) -> AsyncIterator[StreamEvent]`.

Streams yield canonical lm15 events; async iterators are consumed directly, not
awaited first. Completion-only engines work for ordinary calls; requesting streaming
from one raises `dspy.LMUnsupportedFeatureError`. Cleanup methods are optional.
Custom engines are caller-owned: arrange their cleanup yourself. DSPy closes
returned stream iterators with `close` / `aclose`, including on failure or
cancellation, but not the borrowed engines. Engines must not add another DSPy
cache or retry loop.

An async-only custom LM uses `dspy.LM(model, engine=None, async_engine=...)`.
Each call mode requires its own engine: `lm(...)` does not run an async engine,
and `await lm.acall(...)` does not run a sync engine.

A custom engine owns its connection: `api_key`, `api_base`, `timeout`,
`extra_headers` and the other client settings are refused on `dspy.LM(engine=...)`,
on `copy()` and on every call, rather than dropped. `copy(engine=...)` replaces the
engine pair, and each side must be its kind (a coroutine function on the async side).
An engine is saved by `dump_state()`/`load_state()` of its own, and loading one is
gated by `allow_unsafe_lm_state=True` like a custom LM class. For an HTTP provider
lm15 can speak to, declare it with `dspy.lm15.register_provider(...)` instead of
writing an engine: each `dspy.LM` binds the registrations present at construction,
and any permitted compatibility fallback keeps the declared address and credential. See the
[custom-engine tutorial](../tutorials/custom_lm_engines/index.md).

An engine may declare `supports_function_calling`, `supports_reasoning`,
`supports_response_schema` and `supported_params`; adapters read those to decide
whether to use native tool calling, native reasoning, or structured outputs.
The sync engine supplies these hints when present; otherwise the async engine
does. Hints are not combined across engines, so a pair should support the
features declared by its sync engine. Missing hints mean unsupported.

If you subclass `dspy.BaseLM` for extra persistent state, pass your engine to
`super().__init__(model, engine=..., async_engine=...)`. There is no `forward()`
to override; a `BaseLM` without an engine raises when called.

`DummyLM` now uses a canonical engine but retains its scripted answer modes,
reasoning option, adapter formatting, and uncached consumption of repeated calls.

### Copying, saving and cleanup

`lm.copy()` isolates DSPy's history, callbacks and kwargs while sharing runtime
resources. A model or client-setting change selects a distinct native pool.
Native async pools are separate per event loop. Close owned sync pools with
`lm.close()` and this loop's async pools with `await lm.aclose()` after calls finish.
Copies share ownership: closing one releases the shared pools, which can be
recreated on subsequent calls.

JSON LM state preserves named engine selections and excludes API keys. Arbitrary
custom engines require custom `dump_state`/`load_state` methods instead of guessed
reconstruction. Whole-program pickle saving excludes native transport pools and
locks; they are recreated lazily after loading. Existing trusted-loading safeguards
remain in force.

## Custom adapters and custom types

`Adapter.format()` returns a `dspy.adapters.Prompt`, not provider dictionaries.
Custom `Type.format()` methods return text or lm15 content parts. Inspect both
boundaries together without calling an LM or downloading the image:

```python
import dspy
from dspy.adapters import Prompt

prompt = dspy.ChatAdapter().format(
    dspy.Signature("image: dspy.Image -> answer"),
    demos=[], inputs={"image": dspy.Image("https://example.org/photo.jpg")},
)
prompt = Prompt(system=prompt.system + "\nAnswer in French.", messages=prompt.messages)
print(prompt.system)
print(prompt.messages[0].parts)  # TextPart markers surround an ImagePart.
```

In an overridden `format`, return the modified `Prompt` shown above. The
[adapter reference](../api/adapters/Adapter.md#dspy.Adapter.format) documents the
message-list demo/history hooks, text content hooks and `build_request` boundary.
Custom LM wrappers still intercept adapter calls through `__call__` and `acall`.

Use [Image.format](../api/primitives/Image.md#dspy.Image.format) as a working
custom-type template: for example, return `[ImagePart(url=self.url, media_type="image/jpeg")]`
from your method, importing `ImagePart` from `dspy.lm15`. OpenAI-style dictionaries
raise `dspy.adapters.types.base_type.TypeFormatError`, naming the type.
`parse_lm_response(response)` now receives the lm15 `Response`; read
`response.message.parts_of(ThinkingPart)` or `response.citations`.

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
  when a provider returns `CitationPart`s anyway, those win. Restoring native
  Anthropic document citations requires upstream opt-in support; it is not implemented.
- **`Citations.Citation` fields other than `cited_text` are optional.** lm15
  citations carry text, title and URL, but native document indexes and character
  offsets are not preserved, even if the provider reports them. Prompt-parsed
  citations may include these fields; they are model output, not native provenance.
- **Audio and files on Chat Completions endpoints.** The optional LiteLLM engine
  preserves `input_audio` and `file` blocks for `dspy.Audio`, inline `dspy.File`
  data and uploaded file IDs. This is compatibility support, not native parity.
  Native lm15 Chat carries text and images only and, with `engine="lm15"`, raises
  `LMUnsupportedFeatureError` for these inputs. OpenAI Responses supports files,
  **not audio**: switching `gpt-audio-mini` to `model_type="responses"` does not
  fix audio support. Gemini supports native audio and files. Anthropic supports
  documents but not audio. Compatibility text-completion endpoints reject image,
  audio and file inputs rather than silently dropping them.
- **Strict function tools.** The bundled lm15 `FunctionTool` cannot represent
  per-tool `strict=True`. Both Chat Completions and flat Responses tool
  dictionaries with this setting raise `LMUnsupportedFeatureError` rather than
  silently dropping enforcement. `strict=False` remains accepted. This does
  not affect strict structured outputs through `response_format`.
- **Reasoning summaries.** When a Responses-API model runs with native
  reasoning, DSPy asks for `summary="auto"` so `dspy.Reasoning` fields can be
  filled from the response. `reasoning={"enabled": False}` and
  `reasoning={"effort": "none"}` normalize to `Reasoning(effort="off")`;
  explicitly disabled reasoning does not request a summary.

## Breaking replacement of the experimental 3.3 types

The old `dspy.core.types` import raises a migration error. Old type names are no
longer exported from `dspy` or `dspy.core`. This is not a drop-in rename:

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

lm15 uses frozen dataclasses, not Pydantic models. Required identities, roles,
nonempty messages, and media-source validation differ. File names, document
citation settings and arbitrary part metadata do not all have direct canonical
equivalents; see [behaviour changes](#behaviour-changes-worth-knowing).

Old pickled objects containing the removed classes are not automatically migrated.
Load and export them in their original environment first. This is separate from
ordinary provider-response caches and ordinary saved-program configuration.

Custom adapters must use `lm(request)` / `await lm.acall(request)` and parse the
lm15 `Response`, not OpenAI-style dictionaries or list-returning convenience calls.
The removed `dspy.clients.openai_format` module and temporary internal warning
marker are not migration APIs.

DSPy's signature types (`Image`, `Audio`, `File`, `Tool`, `ToolCalls`, `History`,
`Reasoning`, etc.) and error classes are not removed.
