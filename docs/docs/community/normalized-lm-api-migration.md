# DSPy 3.4 LM migration

DSPy's LM layer now uses the lm15 objects bundled inside DSPy. Import them from
`dspy.lm15`; no separate installation is needed. They are the original lm15
classes, not DSPy wrappers or subclasses.

## The 3.5 cutoff

**3.4 is the transition release; 3.5 removes the old LM integration interfaces.**

| Interface | DSPy 3.4 | DSPy 3.5 |
| --- | --- | --- |
| `lm("Hello")` | List-returning convenience | Kept as a convenience over the canonical engine path |
| `lm(Request(...))` / `await lm.acall(Request(...))` | Returns an lm15 `Response` | The adapter and integration contract |
| OpenAI-style `lm(messages=[{"role": ..., "content": ...}])` | Deprecated; use an explicit lm15 request | Removed |
| Custom `BaseLM.forward()` / `aforward()` implementations | Deprecated; migrate to an engine | Removed as an integration interface |
| `LegacyEngine` / `AsyncLegacyEngine` wrappers | Deprecated transition tools | Removed |
| Custom-engine `complete_legacy()` shortcuts | Deprecated | Removed |

In 3.5, DSPy's adapters will build lm15 requests and consume lm15 responses directly,
not use the list-returning convenience path. Provider-wire dictionaries belong
inside engines. LiteLLM itself is **not** deprecated: a LiteLLM engine must follow
that same request/response contract.

These changes do not happen silently in 3.4. Deprecated interfaces still execute;
`DeprecationWarning` announces the cutoff and links to the replacement. Python may
hide these warnings by default. To review them during development, use:

```sh
python -W default::DeprecationWarning your_program.py
```

Normal DSPy program calls do not warn merely because built-in adapters still use
an internal dictionary boundary in 3.4. That implementation is DSPy's migration
responsibility. Using a legacy custom LM still warns, including through a program.

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
OpenAI-style message dictionaries.

Engine selection:

- `engine="auto"` (default): prefer native lm15 for supported routes and
  representable inputs. Unsupported routes and ordinary provider-specific inputs
  that cannot be represented faithfully select LiteLLM **before execution**.
- `engine="lm15"`: require the native backend. Unsupported mappings raise.
- `engine="litellm"`: explicitly use the compatibility backend.

Authentication failures, timeouts and provider errors never cause a switch to
another backend. Native capability errors also raise rather than dropping the
requested feature. Text completions and client settings not implemented by the
native integration remain on LiteLLM.

### Caching, retries, usage and streaming

DSPy owns response caching, retries, candidate fan-out, callbacks and history.
Ordinary calls retain the old cache-key format and can read existing SDK response
entries. Cache reads happen before engine construction. New native entries store
plain serialized data and work with restricted cache deserialization. No old
cache rewrite is required. Explicit typed calls have a separate cache namespace.

Native `n` answers use `n` separate requests in sequence; LiteLLM compatibility
calls retain the backend's native `n` behavior. Separate requests can take longer
and bill input tokens more than once. A failed candidate does not restart earlier
successful candidates, and incomplete sets are not cached. Bounded parallel
execution is a follow-up tracked beside the candidate loops; it can reduce latency,
but does not remove the input-token charges for separate requests.

`num_retries` counts additional attempts, with exponential delays of 1, 2, 4…
seconds, capped at 60 seconds unless the provider specifies `retry_after`.
Backend retries are disabled on the managed LiteLLM path. A stream is not retried
once a chunk has been sent to the caller. Caching a completed stream does not
replay fake token timing on a later ordinary cache hit.

`dspy.streamify` continues to use the existing listeners. Native events are
assembled into an lm15 response and adapted to listener-facing chunks without
mixing reasoning into answer text. LiteLLM's ordinary streaming chunks retain
their original shape. Custom chunk consumers should not assume every native
chunk is a LiteLLM class; its common fields remain available.

`streamify` is the **program-level** API: it selects predictor fields, emits tool
and module status messages, and delivers the final `Prediction`. lm15 events describe
**one model call**, so they do not replace those responsibilities. The 3.5 migration
will move the internal LM boundary to canonical requests/responses and events; it
does not call for removing `streamify` or replacing its program-level role.

Token usage is recorded once per public call, and cache hits add no billed usage.
Native response counters stay provider-verbatim in `Response.usage`; DSPy's
legacy prompt/completion counters are derived separately. A native cost estimate
is **unknown**, not zero, when no reliable pricing is available. Existing cached
SDK responses retain their historical cost metadata.

## Migrating OpenAI-style messages

The public `messages=` argument is deprecated, including provider SDK message
objects as well as dictionaries. This does not deprecate `Request.messages`, which
holds canonical lm15 `Message` objects. For example, this form still runs in 3.4:

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

One typed request returns one response containing one assistant message. Use
ordinary calls for `n` answers. To continue a conversation, add `response.message`
to the next request's messages. LiteLLM typed streaming currently supports Chat
Completions only; the native Responses engine supports Responses streaming.

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

## Custom engines and legacy plugins

New custom backends implement the small engine interface:

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
Supply `async_engine=` with `async complete(request)` and `stream(request)` returning
an async iterator for async calls. DSPy does not silently run a sync backend on the
async path. Engines must not add another DSPy cache or retry loop. Custom engines
are caller-owned and are not closed by DSPy.

Implementing custom LMs through `BaseLM.forward(prompt=None, messages=None, **kwargs)`
or `aforward` is **deprecated**. Implement an engine instead, as shown above and in
the [custom-engine tutorial](../tutorials/custom_lm_engines/index.md). The old subclass
interface remains supported throughout DSPy **3.4** and is scheduled for removal in
**3.5**, along with both legacy-engine wrappers and `complete_legacy()` shortcuts.

Calls through that old interface emit a `DeprecationWarning`, subject to Python's
normal warning filters. Existing plugins are still automatically wrapped by legacy
engines in 3.4, preserving ordinary inputs and outputs. DSPy does not add caching or
retries around automatically wrapped plugins because they may already own those
behaviors. A legacy plugin's private streaming implementation remains its responsibility.

The explicit `LegacyEngine` and `AsyncLegacyEngine` wrappers from
`dspy.clients.engines` are **3.4 transition tools only**, not a permanent escape
hatch. Constructing either wrapper emits a deprecation warning; both are scheduled
for removal in 3.5. Migrate the underlying implementation, rather than only wrapping
it. During the transition, when a plugin already handles caching or retries, disable
those on the outer `dspy.LM` with `cache=False, num_retries=0` to avoid a second layer.

Custom engines must implement `complete(Request) -> Response`, plus their declared
async/streaming counterparts. If DSPy selects a custom engine's `complete_legacy()`
shortcut for an ordinary call, it warns. That shortcut will not exist in 3.5.

An existing `forward_contract="legacy"` declaration is harmless. The experimental
`forward_contract="typed_lm"` contract is removed and rejected explicitly.

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
equivalents; ordinary compatibility calls retain those provider-specific inputs.

Old pickled objects containing the removed classes are not automatically migrated.
Load and export them in their original environment first. This is separate from
ordinary provider-response caches and ordinary saved-program configuration.

Custom adapters should migrate to `lm(request)` / `await lm.acall(request)` and
parse the returned lm15 `Response`. In 3.5 this is the internal contract throughout
DSPy: adapters must not pass OpenAI-style dictionaries or use the list-returning
prompt convenience. Existing 3.4 adapters can continue running during migration,
but must not depend on the removed `dspy.clients.openai_format` module.

The full built-in adapter migration is scheduled for 3.5; the 3.4 implementation
still renders dictionary messages and parses list outputs. The temporary internal
warning marker is not a public API for custom adapters to use.

DSPy's signature types (`Image`, `Audio`, `File`, `Tool`, `ToolCalls`, `History`,
`Reasoning`, etc.) and error classes are not removed.
