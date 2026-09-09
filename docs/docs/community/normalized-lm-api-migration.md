# DSPy 3.4 LM migration

DSPy's LM layer now uses the lm15 objects bundled inside DSPy. Import them from
`dspy.lm15`; no separate installation is needed. They are the original lm15
classes, not DSPy wrappers or subclasses.

## Ordinary programs

Keep using ordinary calls and DSPy modules:

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
outputs = lm("Hello")
outputs = lm(messages=[{"role": "user", "content": "Hello"}])
```

Ordinary calls return lists of strings or dictionaries. `experimental=True` no
longer changes that return type; it still controls other experimental features.

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
successful candidates, and incomplete sets are not cached.

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

Token usage is recorded once per public call, and cache hits add no billed usage.
Native response counters stay provider-verbatim in `Response.usage`; DSPy's
legacy prompt/completion counters are derived separately. A native cost estimate
is **unknown**, not zero, when no reliable pricing is available. Existing cached
SDK responses retain their historical cost metadata.

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

Existing custom `BaseLM.forward(prompt=None, messages=None, **kwargs)` and
`aforward` implementations are wrapped by legacy engines. Ordinary plugin inputs
and outputs remain supported. A warning points authors toward the **planned 3.5
migration**; it is not a removal deadline. DSPy does not add caching or retries
around these plugins because they may already own those behaviors. A legacy
plugin's private streaming implementation remains its responsibility.

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

Custom adapters should call `lm(messages=messages, **lm_kwargs)` and parse the
returned list, not depend on the removed private conversion methods or
`dspy.clients.openai_format`.

DSPy's signature types (`Image`, `Audio`, `File`, `Tool`, `ToolCalls`, `History`,
`Reasoning`, etc.) and error classes are not removed.
