# Typed LM API migration plan

DSPy is moving toward a typed language-model boundary while keeping `dspy.BaseLM` as the public base class for language models.

**Most DSPy users do not need to change anything in DSPy 3.3**. Existing `lm(...)`, modules, and programs keep their current behavior by default. The typed LM API is opt-in in 3.3 with `dspy.context(experimental=True)`.

TLDR: `dspy.LM.forward` is currently untyped and mixes DSPy-specific behavior with OpenAI/LiteLLM-shaped inputs. We will migrate `BaseLM.forward` and `LM.forward` from:

```python
def forward(self, prompt=None, messages=None, **kwargs):
    ...
```

to:

```python
def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
    ...
```

!!! note "Status"
    This is a migration plan for the DSPy 3.3–3.6/4.0 series. Names and exact release timing may change before implementation lands, but the staged compatibility plan below should guide discussion.

!!! info "Community feedback wanted"
    This plan mostly affects custom LMs and adapters. If you maintain one, please review the proposed one-line `forward_contract` migration and share feedback before the default LM path changes.

## Who is affected?

| Group | What to do now | Future requirement |
| --- | --- | --- |
| Most DSPy users | Nothing required. Optionally try the direct `lm(...)` API with `dspy.context(experimental=True)` and provide feedback. | DSPy programs will keep working before, during, and after this migration without user changes. |
| Existing custom LM authors | Nothing required in 3.3. If you want to be explicit, add `forward_contract = "legacy"`. | Add an explicit `forward_contract`; eventually migrate to `forward_contract = "typed_lm"` before legacy support is removed. |
| New custom LM authors | Use `forward_contract = "typed_lm"` and implement `forward(request: dspy.LMRequest) -> dspy.LMResponse`. | No later migration needed if you start with the typed contract. |
| Custom adapter authors | Call `lm(...)`, not `lm.forward(...)`. | Build `LMRequest` objects and parse `LMResponse` directly. |


## Background

Today, `BaseLM` subclasses implement an untyped forward method, with a few optional parameters:

```python
def forward(self, prompt=None, messages=None, **kwargs):
    ...
```

That hook usually receives OpenAI/LiteLLM-shaped inputs and returns an OpenAI-like provider response. DSPy then post-processes that response into a `list[str | dict]` containing outputs.

Because the current parameters are untyped, it is hard to know inside an LM exactly which inputs you will get and what types they will contain. The new contract is typed and provider-neutral. We have designed `LMRequest` and `LMResponse` to be flexible enough for LMs backed by many different provider APIs:

```python
def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
    ...
```

DSPy has settled on the internal LM type system around `LMRequest`, `LMResponse`, typed messages, parts, config, usage, and stream events. These types should be treated as the stable direction for LM implementations. Concrete LMs translate between these DSPy types and their provider API.

## Why this matters

The typed boundary gives DSPy one clear internal representation for LM calls:

```text
LMRequest -> BaseLM -> LMResponse
```

That gives DSPy and the community:

- cleaner custom LM implementations,
- less OpenAI/LiteLLM-shaped logic inside adapters,
- first-class support for multimodal inputs, tool calls, reasoning, citations, usage, and provider metadata,
- a more expressive direct `lm(...)` UX,
- a clearer path for community packages to ship LMs that feel and are treated like first-class DSPy LMs.

The migration is staged so existing code keeps working while new code can opt into the typed path.

## Guide for DSPy users

Most users should not need to change anything in 3.3.

Default behavior remains legacy:

```python
outputs = lm("hello")
# list[str | dict]
```

To try the typed LM API in 3.3, use the existing experimental switch:

```python
with dspy.context(experimental=True):
    response = lm("hello")
    print(response.text)
```

Typed responses carry structured data:

```python
response.text
response.outputs
response.usage
response.cache_hit
response.provider_data
```

The typed path also makes direct `lm(...)` calls more expressive. Strings, typed messages, media parts, previous responses, and explicit `LMRequest` objects all flow through one call API.

!!! warning "Experimental 3.3 API"
    The typed LM symbols are importable without `experimental=True`. In DSPy 3.3, direct typed `lm(...)` calls are available behind `dspy.context(experimental=True)` while the API settles. Key helpers such as `dspy.LMRequest`, `dspy.LMResponse`, `dspy.System`, `dspy.User`, `dspy.Assistant`, `dspy.ToolCall`, and `dspy.ToolResult` are available at the top level. The complete typed LM vocabulary is available under `dspy.core.types`, e.g. `dspy.core.types.LMTextPart` and `dspy.core.types.LMImagePart`.

Multimodal request with instructions:

```python
from dspy.core.types import LMImagePart

with dspy.context(experimental=True):
    response = lm(
        dspy.System("Be concise."),
        dspy.User("Describe this image.", dspy.Image("https://example.com/dog.png")), #Coming soon!
        temperature=0.2,
    )
```

Multi-turn conversation:

```python
with dspy.context(experimental=True):
    response = lm(
        dspy.User("What is DSPy?"),
        dspy.Assistant("DSPy is a framework for programming LM pipelines."),
        dspy.User("Say that in five words."),
    )
```

Tool-call transcript:

```python
with dspy.context(experimental=True):
    response = lm(
        dspy.User("What is the weather in Paris?"),
        dspy.Assistant(dspy.ToolCall(id="call_1", name="get_weather", args={"city": "Paris"})),
        dspy.ToolResult('{"temperature": "22 C"}', call_id="call_1", name="get_weather"),
        dspy.User("Summarize the result."),
    )
```

Passing a previous response back into the conversation:

```python
with dspy.context(experimental=True):
    first = lm("Explain DSPy in one sentence.")
    follow_up = lm(
        dspy.User("Explain DSPy in one sentence."),
        first,
        dspy.User("Now make it even shorter."),
    )
```

## Guide for custom LM authors

Custom LM authors should declare which `forward()` contract their class implements.

Legacy LMs should add:

```python
class MyLegacyLM(dspy.BaseLM):
    forward_contract = "legacy"

    def forward(self, prompt=None, messages=None, **kwargs):
        ...
```

Typed LMs should add:

```python
class MyTypedLM(dspy.BaseLM):
    forward_contract = "typed_lm"

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        ...
```

In DSPy 3.3, classes without an explicit `forward_contract` are treated as legacy for compatibility. In later releases, missing declarations will warn and then may become errors or change defaults.

A minimal typed LM looks like this:

```python
class EchoLM(dspy.BaseLM):
    forward_contract = "typed_lm"

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        return dspy.LMResponse.from_text("hello", model=request.model)
```

### Reference typed LM: the OpenAI-compat engine

`_OpenAICompatLM` (in `dspy/clients/openai_compat_lm.py`) is the first full
production typed LM shipped with DSPy. It connects directly to an OpenAI Chat
Completions-compatible HTTP endpoint without LiteLLM or the OpenAI SDK. It is
an internal engine, not a public API: `dspy.LM` is the user-facing interface,
and the router constructs the engine under the hood. Serialized programs
record `dspy.LM` router state (model string, `api_base`, request defaults)
plus an `engine` block, never the engine class itself.

```python
from dspy.clients.openai_compat_lm import _OpenAICompatLM  # internal

lm = _OpenAICompatLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="local",  # Some servers require any non-empty token; omit if yours doesn't.
)
```

Its implementation in `dspy/clients/openai_compat_lm.py` is the reference for
translating `LMRequest` into a provider request, translating the provider
response into `LMResponse`, and normalizing transport/provider failures into
DSPy's `LMError` hierarchy. It is also the reference for credential handling in
typed LMs: `api_key` accepts a string or a zero-argument callable resolved per
request (so vaults and OAuth refreshers plug in), the resolution ladder is
explicit key, then `api_key_env`, then an opt-in `OPENAI_API_KEY` fallback
(`use_openai_api_key_env=True`), and keys are never serialized or written into
cache keys in the clear. It supports Chat Completions only; the OpenAI
Responses API is not part of its surface. It is also the reference streaming
implementation — see "Streaming contract for typed LMs" below.

### Credential patterns for typed LMs

Typed LMs are data-plane objects: they perform inference, never setup. A missing
credential is a typed, actionable error — not a prompt, a browser window, or a
silent fallback to a different account. The OpenAI-compat engine sets the
pattern that other typed LMs should follow.

**1. Accept a handle, not only a string.** `api_key` takes a string or a
zero-argument callable returning one:

```python
lm = _OpenAICompatLM(
    model="my-model",
    base_url="https://gateway.example.com/v1",
    api_key=lambda: my_vault.read("gateway-token"),
)
```

The callable is invoked on every request, so vaults, OAuth refreshers, SSO
resolvers, and rotating tokens all plug in without the LM knowing which. The
resolved token travels only in the request header.

**2. Keep the resolution ladder short, fixed, and documented.** Resolution
order is:

1. Explicit `api_key` (string or callable) — always wins.
2. `api_key_env` — a named environment variable the user chose.
3. `OPENAI_API_KEY` — only when `use_openai_api_key_env=True`. Ambient fallback
   is opt-in so a key meant for one provider is never sent to another endpoint
   by accident.
4. No credential — by default the request is sent unauthenticated, which is
   correct for local endpoints. Pass `require_auth=True` to instead fail
   locally with a typed `LMNotConfiguredError` whose message names the exits;
   endpoints that reject an unauthenticated request return a typed
   `LMAuthError` either way.

Every rung is inspectable; nothing is discovered behind the user's back.

**3. Secrets stay out of everything.** API keys must never appear in serialized
state, logs, history, or cache keys:

- `dump_state()` never writes a key. When an explicit key was used, the saved
  state also disables the ambient env fallback so loading a program cannot
  silently switch accounts.
- Sensitive headers (`Authorization`, `X-API-Key`, cookies) are stripped from
  serialized `extra_headers`.
- Cache identity uses a SHA-256 fingerprint of the resolved credential, so
  responses from different accounts never collide while the key itself is
  never stored.

**4. Fail with typed errors.** Map provider auth failures onto DSPy's error
hierarchy (`LMAuthError`, `LMBillingError`, `LMRateLimitError`, ...) with
status, provider code, request ID, and `retry_after` populated. The error
message should tell the user exactly what to set and where — for a missing
credential, the error message is the onboarding.

New typed LM implementations should treat these four rules as the contract,
even when the underlying provider SDK offers its own credential discovery.

The callable-credential seam is validated against the major clouds'
OpenAI-compatible surfaces:

- **Azure OpenAI (v1 API):** `azure_ad_token_provider` and openai-python's
  callable `api_key` have the same shape; use
  `azure.identity.get_bearer_token_provider`.
- **Amazon Bedrock (Chat Completions endpoint):** Bedrock API keys work as
  bearer tokens; short-term keys expire, so a refreshing callable is the
  natural fit.
- **Vertex AI (OpenAI-compatible endpoint):** wrap `google-auth` credentials
  in a callable that refreshes on expiry and returns `credentials.token`.

Credentials that live in provider-specific headers, such as Azure's `api-key`
or Google's `x-goog-api-key`, are supported for static keys via
`extra_headers`, which strips sensitive header names from serialized state and
folds header fingerprints into the cache identity. AWS SigV4 request signing
is intentionally out of scope for a token-shaped seam: the signature covers
the request body, so supporting it requires a separate request-signing hook.

### Constructor conventions for typed LMs

Typed LMs deliberately do not continue DSPy's historical reliance on
`**kwargs` for behavior. The conventions, set by the OpenAI-compat engine:

- **Behavioral parameters are explicit and keyword-only.** Endpoint identity,
  credentials, transport, and capability flags are named parameters after
  `(model, base_url)`. New behavioral parameters are added conservatively —
  each must be justified by a real provider need before entering the
  signature — and are never absorbed through `**kwargs`.
- **Request parameters keep the familiar user path, then become typed.**
  `temperature=`, `max_tokens=`, and provider extras still work at
  construction and call time, but they normalize immediately into `LMConfig`
  (with provider-specific extras in `config.extensions`, named and
  inspectable). `**kwargs` exists only at the public constructor and call
  surface for ergonomics; it never travels through internals.
- **No typed LM assigns a new meaning to `**kwargs`.** Anything that would
  have been a kwarg convention becomes either an explicit parameter (if
  behavioral) or an `LMConfig` field or extension (if a request parameter).

### Streaming contract for typed LMs

Streaming is part of the typed LM contract, designed once so the whole typed
LM family implements it the same way. The vocabulary is the stream types in
`dspy/core/types.py`, now public: `LMStreamStartEvent`, `LMStreamDeltaEvent`
(carrying `LMTextDelta`, `LMThinkingDelta`, `LMToolCallDelta`, and friends),
`LMStreamOutputEndEvent`, `LMStreamEndEvent`, and `LMStreamErrorEvent`.

**The user surface is a separate method, never a flag.** `lm.stream(...)` and
`lm.astream(...)` accept the same inputs as `lm(...)` and return
`dspy.LMStream` / `dspy.AsyncLMStream`: iterate for events, then call
`.result()` for the final `LMResponse`. A `stream=True` kwarg that forks the
return type of `__call__` is deliberately rejected.

```python
stream = lm.stream("Write a haiku about rivers.")
for event in stream:
    if event.type == "delta" and event.delta.type == "text_delta":
        print(event.delta.text, end="", flush=True)
response = stream.result()
```

**The provider seam is `forward_stream`.** A typed LM that streams natively
implements `forward_stream(request) -> Iterator[LMStreamEvent]` and declares
`supports_streaming = True`. Async callers get incremental events either from
a native `aforward_stream` or, by default, from the base class bridging the
synchronous stream through a worker thread. The rules, set by
the OpenAI-compat engine:

1. **Every LM streams; only some stream incrementally.** When
   `supports_streaming` is False, `stream()` runs the buffered `forward()`
   call and replays the finished response as events
   (`dspy.core.response_to_stream_events`). Consumers program against one
   event vocabulary and never branch on the backend.
2. **Streamed and buffered calls are observationally identical afterward.**
   History and usage are recorded once, when the stream completes, through
   the same `_finalize_lm_response` path as a non-streaming call — and a
   completed stream stores its final response under the same cache key as
   the buffered call, so either form of the same request hits one cache
   entry and a cache hit replays as events without an HTTP call.
3. **Provider chunk translation is shape mapping, kept out of transport.**
   `ChatCompletionChunkAssembler` in `openai_format.py` turns Chat
   Completions chunks into normalized events; the concrete LM owns SSE
   framing, retries, and errors. Retry only failures that occur before the
   first event is yielded — a stream that already produced output is never
   silently restarted.
4. **Errors are typed, exactly as in `forward()`.** Pre-stream HTTP failures
   normalize through the same `LMError` mapping; mid-stream failures raise
   during iteration.

**Explicitly out of scope, on purpose.** The existing `dspy.streamify` /
`StreamListener` path — which parses adapter wire formats (`[[ ## ... ## ]]`,
partial JSON, XML tags) out of raw provider bytes — is untouched and remains
the way module-level streaming works today. Bridging it onto typed events
(format-keyed listeners consuming `LMStreamDeltaEvent`s instead of raw bytes,
so adapter grammars stop being duplicated inside the listener) is the
intended future step; it is named here so nobody designs against it, but it
is not part of this contract yet.

### Planned typed LM family and routing

The patterns above were designed to survive the next implementations without
redesign. The intended sequence and the decisions already made:

1. **`AnthropicLM`, then `OpenAIResponsesLM`, then `GoogleLM`** — one typed LM
   per PR. The credential rules are placement-agnostic: what varies per
   provider is only *where* the resolved credential goes (`Authorization:
   Bearer` for OpenAI-shaped APIs, `x-api-key` plus `anthropic-version` for
   Anthropic, `x-goog-api-key` for Google API keys, with OAuth tokens using
   the Bearer slot everywhere). Each typed LM owns its placement; the
   resolution ladder, callable handles, serialization hygiene, and typed
   errors are identical.
2. **Extract the shared credential resolver when the second typed LM lands**
   (not before): a small helper implementing `explicit key (str or callable)
   -> named api_key_env -> opt-in ambient env -> None or LMNotConfiguredError`,
   parameterized only by the provider's ambient env name. Shared helpers are
   extracted once a second implementation exists, not before: one
   implementation is a pattern, two are shared code.
3. **Providers whose API keys live in a non-Bearer header** select placement
   by credential kind, not by widening the handle: the handle stays a string
   or zero-argument callable.
4. **Routing with LiteLLM fallback** (phases 3.4 and 3.5 in the migration
   table below): a pure-lookup
   registry mapping model-string prefixes to typed LM classes, consulted by a
   factory; a miss falls back to the LiteLLM-backed `dspy.LM`. This composes
   without adapter changes because `BaseLM.__call__` already normalizes both
   forward contracts, and without serialization changes because saved states
   record the concrete LM class. Requirements carried forward from the design
   work: the resolution must be able to name which rung fired (typed class vs
   LiteLLM fallback) — no silent magic; routing must not cut users off from
   LiteLLM-only capabilities such as fine-tuning (route those back to
   LiteLLM or raise `LMUnsupportedFeatureError` naming the fallback); and any
   registry/catalog metadata used to populate the routing table is advisory
   only — it may suggest capability flags but never inject credentials or
   change what is sent on the wire.

## Guide for custom adapter authors

Adapters should call the LM object, not `forward()` directly.

Preferred typed boundary:

```python
request = dspy.LMRequest.from_call(
    model=lm.model,
    messages=messages,
    **lm_kwargs,
)
response = lm(request)
```

Avoid this in adapters:

```python
lm.forward(...)
```

`BaseLM.__call__()` is the compatibility boundary. It owns input normalization, choosing the legacy or typed `forward()` path, adapting legacy outputs into `LMResponse`, and preserving public return behavior unless `experimental=True` is enabled.

During the transition, adapters may still convert `LMResponse` back to legacy parser inputs. The long-term direction is for adapters to parse `LMResponse` directly.

## Version sequence

| Version | Custom `BaseLM.forward` contract | Public `lm(...)` behavior | LiteLLM role |
| --- | --- | --- | --- |
| 3.3 | Missing `forward_contract` is treated as legacy. | Typed returns available only through `experimental=True` or explicit `LMRequest` calls. | Current `dspy.LM` LiteLLM path remains the default. |
| 3.4 | Missing `forward_contract` is treated as legacy and warns. | Still requires `experimental=True` or explicit `LMRequest` while migration continues. | Native typed LMs become preferred where available; LiteLLM is used as a compatibility fallback. |
| 3.5 | Require explicit contract or flip default after final review. | Typed path becomes default with a legacy escape hatch. | Native typed LMs remain preferred; LiteLLM is used as a compatibility fallback but may require manual installation. |
| 3.6 or 4.0 | Remove the legacy `forward(prompt, messages, **kwargs)` implementation contract after final review. | `forward(request: LMRequest) -> LMResponse` is the only supported `BaseLM` implementation contract. | TBD whether the LiteLLM fallback remains. |

The important distinction is that removing the legacy `BaseLM.forward(prompt, messages, **kwargs)` contract does not require removing LiteLLM. LiteLLM can continue as a typed compatibility implementation that accepts `LMRequest` internally and returns `LMResponse`.

Before changing the default, DSPy will give custom LM authors enough time to add one of:

```python
forward_contract = "legacy"
```

or:

```python
forward_contract = "typed_lm"
```
