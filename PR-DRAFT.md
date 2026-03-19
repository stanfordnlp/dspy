# feat: Add `AppleFoundationLM` and `AppleLocalLM` — native Apple Silicon & Apple Intelligence backends

## Summary

This PR adds two new `BaseLM` subclasses that allow DSPy programs to run entirely on-device
on Apple Silicon, with no cloud dependency:

- **`dspy.AppleFoundationLM`** — wraps Apple's `LanguageModelSession` (macOS 26+, Apple
  Intelligence) with native constrained decoding for structured outputs and full `fm.Tool`
  support for tool calling.
- **`dspy.AppleLocalLM`** — wraps `mlx-lm` to run any HuggingFace model natively on Apple
  Silicon. Supports the full DSPy optimizer workflow including `BootstrapFewShot` and
  `MIPROv2`.

Both adapters are **zero-regression on Linux CI**: all platform-gated imports are lazily
loaded and wrapped in `try/except` guards. The entire test suite (69 unit tests) passes on
Linux/WSL with no macOS dependencies.

---

## Motivation

DSPy programs that optimize prompts via `BootstrapFewShot` or `MIPROv2` typically make
hundreds or thousands of LM calls. Running those optimization loops against cloud LLMs is
expensive and slow. Apple Silicon's unified memory makes it possible to run a capable
quantized model (e.g. `mlx-community/Llama-3.2-3B-Instruct-4bit`) at sub-100ms latency
with zero API cost. Developers can now:

1. Optimize programs locally on their Mac using `AppleLocalLM`.
2. Switch the configured LM to their production cloud model for the final deployment run.
3. Use `AppleFoundationLM` in shipping macOS apps that need private, on-device inference.

---

## Changes

### New files

| File | Description |
|------|-------------|
| `dspy/clients/apple_fm.py` | `AppleFoundationLM` adapter + shared response types (`_FMResponse`, `_FMUsage`, `_FMChoice`, `_FMMessage`) |
| `dspy/clients/apple_local.py` | `AppleLocalLM` adapter (MLX backend, CoreML stub), `_LocalStreamChunk` |
| `tests/clients/test_apple_fm.py` | 44 unit tests for `AppleFoundationLM` |
| `tests/clients/test_apple_local.py` | 25 unit tests for `AppleLocalLM` (includes streaming) |
| `tests/integration/test_apple_fm_integration.py` | 11 Mac-only live-SDK integration tests (skip cleanly on Linux) |
| `docs/docs/api/models/AppleFoundationLM.md` | API reference |
| `docs/docs/api/models/AppleLocalLM.md` | API reference |
| `examples/apple_on_device_lm.py` | 6 selectable runnable demos |

### Modified files

| File | Change |
|------|--------|
| `dspy/clients/__init__.py` | Guarded `try/except ImportError` exports for both adapters |
| `dspy/__init__.py` | `dspy.AppleFoundationLM`, `dspy.AppleLocalLM` top-level exports |
| `docs/docs/learn/programming/language_models.md` | Apple Foundation + Apple Silicon tabs |
| `pyproject.toml` | `norecursedirs` guard to prevent pytest from entering `experiments/` |

---

## Architecture

### `AppleFoundationLM` — Apple Intelligence (macOS 26+)

```python
lm = dspy.AppleFoundationLM()
dspy.configure(lm=lm)
result = dspy.Predict("question -> answer")(question="What is DSPy?")
```

**Native structured outputs.** When DSPy passes `response_format=SomePydanticModel` (its
standard structured-output path), this adapter intercepts it before it becomes a prompt
injection. `_pydantic_to_generable()` maps Pydantic field constraints to Apple's
`@generable` constrained decoding:

- `Literal["a", "b"]` → `fm.guide(anyOf=[...])`
- `int = Field(ge=1, le=5)` → `fm.guide(range=(1, 5))`
- `str = Field(pattern=r"\d+")` → `fm.guide(regex=...)`

The model is then called with `session.respond(generating=<generable_cls>)`, which guarantees
valid typed output at the token level — not a JSON parse of free text. The result is serialized
back to JSON so DSPy's output parser sees the same contract it would from the prompt path.

If `_pydantic_to_generable()` can't map a field (e.g. complex nested type), or if the Swift
grammar compiler rejects the schema at runtime, the adapter logs a warning, recreates a fresh
session, and retries without `generating=`, falling back gracefully to DSPy's standard
prompt-injection path.

**Tool calling.** Apple's SDK requires tools to be subclasses of `fm.Tool`, not plain
callables. `_dspy_tool_to_apple_tool()` dynamically subclasses `fm.Tool` at call time for
each DSPy tool, wiring `call(**kwargs)` to the DSPy callable. Generated subclasses are cached
by `(tool_name, id(func))` so Apple's per-class SDK registration fires exactly once per
unique tool.

**Async bridging.** Apple's SDK is async-only. `forward()` bridges to sync via
`asyncio.run()` with `nest_asyncio` support for Jupyter notebooks.

### `AppleLocalLM` — MLX (Apple Silicon, macOS 14+)

```python
lm = dspy.AppleLocalLM("mlx-community/Llama-3.2-3B-Instruct-4bit", bits=4)
dspy.configure(lm=lm)
```

**Mixed-LM pipelines.** The primary use case is cheap on-device preprocessing before
expensive cloud reasoning:

```python
local_lm = dspy.AppleLocalLM("mlx-community/Llama-3.2-3B-Instruct-4bit")
cloud_lm = dspy.LM("anthropic/claude-sonnet-4-6")

class PreprocessAndReason(dspy.Module):
    def __init__(self):
        self.extract = dspy.Predict("raw_text -> entities, dates", lm=local_lm)
        self.reason  = dspy.Predict("entities, dates -> verdict", lm=cloud_lm)
```

**Streaming.** `AppleLocalLM` supports `dspy.streamify()` via DSPy's `send_stream` protocol.
Wrapping any program with `dspy.streamify()` causes `forward()` to call
`mlx_lm.stream_generate()` and push each `_LocalStreamChunk` token to the stream in real time.

**Concurrency gate.** DSPy optimizers issue many parallel `aforward()` calls. Unconstrained
concurrent MLX inference jobs would exhaust Apple Silicon's unified memory pool and OOM.
`aforward()` gates all calls through a lazily-initialized `asyncio.Semaphore(max_concurrency)`
(default: 1) before offloading to `asyncio.to_thread()`. Users with spare RAM can raise the
limit at construction time; the adapter warns if `max_concurrency > 1` since MLX
thread-safety on a single model instance is undocumented.

**Context window tracking.** `context_window` is read from
`tokenizer.model_max_length` (with a 4096 fallback). A warning is logged when a prompt
would exceed the window rather than silently truncating.

### Shared design decisions

**Explicit caching.** Both adapters bypass LiteLLM, so LiteLLM's automatic caching is
unavailable. `dspy.cache.get/put` is wired explicitly in each `forward()`. The cache key
covers `{model, messages, temperature, max_tokens}`; DSPy-internal keys (`num_retries`,
`stream`, `n`) are excluded to prevent spurious misses. Unknown kwargs are warned and cleared
so they cannot silently fragment the cache.

**`BaseLM` response contract.** `_FMUsage` implements `__iter__` to yield `(key, value)`
pairs so `dict(response.usage)` works as expected by `BaseLM._process_completion`.
`_FMResponse` carries an explicit `_hidden_params={"response_cost": 0.0}` field — `None`
cost would cause `sum([None, ...])` to raise `TypeError` in DSPy's history aggregator.

**Explicit errors over silent degradation.** `stream=True` raises `NotImplementedError`
(a streaming caller expects an async generator, not a string — use `dspy.streamify()`).
`tools=[...]` raises `NotImplementedError` in `AppleLocalLM` (mlx-lm has no native tool API)
with a pointer to `AppleFoundationLM` for users who need tools. Unknown backends raise
`ValueError`.

---

## Testing

### Unit tests (Linux/WSL — zero macOS dependencies)

Each test file defines `_make_fake_fm_sdk()` / `_make_fake_mlx_lm()` factories that inject
synthetic `types.ModuleType` instances into `sys.modules` via `autouse` fixtures. This lets
every logical path — message flattening, Pydantic→generable conversion, tool wrapping, cache
hit/miss, concurrency gating, kwarg warn/clear, context overflow warning, ARC session
fallback, streaming chunk emission — be exercised without any Apple hardware or SDK.

```
tests/clients/test_apple_fm.py    — 44 tests
tests/clients/test_apple_local.py — 25 tests
```

### Integration tests (Mac only)

```
tests/integration/test_apple_fm_integration.py — 11 tests
```

These import the real `apple_fm_sdk` and skip cleanly on non-macOS platforms:

```
===== 11 skipped in 0.04s =====   ← Linux CI
===== 11 passed in  X.XXs =====   ← macOS 26+ with Apple Intelligence
```

Coverage includes: live round-trip generation, structured output via `@generable`, tool
invocation, cache round-trip against the real `dspy.cache`, and `AppleLocalLM` mlx-lm
generation.

---

## Design Decisions

Non-obvious implementation choices made during development.

---

### Why `fm.Tool` is subclassed dynamically at runtime

Apple's SDK requires tools to be registered as subclasses of `fm.Tool` — you can't pass a
callable or wrap a plain function. DSPy tools, on the other hand, are arbitrary Python objects
(callables, instances with `.func`, or `dspy.Tool` wrappers). There's no static base class to
subclass at module level because `fm.Tool` doesn't exist until `import apple_fm_sdk` runs,
which can only happen on macOS 26+.

`_dspy_tool_to_apple_tool()` uses `type()` at call time to dynamically create a fresh subclass
of `fm.Tool` for each DSPy tool, wiring `call(**kwargs)` to the DSPy callable. A top-level
`class _WrappedTool(fm.Tool): ...` would make the entire module unimportable on Linux. Dynamic
subclassing keeps the import guard clean: the class is only created inside `aforward()`, which
is only reached after `__init__` has already validated the platform and imported the SDK.
Generated subclasses are cached by `(tool_name, id(func))` so any per-class SDK-side
registration fires exactly once per unique tool.

---

### How we mocked an entire OS-specific SDK to get unit tests passing on Linux

`apple_fm_sdk` doesn't exist on Linux. `mlx_lm` doesn't exist outside Apple Silicon. Both are
imported lazily inside methods. Each test file defines a factory that returns a
`types.ModuleType` populated with hand-rolled Python stand-ins, then injects it into
`sys.modules` via an `autouse` fixture before any import of the real package can occur.

Key constraints that shaped the fakes:

1. **`guide()` must return `""`, not `MagicMock`.** `_pydantic_to_generable()` passes guide
   return values as dataclass field defaults, then calls `dataclasses.asdict()`. If a field
   holds a `MagicMock`, `asdict()` raises `TypeError: Object of type MagicMock is not JSON
   serializable`.

2. **`generable()` must be a passthrough decorator.** If `fm.generable(cls)` returns a
   MagicMock, `dataclasses.make_dataclass` produces a class the fake `session.respond()` can't
   instantiate.

3. **`LanguageModelSession` must be async-context-safe.** The fake `respond()` is an
   `async def` that returns a plain string or a dataclass instance depending on whether
   `generating=` was passed.

4. **Platform checks are patched at the `platform` module level.** Patching
   `platform.system` globally in the fixture means even indirect callers see `"Darwin"`.

5. **`mlx_lm.sample_utils` must be registered as a real submodule in `sys.modules`.** A flat
   `types.ModuleType` with an attribute `sample_utils` is not the same as a registered submodule
   — Python's import system resolves `from mlx_lm.sample_utils import make_sampler` by looking
   up `"mlx_lm.sample_utils"` in `sys.modules` directly.

---

### Why `_FMUsage` implements `__iter__`

`BaseLM` calls `dict(response.usage)` to record token counts. This works for LiteLLM objects
because their `Usage` class supports the mapping protocol. `_FMUsage` is a plain dataclass.
Adding `__iter__` to yield `(key, value)` pairs makes `dict()` work without converting
`_FMUsage` to a dict subclass. Cheapest fix that satisfies the contract.

---

### Why caching lives in `forward()`, not `BaseLM.__call__`

`dspy.LM` gets automatic caching via LiteLLM's response cache. `BaseLM` subclasses that
bypass LiteLLM get no caching — `BaseLM.__call__` does not cache. Both Apple adapters wire
`dspy.cache.get() / put()` explicitly in `forward()`. The cache key covers
`{model, messages, temperature, max_tokens}` with DSPy-internal keys (`num_retries`, `stream`,
`n`) excluded to prevent spurious misses.

---

### Why `response_format` is intercepted in `aforward()`, not at the DSPy adapter layer

DSPy's `ChatAdapter` injects a JSON schema into the prompt for structured output requests, then
parses the text response back. Apple's SDK offers something better:
`session.respond(generating=SomeGenerableClass)` triggers native constrained decoding — the
model is guaranteed to emit valid tokens for that schema.

Intercepting `response_format` in `aforward()` (before it becomes a prompt injection) lets us
route Pydantic models through the native path. The result is serialized back to a JSON string
so DSPy's output parser sees exactly what it would have seen from the prompt-based path — same
contract, better reliability on small on-device models.

---

### Per-call `LanguageModelSession` (stateless pattern)

Apple's `LanguageModelSession` is designed to maintain conversational state across turns. DSPy
modules are stateless — each `forward()` call is independent, and DSPy manages its own prompt
construction (including injecting few-shot examples). Reusing a session across calls would
accumulate spurious conversation history and produce wrong outputs. A new session is created on
every `aforward()` call; the overhead is acceptable for on-device inference (no network
round-trip).

---

### Why `aforward()` uses a lazy `asyncio.Semaphore`, not a threading lock

DSPy optimizers evaluate candidate prompts in parallel via many concurrent `aforward()` calls.
Without a gate, 20 concurrent optimizer candidates submit 20 simultaneous MLX inference jobs,
each allocating activation memory in Apple Silicon's unified memory pool — instant OOM.

An `asyncio.Semaphore` in `aforward()` is the natural gate: callers suspend cooperatively
before submitting to the thread pool, so only `max_concurrency` blocking jobs run at a time. A
`threading.Semaphore` inside the sync path would also work but would block a thread-pool thread
while waiting, wasting thread resources.

The semaphore is initialized lazily on first `aforward()` call because `asyncio.Semaphore` must
be created in the event loop that will use it — `__init__` often runs outside any event loop.

---

### Why `tools` raises `NotImplementedError` in `AppleLocalLM` rather than being silently dropped

`mlx-lm` has no native tool-calling API. Silently dropping `tools=[...]` would let DSPy
programs appear to run successfully while actually skipping all tool invocations, producing
wrong outputs with no diagnostic. The error message points directly to `AppleFoundationLM`,
which has full native `fm.Tool` support.

---

### Why token counts are computed from the tokenizer, and why `response_cost = 0.0`

Apple's Foundation Model SDK exposes no tokenizer. `mlx-lm` loads a HuggingFace tokenizer as
part of `mlx_lm.load()`. Token counts are computed by encoding the flat prompt and the
generated text with `tokenizer.encode()` after inference. Accurate counts matter for DSPy's
optimization budget tracking — `BaseLM` stores `dict(response.usage)` in history and optimizer
callbacks read `prompt_tokens` / `completion_tokens` to estimate cost.

`response_cost = 0.0` rather than `None`: on-device inference has no monetary cost, but DSPy's
history aggregator sums `entry["cost"]` across all calls. `sum([None, ...])` raises `TypeError`.
Setting `0.0` explicitly makes the sum safe while accurately representing zero cost.

---

### Why unknown kwargs are warned-and-cleared instead of forwarded

Unknown kwargs (e.g. `top_p=0.9`) would change the cache key without changing the model output
— every unique `top_p` value creates a new cold cache entry for what is functionally the same
generation. Clearing them after warning prevents silent cache fragmentation and surfaces the
mismatch to users who set global `dspy.configure` options expecting them to apply.

---

### Streaming strategy for `AppleLocalLM`

Streaming is supported via `dspy.streamify()` using DSPy's `dspy.settings.send_stream`
protocol, **not** via a `stream=True` kwarg. `forward(stream=True)` raises
`NotImplementedError` with a message directing users to `streamify()`.

**Two code paths:**

1. **Primary path — `forward()` in anyio worker thread (via `asyncify`):**
   `streamify()` wraps `Predict.__call__` with `asyncify`, which runs it in an anyio-managed
   worker thread. `Predict.forward()` calls `lm.forward()` from that thread. When
   `dspy.settings.send_stream` is set, `forward()` calls `mlx_lm.stream_generate()`
   synchronously and pushes each `_LocalStreamChunk` to the anyio `MemoryObjectSendStream` via
   `anyio.from_thread.run(send_stream.send, chunk)`.

2. **Secondary path — `aforward()` for direct async callers:**
   When `await lm.aforward()` is called directly (bypassing `Predict`),
   `_stream_generate_async()` bridges `mlx_lm.stream_generate()` (sync) to an async generator
   via `asyncio.Queue` + `loop.call_soon_threadsafe()`.

`mlx_lm.stream_generate()` is used rather than the lower-level `generate_step()` because it
is the public high-level API that handles EOS detection, max-token limits, and token decoding
internally — avoiding fragile reimplementation of per-token control logic.

`_LocalStreamChunk(text, model, predict_id)` is a custom dataclass, not a litellm
`ModelResponseStream`. DSPy's `streamify()` passes custom chunk types through its wildcard
branch to the caller; `StreamListener` field-extraction is unavailable and all tokens stream
raw.

---

### Why `session.respond(generating=...)` is wrapped in `try/except` with session recreation

`_pydantic_to_generable()` can return a valid `@generable` class, but the underlying Swift
grammar engine can still reject the schema at inference time (e.g. a `Union[str, List[str]]`
field might compile without error yet fail when Apple's constrained-decoding compiler tries to
build the grammar automaton). On failure:

1. Log a `WARNING` (so integration-test logs reveal schema issues).
2. `del session` — the failed session may have advanced internal state.
3. Recreate a fresh `LanguageModelSession` from the already-built `session_kwargs`.
4. Retry with `await session.respond(prompt=flat_prompt)` (no `generating=`).

`except Exception` rather than `TypeError` / `ValueError` specifically: the Swift bridge
surfaces errors as Python `Exception` subclasses whose exact types depend on the SDK version
and are undocumented. The intent is unconditional fallback.

---

### Why `max_concurrency > 1` emits a warning instead of being hard-capped at 1

MLX's Python bindings call into a C++/Metal backend. It is undocumented whether a single
`mlx.nn.Module` instance supports concurrent `generate()` calls from multiple Python threads.
If it does not, `max_concurrency > 1` can cause Metal command queue deadlocks or segfaults
with no Python traceback.

Warn rather than cap: hard-capping would deny the benefit to users who test and confirm
thread-safety on their specific hardware + MLX version, or who load separate model instances
per thread. The default is `max_concurrency=1` (always safe); users who want higher throughput
opt in explicitly.

---

## Notes for reviewers

- `apple_fm_sdk` is not yet available on PyPI — the package name is provisional. The import
  guard in `dspy/clients/__init__.py` means this PR is safe to merge before the SDK ships.
- `AppleLocalLM(backend="coreml")` raises `NotImplementedError` with an invitation to
  contribute. The CoreML path is stubbed (not deleted) so the `backend=` parameter is
  part of the public API from day one.
- The `LanguageModelSession` is intentionally created per-call (stateless pattern). DSPy
  manages all prompt construction including few-shot injection; reusing a session across calls
  would accumulate spurious conversational history.
