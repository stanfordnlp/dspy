# DSPy × Apple Foundation Models — Project TODO

## Goal
Add `dspy.AppleFoundationLM` and `dspy.AppleLocalLM` — PR-quality DSPy backends for Apple's
on-device models. Not just thin wrappers: native guided generation maps Apple's `@generable`
constrained decoding to DSPy's typed output pipeline, and `AppleLocalLM` wraps mlx-lm for
any HuggingFace model on Apple Silicon.

---

## Status: Implementation complete on WSL. Ready for Mac integration testing.

### Shipped
- [x] Clone DSPy into this directory
- [x] `python -m venv .venv && source .venv/bin/activate`
- [x] `pip install -e ".[dev]"`
- [x] `git checkout -b feature/apple-foundation-models`
- [x] `dspy/clients/apple_fm.py` — `AppleFoundationLM`, all helpers
- [x] `dspy/clients/apple_local.py` — `AppleLocalLM`, MLX backend, CoreML stub
- [x] `dspy/clients/__init__.py` — guarded exports for both adapters
- [x] `dspy/__init__.py` — `dspy.AppleFoundationLM`, `dspy.AppleLocalLM` exposed
- [x] `tests/clients/test_apple_fm.py` — 38 unit tests, all pass on WSL
- [x] `tests/clients/test_apple_local.py` — 10 unit tests, all pass on WSL
- [x] `tests/integration/test_apple_fm_integration.py` — 11 Mac-only tests, skip cleanly on Linux
- [x] `docs/docs/learn/programming/language_models.md` — Apple Foundation + Apple Silicon tabs
- [x] `docs/docs/api/models/AppleFoundationLM.md` / `AppleLocalLM.md`
- [x] `examples/apple_on_device_lm.py` — 4 selectable demos
- [x] Pushed to `git@github.com:zombat/DSPy-AppleFM.git` on `feature/apple-foundation-models`

### Remaining
- [x] Mac integration test run (see Verification Checklist)
- [x] Fix any live-SDK issues surfaced by integration tests
- [ ] Open PR against `stanfordnlp/dspy main`

---

## Architecture Decision Record

These are the non-obvious calls made during implementation. Read this before touching the code.

---

### ADR-1: Why `fm.Tool` is subclassed dynamically at runtime

**Problem:** Apple's SDK requires tools to be registered as subclasses of `fm.Tool` — you can't
pass a callable or wrap a plain function. DSPy tools, on the other hand, are arbitrary Python
objects (callables, instances with `.func`, or `dspy.Tool` wrappers). There's no static base
class to subclass at module level because `fm.Tool` doesn't exist until `import apple_fm_sdk`
runs, which can only happen on macOS 26+.

**Decision:** `_dspy_tool_to_apple_tool()` uses `type()` at call time to dynamically create a
fresh subclass of `fm.Tool` for each DSPy tool, wiring `call(**kwargs)` to the DSPy callable.
The class is named after the DSPy tool's `.name` attribute so Apple's SDK can identify it.

**Why not a static subclass?**
- `fm.Tool` is only importable on macOS 26+ with Apple Intelligence enabled. A top-level
  `class _WrappedTool(fm.Tool): ...` would make the entire module unimportable on Linux,
  breaking the `try/except ImportError` guard in `__init__.py` that lets non-Mac installs
  work fine.
- Dynamic subclassing keeps the guard clean: the class is only created inside `aforward()`,
  which is only called after `__init__` has already validated the platform and imported the SDK.

**Class caching:** Generated subclasses are cached in the module-level `_tool_class_cache` dict,
keyed by `(tool_name, id(func))`. The same DSPy tool object reuses the same class across calls,
so the class is not a throwaway anymore. `id(func)` is stable for the lifetime of DSPy program
objects. If Apple adds per-class SDK-side registration (e.g. schema validation on subclass
creation), the cache means that registration only fires once per unique tool — no change to the
public interface would be needed.

---

### ADR-2: How we mocked an entire OS-specific SDK to get unit tests passing on Linux

**Problem:** `apple_fm_sdk` doesn't exist on Linux. `mlx_lm` doesn't exist outside Apple Silicon.
Both are imported lazily inside methods, but the test infrastructure still needed to exercise
`_pydantic_to_generable()`, `_flatten_messages()`, `_apply_chat_template()`, tool conversion,
caching logic, and response structure — none of which touch the real hardware at all.

**Decision:** Each test file defines a `_make_fake_fm_sdk()` / `_make_fake_mlx_lm()` factory
that returns a `types.ModuleType` populated with hand-rolled Python stand-ins, then injects it
into `sys.modules` via an `autouse` fixture before any import of the real package can occur.

Key constraints that shaped the fake:

1. **`guide()` must return `""`, not `MagicMock`.**
   `_pydantic_to_generable()` passes guide return values as dataclass field defaults, then
   calls `dataclasses.asdict()` on the result for JSON serialization. `asdict()` recurses into
   field values — if a field holds a `MagicMock`, it hits `TypeError: Object of type MagicMock
   is not JSON serializable`. Returning `""` (a plain str) keeps the dataclass picklable and
   JSON-safe end to end.

2. **`generable()` must be a passthrough decorator, not a MagicMock.**
   If `fm.generable(cls)` returns a MagicMock, `dataclasses.make_dataclass` produces a class
   that the fake `session.respond()` can't instantiate. The fake `generable` just returns its
   argument unchanged.

3. **`LanguageModelSession` must be async-context-safe.**
   Apple's real session is an async object. The fake `respond()` is an `async def` that returns
   a plain string or a dataclass instance depending on whether `generating=` was passed.

4. **Platform checks are patched at the `platform` module, not inside the class.**
   `AppleFoundationLM.__init__` calls `platform.system()`. Patching `platform.system` globally
   in the fixture means even indirect callers (lazy imports inside methods) see `"Darwin"`.
   We also patch `apple_fm_sdk.SystemLanguageModel.is_available` to return `(True, "")`.

5. **Caching tests patch `dspy.cache` directly with `patch.object`.**
   Early versions tried to instantiate a second `AppleFoundationLM(cache=True)` to test cache
   miss/hit behaviour. This re-triggered `__init__` platform checks inside the patch scope,
   which worked but was fragile. The fix: set `lm.cache = True` on the fixture instance and
   use `patch.object(dspy.cache, "get") / patch.object(dspy.cache, "put")` to assert call
   counts directly. No second instantiation needed.

6. **`test_fallback_when_no_chat_template` replaces the tokenizer object entirely.**
   The original approach was `del lm._mlx_tokenizer.apply_chat_template`. This fails because
   instance method lookup goes through the class — you can't delete a method from an instance.
   Fix: replace `lm._mlx_tokenizer` with a bare `_BareTokenizer()` instance (a local class
   that simply doesn't define `apply_chat_template`). The attribute lookup then correctly falls
   through to the fallback path.

**Result:** 48 unit tests run on WSL with zero macOS dependencies. The integration tests
import the real SDK and skip cleanly (`===== 11 skipped in 0.04s =====`) when not on macOS 26+.

---

### ADR-3: Why `_FMUsage` implements `__iter__`

`BaseLM` (and DSPy's history tracking) calls `dict(response.usage)` to record token counts.
This works naturally for LiteLLM response objects because their `Usage` class supports the
mapping protocol. Our `_FMUsage` is a plain dataclass. Adding `__iter__` to yield `(key, value)`
pairs makes `dict()` work without converting `_FMUsage` to a dict subclass or adding a
`__getitem__` / `keys()` pair. Cheapest fix that satisfies the contract.

---

### ADR-4: Why caching lives in `forward()`, not `BaseLM.__call__`

`dspy.LM` gets automatic caching via LiteLLM's response cache, which is wired inside
`litellm.completion()`. `BaseLM` subclasses that bypass LiteLLM (as both Apple adapters do)
get no caching for free — `BaseLM.__call__` does not cache. We implemented explicit
`dspy.cache.get() / put()` calls inside each adapter's `forward()`. The cache key is a dict
of `{model, messages, temperature, max_tokens, ...remaining kwargs}` with DSPy-internal keys
(`num_retries`, `stream`, `n`) excluded so they don't create spurious cache misses.

---

### ADR-5: Why `response_format` is intercepted in `aforward()`, not at the DSPy adapter layer

DSPy's `ChatAdapter` injects a JSON schema into the prompt when it sees a structured output
request, then parses the model's text response back. For most LLMs this is the only option.
Apple's SDK offers something better: `session.respond(generating=SomeGenerableClass)` triggers
native constrained decoding — the model is guaranteed to emit valid tokens for that schema.

Intercepting `response_format` in `aforward()` (before it could become a prompt injection)
lets us route Pydantic models through the native path. The result is serialized back to a JSON
string so DSPy's output parser sees exactly what it would have seen from the prompt-based path —
same contract, better reliability on small on-device models.

Fallback: if `_pydantic_to_generable()` can't map a field type cleanly, it logs a warning and
returns `None`. `aforward()` then skips the `generating=` arg and lets DSPy's normal prompt
injection handle it.

---

### ADR-6: Per-call `LanguageModelSession` (stateless pattern)

Apple's `LanguageModelSession` is designed to maintain conversational state across turns.
DSPy modules are stateless — each `forward()` call is independent, and DSPy manages its own
prompt construction (including injecting few-shot examples). Reusing a session across calls
would accumulate spurious conversation history and produce wrong outputs.

Decision: create a new `LanguageModelSession` on every `aforward()` call. The overhead is
acceptable for the on-device model (no network round-trip). If Apple exposes a stateless
single-call API in a future SDK version, we can switch without changing the public interface.

---

### ADR-7: Why `aforward()` uses a lazy `asyncio.Semaphore`, not a threading lock

DSPy optimizers (MIPROv2, BootstrapFewShot) evaluate candidate prompts in parallel by issuing
many concurrent `aforward()` calls. `AppleLocalLM.aforward()` offloads each call to the default
thread-pool executor via `loop.run_in_executor()`. Without a gate, 20 concurrent optimizer
candidates submit 20 simultaneous MLX inference jobs, each trying to allocate activation memory
in Apple Silicon's unified memory pool — instant OOM.

**Why `asyncio.Semaphore` and not `threading.Semaphore`?**
`aforward()` is the async entry point; all concurrent callers share the same event loop. An
`asyncio.Semaphore` in `aforward()` is the natural gate: callers suspend cooperatively before
submitting to the thread pool, so only `max_concurrency` blocking jobs run at a time.
A `threading.Semaphore` inside `_generate()` would also work but would block a thread-pool
thread while waiting, wasting thread resources.

**Why lazy initialisation (not in `__init__`)?**
`asyncio.Semaphore` must be created in the event loop that will use it. `__init__` runs in
whatever context the user calls it from — often outside an event loop (plain script,
`pytest` setup). Creating the semaphore lazily on the first `aforward()` call guarantees it
is always bound to the correct loop.

**Default `max_concurrency=1`**: serialises all MLX calls to protect unified memory. Users with
spare RAM or quantized models can raise it via the constructor.

---

### ADR-8: Why `tools` raises `NotImplementedError` rather than being silently dropped

`mlx-lm` has no native tool-calling API — tools can only be approximated via the HuggingFace
chat template's Jinja tool-formatting syntax, which requires non-trivial extra work and does
not guarantee structured output parsing. Silently dropping `tools=[...]` would let DSPy
programs appear to run successfully while actually skipping all tool invocations, producing
wrong outputs with no diagnostic.

Raising `NotImplementedError` with a message pointing to `AppleFoundationLM` (which has full
native `fm.Tool` support) surfaces the problem immediately and tells the user exactly what to
do. This is consistent with the `backend="coreml"` guard in `__init__` — the pattern for
"this feature is not yet implemented" is an explicit error, not silent degradation.

---

### ADR-9: Why token counts are computed from the tokenizer, and why `response_cost = 0.0`

**Token counts:** Apple's Foundation Model SDK exposes no tokenizer. `mlx-lm` loads a
HuggingFace tokenizer (`self._mlx_tokenizer`) as part of `mlx_lm.load()`. Token counts are
computed by encoding the flat prompt and the generated text with `tokenizer.encode()` after
inference. The flat prompt is now returned from `_generate()` alongside the text to avoid
applying the chat template twice.

Accurate token counts matter for DSPy's optimization budget tracking: `BaseLM` stores
`dict(response.usage)` in history, and optimizer callbacks read `prompt_tokens` /
`completion_tokens` to estimate cost. Zeros would make all budget estimates useless.

**`response_cost = 0.0`:** On-device inference has no monetary cost, but DSPy's history
aggregator sums `entry["cost"]` across all calls. When `cost` is `None` (the previous
behaviour, where `_hidden_params` was `{}`), `sum([None, None, ...])` raises `TypeError`.
Setting `response_cost = 0.0` explicitly makes the sum safe while accurately representing
that on-device inference costs nothing.

---

### ADR-10: Why `aforward()` uses `asyncio.to_thread()` instead of `get_event_loop().run_in_executor()`

`asyncio.get_event_loop()` is deprecated since Python 3.10 — it emits a
`DeprecationWarning` when called outside a running event loop and may return a different
loop than the one actually running the task. `asyncio.to_thread()` (Python 3.9+) is the
idiomatic replacement: it always schedules work on the currently-running event loop's default
executor and takes positional/keyword arguments directly (no `lambda` wrapper needed).

Functionally identical: both offload the blocking `forward()` call to the OS thread pool,
freeing the event loop to process other coroutines while MLX grinds on the GPU.

---

### ADR-11: Why `context_window` uses `model_max_length` with a 4096 fallback, and why overflow is a warning not an error

**Source of the value:**
HuggingFace tokenizers write `model_max_length` into `tokenizer_config.json` when saving.
mlx-lm loads this via the standard HuggingFace `AutoTokenizer`, so it's available on
`self._mlx_tokenizer.model_max_length` for any model that ships a valid tokenizer config.
The 4096 fallback is conservative (all mainstream 7B–13B models have at least 4096 context)
but safe for production.

For `AppleFoundationLM`, Apple's SDK does not expose a context-size query. 4096 is the
documented limit for the initial Apple Intelligence release; this should be updated if Apple
exposes a programmatic API.

**Why a warning, not a hard error:**
MLX truncates over-long inputs internally (no crash, just silent truncation). A hard error
would break legitimate use cases where the user intentionally passes a long prompt and relies
on truncation. A warning surfaces the issue in logs without halting execution.

---

### ADR-12: Why unknown kwargs are warned-and-cleared instead of forwarded

**Cache key cleanliness:** `AppleLocalLM.forward()` spreads remaining `**kwargs` into the
`cache_request` dict. Unknown kwargs (e.g. `top_p=0.9`) would change the cache key without
changing the model output — every unique `top_p` value would create a new cold cache entry
for what is functionally the same generation. Clearing them after warning prevents this
silent cache fragmentation.

**Honest failure mode:** DSPy passes global settings (via `dspy.configure`) down to all
backends. A user who sets `dspy.configure(lm=my_apple_lm, top_p=0.9)` expecting probability
sampling should be told it has no effect, not silently get greedy sampling with a false
cache hit. The warning turns an invisible mismatch into a visible one.

---

### ADR-13: Streaming strategy for AppleLocalLM

**Status:** Implemented.

**Decision:** Streaming is supported via `dspy.streamify()` using DSPy's
`dspy.settings.send_stream` protocol, **not** via a `stream=True` kwarg.
`forward(stream=True)` still raises `NotImplementedError` with a message
directing users to `streamify()`.

**Two code paths:**

1. **Primary path — `forward()` in anyio worker thread (via `asyncify`):**
   `streamify()` wraps `Predict.__call__` with `asyncify`, which runs it in
   an anyio-managed worker thread.  `Predict.forward()` then calls
   `lm.forward()` from that thread.  When `dspy.settings.send_stream` is set,
   `forward()` calls `mlx_lm.stream_generate()` synchronously and pushes each
   `_LocalStreamChunk` to the anyio `MemoryObjectSendStream` via
   `anyio.from_thread.run(send_stream.send, chunk)`.

2. **Secondary path — `aforward()` for direct async callers:**
   When `await lm.aforward()` is called directly (bypassing `Predict`),
   `_stream_generate_async()` bridges `mlx_lm.stream_generate()` (sync) to
   an async generator via `asyncio.Queue` + `loop.call_soon_threadsafe()`.
   Each token is forwarded to `send_stream` with `await send_stream.send(chunk)`.

**Why `mlx_lm.stream_generate()` instead of `generate_step()`:**
`stream_generate()` is the public high-level API that wraps `generate_step`
internally, handles EOS detection, max-token limits, and token decoding.
Using it directly avoids reimplementing per-token control logic and is less
fragile against internal mlx-lm API changes.

**Why `forward()` is the primary path, not `aforward()`:**
`streamify()` → `asyncify` → anyio thread → `Predict.forward()` → `lm.forward()`.
`lm.aforward()` is never called in this path.  The secondary path exists only
for callers who invoke `await lm.aforward()` directly.

**Chunk type:** `_LocalStreamChunk(text, model, predict_id)` — a custom
dataclass, not a litellm `ModelResponseStream`.  DSPy's `streamify()` passes
custom chunk types through its wildcard branch to the caller.  `StreamListener`
field-extraction is therefore unavailable; all tokens stream raw.

---

### ADR-14: Why `session.respond(generating=...)` is wrapped in `try/except` with session recreation

**Problem:** `_pydantic_to_generable()` can return a valid `@generable` class (the Pydantic
schema was translated successfully), but the underlying Swift grammar engine can still reject
the schema at inference time. For example, a `Union[str, List[str]]` field might compile to a
Python dataclass without error, yet fail when Apple's native constrained-decoding compiler tries
to build the grammar automaton for it. Without a try/except, this raises an unhandled exception
and the caller (DSPy optimizer) sees a hard crash instead of a degraded-but-working response.

**Decision:** Wrap `await session.respond(generating=...)` in `try/except Exception`. On failure:

1. Log a `WARNING` explaining the fallback (so integration-test logs reveal schema issues).
2. `del session` — the failed session is in an undefined state; don't reuse it.
3. Recreate a fresh `LanguageModelSession` using the already-built `session_kwargs`.
4. Retry with `await session.respond(prompt=flat_prompt)` (no `generating=`), letting DSPy's
   standard JSON-schema prompt injection handle the structured output.

**Why not catch `TypeError` / `ValueError` specifically?**
The Swift bridge surfaces errors as Python `Exception` subclasses whose exact types depend on
the SDK version (and are undocumented). A broad `except Exception` is appropriate here because
the intent is unconditional fallback: any failure in the native path should degrade gracefully
rather than crash.

**Why recreate the session?**
A session that raised during `respond()` may have advanced internal state (partial token
emission, grammar automaton in a broken state). Reusing it for the fallback call risks
corrupting the next response. A fresh session from `fm.LanguageModelSession(**session_kwargs)`
guarantees a clean slate.

---

### ADR-15: Why `max_concurrency > 1` emits a warning instead of being hard-capped at 1

**Problem:** MLX's Python bindings call into a C++/Metal backend. The `mlx_lm.generate()`
function invokes the Metal command queue and allocates activation buffers in unified memory.
While HuggingFace tokenizers are thread-safe, it is undocumented whether a single `mlx.nn.Module`
instance supports concurrent `generate()` calls from multiple Python threads. If it does not,
`max_concurrency > 1` can cause Metal command queue deadlocks or segmentation faults that kill
the Python process with no traceback.

**Decision:** Warn rather than cap. Hard-capping would deny the benefit to users who test
and confirm thread-safety on their specific hardware + MLX version, or who load separate model
instances per thread. The warning surfaces the risk clearly so users can make an informed choice.

**What to do if you hit a crash:** If `max_concurrency > 1` hard-crashes the Python process
(segfault instead of a Python exception), MLX does not support concurrent generation on a single
model instance. Set `max_concurrency=1` or load a separate `AppleLocalLM` instance per thread.

**Why is the default `max_concurrency=1`?** It is always safe. DSPy optimizers issue many
parallel `aforward()` calls, but the `asyncio.Semaphore(1)` in `aforward()` serializes them
cooperatively, preventing OOM (ADR-7). A user who wants higher throughput and has validated
safety can opt in by raising the limit at construction time.

---

## Verification Checklist (run on Mac)

```bash
git clone git@github.com:zombat/DSPy-AppleFM.git
cd DSPy-AppleFM
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install apple-fm-sdk mlx-lm
```

- [ ] `pytest tests/clients/test_apple_fm.py tests/clients/test_apple_local.py -v` — 48 pass
- [ ] `python -c "import dspy; print(dspy.AppleFoundationLM)"` — prints class, no error
- [ ] `pytest tests/ -v --ignore=tests/integration` — no regressions in existing DSPy tests
- [ ] `pytest tests/integration/test_apple_fm_integration.py -v` — 11 pass on macOS 26+
- [ ] `python examples/apple_on_device_lm.py foundation` — live generation round-trip
- [ ] `python examples/apple_on_device_lm.py local` — mlx-lm generation round-trip
- [ ] `python examples/apple_on_device_lm.py structured` — native guided generation demo
