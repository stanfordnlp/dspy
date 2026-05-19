# Plan: In-place `BaseLM` evolution to the normalized LM contract

## Context

PR [stanfordnlp/dspy#9765](https://github.com/stanfordnlp/dspy/pull/9765) and its stacked follow-ups introduce a normalized LM foundation (typed `LMRequest`/`LMResponse`, capability declarations, native streaming events, error taxonomy, multi-provider backends) by adding a **parallel** `dspy.LanguageModel` class alongside the existing `dspy.BaseLM`. The stack then carries dual-contract dispatch through adapters, predict, history, and streaming — paying for the parallel hierarchy in roughly 1,000 lines of `isinstance(lm, BaseLM | LanguageModel)` forks.

The Slack thread on naming (`p1778753830737829` in `#C09SX8P2YRJ`) surfaced the real constraint: the project wants **stable contracts** as a cultural promise. A parallel class with no removal path violates that — both names must be maintained indefinitely, and any "transitional" name (`BaseLMv2`) becomes permanent the moment external subclassers pick it up (`dspy-mlx`, `dspy-vllm-lm`, Databricks, etc.).

This plan delivers the same user-visible wins as the PR stack — typed contract, native streaming, error normalization, OpenAI/Anthropic/Gemini backends without LiteLLM, multi-provider router — by **evolving `BaseLM` in place**. The migration mechanism is the one the PR itself already lands in stack #4: `__init_subclass__` introspects the subclass's `forward` signature, sets a contract-version sentinel, and emits a one-shot `DeprecationWarning` at the user's `class MyLM(BaseLM):` line. Legacy subclasses keep working through a translation shim inside `BaseLM.__call__`. The v1 signature is removed in DSPy 4.0.

Net effect: single durable base class name, no `LanguageModel` vs `BaseLM` ambiguity, no `dspy.LM` namespace expansion, ~70% smaller adapter integration, single global history, single LM type in the public API. Every concrete capability the PR provides is preserved.

## Constraints from prior discussion

- `dspy.LM` stays as the user-facing entry point; eventually it can become a router by swapping construction without changing the documented constructor surface.
- No mixins. Capabilities are signaled by **method overrides on `BaseLM`**, detected via `type(self).forward_stream is not BaseLM.forward_stream`.
- New methods (`forward_stream`, `aforward_stream`, `normalize_error`, `load_state`, `stream`, `astream`) are added to `BaseLM` directly.
- The `forward` signature is evolved through a deprecation cycle, not renamed.
- The removal version is committed in code and docs from day one (target: DSPy 4.0).

## Commit-by-commit plan

Each commit mirrors a step in the PR stack but applies it to `BaseLM` rather than introducing `LanguageModel`. The branch ends at the same user-visible surface as `pr/new-lm-docs`, minus the parallel class.

---

### Commit 1 — Foundation types (additive)

Mirrors `pr/new-lm-foundation` (the base PR #9765) but **without** the `LanguageModel` class.

**Adds:**
- `dspy/clients/lm/types.py` — `LMRequest`, `LMResponse`, `LMOutput`, `LMUsage`, `LMMessage`, `LMConfig`, `LMReasoningConfig`, `LMToolChoice`, `LMToolSpec`, `LMCacheConfig`, `LMPromptCacheConfig`, `LMHistoryEntry`; parts (`LMTextPart`, `LMImagePart`, `LMAudioPart`, `LMFilePart`, `LMToolCallPart`, `LMToolResultPart`, `LMThinkingPart`, `LMCitationPart`, `LMRefusalPart`, `LMBasePart`); stream deltas/events (`LMTextDelta`, `LMThinkingDelta`, `LMToolCallDelta`, `LMCitationDelta`, `LMImageDelta`, `LMAudioDelta`, `LMStreamStartEvent`, `LMStreamDeltaEvent`, `LMStreamOutputEndEvent`, `LMStreamEndEvent`, `LMStreamErrorEvent`); helpers `LMOutputBuilder`, `LMStream`, `AsyncLMStream`; message helpers `System`, `Developer`, `User`, `Assistant`, `ToolCall` (renamed to `LMToolCall` — see open question Q3), `ToolResult`.
- `dspy/clients/lm/__init__.py` — public surface for the new types. Re-exports everything so `dspy.lm.X` works for every name; `dspy/__init__.py` separately re-exports the curated top-level subset (see "Confirmed decisions" above).
- `dspy/clients/_litellm.py` — lazy `get_litellm()` accessor.
- `dspy/utils/lazy_import.py` — generic lazy-import helper.
- `dspy/utils/exceptions.py` — new error classes: `DSPyError`, `LMError`, `LMAuthError`, `LMBillingError`, `LMConfigurationError`, `LMInvalidRequestError`, `LMNotConfiguredError`, `LMProviderError`, `LMRateLimitError`, `LMServerError`, `LMTimeoutError`, `LMTransportError`, `LMUnsupportedFeatureError`, `LMUnsupportedModelError`. `ContextWindowExceededError` re-classified under `LMError`.
- `dspy/clients/__init__.py` — make LiteLLM imports lazy (same edits as PR #9765).

**Does not touch:** `dspy/clients/base_lm.py`, `dspy/clients/lm.py` (except for lazy-LiteLLM in `lm.py`).

**Tests** (mirror `tests/newlm/test_language_model_foundations.py`):
- `tests/lm/types/test_request_response.py`
- `tests/lm/types/test_messages.py`
- `tests/lm/types/test_streaming_events.py`
- `tests/lm/types/test_errors.py`

**Why this is a separate commit:** the types are pure addition; they ship independently and can land first to unblock backend work.

---

### Commit 2 — Evolve `BaseLM` to the dual-contract base

Replaces the parallel `LanguageModel` class. This is the load-bearing commit.

**Modifies `dspy/clients/base_lm.py`:**

1. **Add new methods (typed contract):**
   ```python
   def forward(self, request: LMRequest) -> LMResponse: ...
   async def aforward(self, request: LMRequest) -> LMResponse: ...
   def forward_stream(self, request: LMRequest) -> Iterator[LMStreamEvent]: ...
   async def aforward_stream(self, request: LMRequest) -> AsyncIterator[LMStreamEvent]: ...
   def normalize_error(self, error: Exception, request: LMRequest) -> Exception: ...
   @classmethod
   def load_state(cls, state: dict) -> Self: ...
   def stream(self, *items, ...) -> LMStream: ...
   def astream(self, *items, ...) -> AsyncLMStream: ...
   ```
   Each defaults to `NotImplementedError` (or `LMUnsupportedFeatureError`) unless overridden.

2. **Add `__init_subclass__` for contract detection** (lifts the code already in `pr/new-lm-adapters`, generalized):
   ```python
   def __init_subclass__(cls, *, _internal: bool = False, **kwargs):
       super().__init_subclass__(**kwargs)
       cls._lm_contract_version = _detect_contract_version(cls)
       if _internal or cls.__module__.startswith("dspy."):
           return
       if cls._lm_contract_version == 1:
           warnings.warn(
               "Override `forward(self, request: LMRequest) -> LMResponse` on dspy.BaseLM. "
               "The legacy `forward(prompt, messages)` signature is deprecated and will be "
               "removed in DSPy 4.0; see https://dspy.ai/migration/baselm.",
               DeprecationWarning, stacklevel=2,
           )
   ```
   `_detect_contract_version(cls)` inspects `cls.forward`'s signature:
   - Has a parameter named `prompt` or `messages` → v1.
   - Single non-self positional parameter (optionally annotated `LMRequest`) → v2.
   - Ambiguous (`*args, **kwargs` passthrough) → v1 + warn.

3. **Add typed dispatch in `__call__` / `acall`:**
   - Always normalize input into `LMRequest` (`normalize_request(*items, prompt=, messages=, request=, **kwargs)`).
   - If `type(self)._lm_contract_version == 2`: call `_forward_with_retry(request)` → `_finalize_response` → return `LMResponse`. Built-in retry/cache/callback/history scaffolding (lift verbatim from `dspy/clients/language_models/base.py` in PR #9765).
   - If `type(self)._lm_contract_version == 1`: translate `request` → `(prompt, messages, **kwargs)`, call legacy `forward`, parse OpenAI-shape result via the existing `_process_completion`/`_process_response` code into an `LMResponse`. Same callbacks/history wrapping.

4. **Capability properties become override-derived:**
   ```python
   @property
   def supports_streaming(self) -> bool:
       return type(self).forward_stream is not BaseLM.forward_stream

   @property
   def supports_async(self) -> bool:
       return type(self).aforward is not BaseLM.aforward
   ```
   Existing `supports_function_calling` / `supports_reasoning` / `supports_response_schema` / `supported_params` stay as overridable properties (today's contract is preserved).

5. **History entry shape:** keep the v1 entry shape for v1 subclasses; emit the new normalized entry shape for v2 subclasses. Same `GLOBAL_HISTORY` list — no bifurcation.

6. **`dump_state` / `load_state`:** record `f"{type(lm).__module__}.{type(lm).__qualname__}"` as `_dspy_lm_class`. `BaseLM.load_state` resolves the class path and constructs.

**Modifies `dspy/clients/lm.py`:**
- Class declaration becomes `class LM(BaseLM, _internal=True):` to suppress the deprecation warning on the in-tree v1 subclass.
- LM stays v1 in this commit; migrated to v2 in commit 3 along with the OpenAI-backend introduction.

**Tests:**
- `tests/lm/base/test_init_subclass_detection.py` — v1 / v2 / ambiguous signatures.
- `tests/lm/base/test_legacy_dispatch.py` — v1 subclass receives translated args, OpenAI-shape return is wrapped into `LMResponse`.
- `tests/lm/base/test_typed_dispatch.py` — v2 subclass receives `LMRequest`, returns `LMResponse`, all callbacks/retry/cache fire.
- `tests/lm/base/test_capability_detection.py` — override-derived properties.
- `tests/lm/base/test_deprecation_warning.py` — `__init_subclass__` warns at user line, suppressed for in-tree subclasses.

---

### Commit 3 — OpenAI backends as `BaseLM` subclasses

Mirrors `pr/new-lm-openai`. Same code as that stack, parented onto `BaseLM` instead of `LanguageModel`.

**Adds:**
- `dspy/clients/lm/backends/openai.py` — `OpenAIChatLM(BaseLM)`, `OpenAITextLM(BaseLM)`, `OpenAIResponsesLM(BaseLM)`. Each overrides typed `forward`, `aforward`, `forward_stream`, `aforward_stream`, `normalize_error`, `dump_state`, with `capabilities` declared. No LiteLLM dependency.
- `dspy/clients/lm/backends/openai_format.py` — `completion_stream_to_events`, `responses_stream_to_events`, request/response converters.

**Modifies `dspy/clients/lm.py`:**
- Migrate `LM` from v1 to v2: implement typed `forward(request)` that calls the LiteLLM bridge. Remove the `_internal=True` opt-out (no longer needed).
- Existing constructor surface (`model_type`, `temperature`, `max_tokens`, `cache`, `callbacks`, `num_retries`, `provider`, `finetuning_model`, `launch_kwargs`, `train_kwargs`, `use_developer_role`, `**kwargs`) preserved bit-for-bit.

**Tests** (mirror `tests/newlm/test_openai_lms.py`, `tests/newlm/test_language_model_foundations.py` for OpenAI parts):
- `tests/lm/backends/test_openai_chat.py`
- `tests/lm/backends/test_openai_responses.py`
- `tests/lm/backends/test_openai_text.py`
- `tests/lm/backends/test_openai_streaming.py`

---

### Commit 4 — Router for `dspy.LM`

Mirrors `pr/new-lm-router`.

**Adds:**
- `dspy/clients/lm/router.py` with:
  - `LMRouter` as an **actual class** subclassing `BaseLM` (resolved at construction time by delegating `forward` etc. to a routed backend). See open question Q4.
  - `register_lm_backend(factory_or_class, prefix=...)` plugin hook.
  - Default routes: `openai/…` → `OpenAIChatLM`/`OpenAIResponsesLM`/`OpenAITextLM` (selected by `model_type` and endpoint URL).
- `dspy/clients/lm/backends/litellm.py` — `LiteLLMLM(BaseLM)` wrapping LiteLLM as a routable backend (factored out of `dspy/clients/lm.py`).
- `dspy.settings.experimental = False` (new) — gates router opt-in.

**Modifies `dspy/clients/__init__.py`:**
- `LM.__new__` reads `settings.experimental`: opt-in returns `LMRouter(...)`; default returns `super().__new__(cls)` (legacy LiteLLM-backed construction). Same dispatch mechanism as `pr/new-lm-router`, but the routed result is also a `BaseLM` — no separate type.

**Tests** (mirror `tests/newlm/test_lm_router*.py`):
- `tests/lm/router/test_basic_routing.py`
- `tests/lm/router/test_register_backend.py`
- `tests/lm/router/test_provider_prefix_routes.py`
- `tests/lm/router/test_dspy_lm_dispatch.py`

---

### Commit 5 — Adapter integration

Mirrors `pr/new-lm-adapters` but **without dual dispatch** — the integration code is ~70% smaller because there's only one base class.

**Modifies `dspy/adapters/base.py`:**

1. `Adapter.__call__` becomes:
   ```python
   processed_signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)
   inputs = self.format(processed_signature, demos, inputs)
   request = lm.normalize_request(messages=inputs, **lm_kwargs)
   if settings.send_stream is not None and lm.supports_streaming:
       response = self._stream_lm_response(lm, request)
   else:
       response = lm(request=request)
   return self._call_postprocess(processed_signature, signature, response, lm, lm_kwargs)
   ```
   No `isinstance` branches. Legacy v1 subclasses keep working because `lm(request=...)` translates inside `BaseLM.__call__` (commit 2).

2. **Adapter-level field parsing** (orthogonal-but-adopted refactor from `pr/new-lm-adapters`): add `stream_start_identifier(field)`, `consume_stream_field_buffer(field, buffer, *, final)`, `make_stream_response(listener, token, *, is_last_chunk)` as abstract on `Adapter`. Implement in `ChatAdapter`, `JSONAdapter`, `XMLAdapter` — lifting today's per-adapter logic out of `StreamListener`.

3. `_stream_lm_response(lm, request)` / `_astream_lm_response(lm, request)` (lift verbatim from `pr/new-lm-adapters`): drive `lm.stream(request)`, push `StreamResponse`s into `settings.send_stream` for listener-bound fields, also push raw `LMStreamEvent`s when `settings.stream_include_lm_events` is on. Return `stream.result()`.

4. Postprocess reads `LMResponse.outputs` parts: text → adapter parse; tool calls → `ToolCalls.from_dict_list`; native types (Citations, Reasoning) → `field_annotation.parse_lm_response(output_dict)`.

**Modifies `dspy/predict/predict.py`:**
- Type union goes away: `isinstance(lm, BaseLM)` only.
- `dump_state`/`load_state` use the class-path serialization from commit 2.

**Modifies `dspy/adapters/{chat_adapter,json_adapter,xml_adapter,two_step_adapter}.py`:**
- Implement the new field-parsing methods.

**Tests** (mirror `tests/newlm/test_language_model_program_integration.py`, `test_lm_router_adapter_compat.py`):
- `tests/lm/integration/test_adapter_typed_dispatch.py`
- `tests/lm/integration/test_adapter_legacy_dispatch.py` — confirms v1 `BaseLM` subclasses still work end-to-end.
- `tests/lm/integration/test_predict_lm_state.py`
- `tests/lm/integration/test_inspect_history.py` — single global history.

---

### Commit 6 — Streaming wiring

Mirrors `pr/new-lm-streaming`. Very small — the side-channel design survives unchanged.

**Modifies `dspy/streaming/streamify.py`:**
- Add `include_lm_events: bool = False` arg to `streamify`. Maps to `settings.stream_include_lm_events`.
- `streaming_response` SSE serializer learns `LMStreamEvent`, `StreamResponse`, `StatusMessage` chunk types (in addition to legacy `litellm.ModelResponseStream`).
- `settings.send_stream` mechanism preserved.

**Modifies `dspy/streaming/streaming_listener.py`:**
- Strip per-adapter parsing (now in `Adapter` from commit 5). `StreamListener` becomes a simple buffer + parser-delegate.

**Modifies `dspy/dsp/utils/settings.py`:**
- Add `stream_include_lm_events: bool = False`.

**Tests:**
- `tests/lm/streaming/test_streamify_lm_events.py`
- `tests/lm/streaming/test_streamify_legacy_chunks.py` — backwards compat for v1 LMs that push raw `ModelResponseStream`.

---

### Commit 7 — Anthropic + Gemini backends

Mirrors `pr/new-lm-providers`.

**Adds:**
- `dspy/clients/lm/backends/anthropic.py` — `AnthropicLM(BaseLM)`.
- `dspy/clients/lm/backends/gemini.py` — `GenAILM(BaseLM)`.

**Modifies `dspy/clients/lm/router.py`:**
- Add prefix routing for `anthropic/`, `gemini/`, `google/`, `genai/`.

**Tests:**
- `tests/lm/backends/test_anthropic.py`
- `tests/lm/backends/test_gemini.py`
- `tests/lm/backends/test_live_*.py` (skipped without credentials).

---

### Commit 8 — Docs and migration guide

Mirrors `pr/new-lm-docs`.

**Adds/modifies:**
- `docs/docs/api/models/LMMessage.md`, request/response API docs.
- `docs/docs/learn/programming/language_models.md` — typed contract, override `forward(request)`, capabilities via overrides, removal target for v1 signature.
- `docs/docs/migration/baselm.md` — **the migration guide referenced from the `DeprecationWarning`.** Covers: signature change, removal version (DSPy 4.0), worked example of porting a v1 `BaseLM` subclass to v2, capability declaration, streaming opt-in.
- `docs/docs/tutorials/streaming/index.md` updates.
- `tests/conftest.py` adjustments (silence in-tree deprecation warnings; surface external ones).

---

## File map: paths to be modified

**New files:**
- `dspy/clients/lm/__init__.py` (re-exports public types)
- `dspy/clients/lm/types.py` (≈1,700 lines, lifted from PR #9765's `language_models/types.py`)
- `dspy/clients/lm/router.py` (≈300 lines)
- `dspy/clients/lm/backends/__init__.py`
- `dspy/clients/lm/backends/openai.py` (≈680 lines)
- `dspy/clients/lm/backends/openai_format.py` (≈800 lines)
- `dspy/clients/lm/backends/litellm.py` (≈170 lines, factored from current `lm.py`)
- `dspy/clients/lm/backends/anthropic.py` (≈520 lines)
- `dspy/clients/lm/backends/gemini.py` (≈530 lines)
- `dspy/clients/_litellm.py` (≈40 lines)
- `dspy/utils/lazy_import.py` (≈130 lines)
- `docs/docs/migration/baselm.md`

**Heavily modified:**
- `dspy/clients/base_lm.py` — grows from ~350 to ~1,200 lines (absorbs the scaffolding that lived in `language_models/base.py` in the PR stack).
- `dspy/clients/lm.py` — typed-contract migration in commit 3.
- `dspy/clients/__init__.py` — exports, lazy LiteLLM, router dispatch.
- `dspy/__init__.py` — new error exports, type re-exports (see Q2 on namespace policy).
- `dspy/adapters/base.py` — single-dispatch integration + adapter-level field parsing (~80 added lines, not 311).
- `dspy/adapters/{chat,json,xml,two_step}_adapter.py` — implement field-parsing methods.
- `dspy/predict/predict.py` — class-path-aware state serialization (no `BaseLM | LanguageModel` union).
- `dspy/streaming/streamify.py` — SSE chunk-type handling.
- `dspy/streaming/streaming_listener.py` — delegate to adapter parsers.
- `dspy/dsp/utils/settings.py` — `experimental`, `stream_include_lm_events`.
- `dspy/utils/exceptions.py` — new `LM*` error taxonomy.

## Existing utilities to reuse

- `dspy/utils/callback.py` — `with_callbacks`, `BaseCallback`, `ACTIVE_CALL_ID` machinery (used by both `BaseLM.__call__` paths).
- `dspy/clients/cache.py` — `request_cache` decorator (wraps both v1 and v2 dispatch paths).
- `dspy/utils/inspect_history.py` — `pretty_print_history` (single, no merge needed).
- `dspy/streaming/messages.py` — `StreamResponse`, `StatusMessage`, `sync_send_to_stream` (consumed by adapter streaming helpers).
- `dspy/adapters/types/tool.py` — existing `ToolCalls.from_dict_list` reused in `_call_postprocess` for native tool-call output fields.
- Existing `BaseLM._process_completion` / `_process_response` — repurposed as the **legacy translation shim** inside `BaseLM.__call__` when `_lm_contract_version == 1`. No new parsing code needed for v1 subclasses.
- PR-stack-authored types and backends (`language_models/types.py`, `language_models/openai.py`, etc.) — lift wholesale into the new file layout; the only edits are reparenting subclasses from `LanguageModel` to `BaseLM` and adjusting imports.

## What we deliberately do not do

- No `dspy.LanguageModel`, no `dspy.BaseLanguageModel`, no `dspy.BaseLMv2`. The Slack naming debate is closed by not introducing a second class.
- No `is_legacy_lm` class attribute; the v1/v2 distinction is computed at `__init_subclass__` from the `forward` signature, not declared manually.
- No two parallel global histories. One list, entry shape normalized per subclass.
- No `BaseLM | LanguageModel` type unions in any public API (`Predict.__init__`, `Adapter.__call__`, callback signatures).
- No mixins (`StreamingMixin`, `AsyncMixin`, etc.). Capability is "did the subclass override the method?"

## Removal target

DSPy 4.0 drops v1 `forward(prompt, messages)` signature support from `BaseLM`. The `DeprecationWarning` message names this version. The migration guide (`docs/docs/migration/baselm.md`) lands in commit 8 with worked examples. `__init_subclass__` v1 detection and the v1 translation shim are removed in 4.0; `BaseLM.forward` becomes `def forward(self, request: LMRequest) -> LMResponse` only. The `_lm_contract_version` sentinel and `_internal=True` opt-out can also be deleted in 4.0.

## Confirmed decisions

- **Removal target: DSPy 4.0.** The `DeprecationWarning` in commit 2 and the migration guide in commit 8 both cite "removed in DSPy 4.0." `__init_subclass__` v1 detection and the v1 translation shim are deleted in 4.0; `BaseLM.forward` becomes typed-only.
- **Namespace policy: hybrid.** Promoted to top-level `dspy.*`: `LMRequest`, `LMResponse`, `LMMessage`, `LMConfig`, `LMUsage`, `LMHistoryEntry`, `LMCapabilities`, the message helpers `System`/`Developer`/`User`/`Assistant`/`LMToolCall`/`ToolResult`, and the new `LM*` error classes. Kept under `dspy.lm.*` (no top-level promotion): parts (`LMTextPart` etc.), deltas (`LMTextDelta` etc.), stream events (`LMStreamDeltaEvent` etc.), `LMOutputBuilder`, `LMStream`, `AsyncLMStream`, `LMReasoningConfig`, `LMToolChoice`, `LMToolSpec`, `LMCacheConfig`, `LMPromptCacheConfig`, `LMOutput`, `LMBasePart`, `LMDelta`, `LMStreamEvent`. Cuts the top-level commitment surface from ~50 names to ~15.
- **Tool-call helper name: `LMToolCall`.** Matches the `LM*` prefix of `LMRequest`/`LMResponse`/`LMMessage` and avoids the singular/plural footgun with the existing `dspy.ToolCalls` adapter output type. `ToolResult` keeps its name (no collision exists).
- **`LMRouter` is a proper `BaseLM` subclass.** Instances delegate `forward`/`aforward`/`forward_stream`/`aforward_stream`/`normalize_error`/`dump_state` to a routed backend selected at construction time from the registered factories. `isinstance(lm, LMRouter)` is meaningful; `isinstance(lm, BaseLM)` is True.

## Verification

End-to-end checks per commit:

**Commit 1 (types):**
```bash
pytest -q tests/lm/types/
python -c "from dspy import LMRequest, LMResponse, User, Assistant; print(LMRequest)"
```

**Commit 2 (BaseLM evolution):**
```bash
pytest -q tests/lm/base/
# Smoke: define a v1 subclass, expect DeprecationWarning at class line
python -W error::DeprecationWarning -c "
import dspy
try:
    class V1LM(dspy.BaseLM):
        def forward(self, prompt=None, messages=None, **kw): ...
except DeprecationWarning as e:
    print('warned:', e); raise SystemExit(0)
raise SystemExit(1)
"
# Smoke: define a v2 subclass, expect NO warning
python -W error::DeprecationWarning -c "
import dspy
class V2LM(dspy.BaseLM):
    def forward(self, request): return dspy.LMResponse.from_text('hi', model='t')
print('ok')
"
# Existing test suite must pass unchanged (legacy path still works):
pytest -q tests/clients/
```

**Commit 3 (OpenAI backends):**
```bash
pytest -q tests/lm/backends/test_openai_*.py
# Live (optional, with OPENAI_API_KEY):
pytest -q tests/lm/backends/test_live_openai.py
```

**Commit 4 (router):**
```bash
pytest -q tests/lm/router/
# Smoke: dspy.LM under experimental flag routes through LMRouter
python -c "
import dspy
dspy.settings.configure(experimental=True)
lm = dspy.LM('openai/gpt-4o-mini')
assert isinstance(lm, dspy.BaseLM)
print(type(lm).__name__)
"
```

**Commit 5 (adapter integration):**
```bash
pytest -q tests/lm/integration/
pytest -q tests/adapters/  # existing tests, must pass with v2 default LM
# Smoke: a v1 BaseLM subclass works inside Predict
python -c "
import dspy
class FakeV1(dspy.BaseLM):
    def forward(self, prompt=None, messages=None, **kw):
        import types
        c = types.SimpleNamespace(message=types.SimpleNamespace(content='[[ ## answer ## ]]\n42', tool_calls=None), finish_reason='stop')
        r = types.SimpleNamespace(choices=[c], usage={}, model='fake', _hidden_params={})
        return r
lm = FakeV1(model='fake')
dspy.settings.configure(lm=lm)
out = dspy.Predict('q -> answer')(q='?')
print(out)
"
```

**Commit 6 (streaming):**
```bash
pytest -q tests/lm/streaming/
pytest -q tests/streaming/  # existing tests must pass unchanged
```

**Commit 7 (Anthropic + Gemini):**
```bash
pytest -q tests/lm/backends/test_anthropic.py tests/lm/backends/test_gemini.py
# Live tests skipped without credentials
```

**Commit 8 (docs):**
```bash
cd docs && mkdocs build --strict
# Confirm migration guide URL is the one cited in the DeprecationWarning
grep -r "dspy.ai/migration/baselm" dspy/clients/base_lm.py
```

**Full-stack verification (final commit):**
```bash
pytest -q  # entire suite green
pytest -q tests/lm/  # all new-LM tests green
ruff check dspy/ tests/
mkdocs build --strict
# Manual: dspy.LM("openai/gpt-4o-mini") works without LiteLLM installed,
#         under both default and experimental=True modes.
```
