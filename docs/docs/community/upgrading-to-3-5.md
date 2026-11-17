# Upgrading from DSPy 3.4.1 to 3.5

**Re-evaluate optimized programs.** This release changes prompts, defaults and integration APIs, even when your Signature stays the same.

Start with the sections that affect you:

- All programs: [XML output](#xml-is-the-default-adapter), [reasoning](#reasoning-and-explicit-plans), [API routing](#api-routing-defaults-to-auto).
- Agents: [ReAct](#reactv2-replaces-react), [RLM History and interpreters](#rlm-history-and-interpreters).
- Integrations: [removed APIs](#removed-apis), [custom engines and adapters](#custom-engines-and-adapters), [dependencies and embeddings](#dependencies-and-embeddings).
- Cost and persistence: [caching](#caches-regenerate-and-load-restrictively), [candidates and accounting](#candidates-and-accounting).

## Changed defaults and prompts

### XML is the default adapter

**All predictor users:** the default is now `XMLAdapter(use_native_function_calling=True)`, not `ChatAdapter`. Prompts, demonstrations, output instructions and parsing change even with the same Signature. Re-evaluate quality and rebuild format-dependent examples or fixtures.

To retain Chat field delimiters, configure `dspy.ChatAdapter()` explicitly; this does not undo the other changes below. Explicit `dspy.XMLAdapter()` still defaults native function calling **off**.

**Existing XML users:** scalar values are now literal text, not XML-escaped text. Review custom completions/parsers, nullable fields and nested schemas: `<value/>` means null for a nullable field; `<value></value>` means an empty string. Use concrete annotations instead of ambiguous `Any` or string/container unions. There is no one-line replacement that guarantees unchanged quality.

### Reasoning and explicit plans

**CoT/RLM users:** native reasoning is preferred; the fallback requests a short plan. CoT's `reasoning` is now `dspy.Reasoning` or `None`, not always a string. Update string-based metrics and feedback consumers.

Request explicit text when the task benefits from planning before acting or needs a human-auditable plan:

```python
# Before: default reasoning was explicit text
cot = dspy.ChainOfThought("question -> answer")

# After: opt into an explicit text field
cot = dspy.ChainOfThought("question -> answer", rationale_field_type=str)
```

This opts into text, not the old step-by-step prompt. Re-evaluate saved demos and reasoning-dependent optimization.

### API routing defaults to auto

**Chat-only endpoints, proxies and Azure deployments:** `model_type` defaults to `"auto"`, which selects Responses for OpenAI/Azure. This applies even with `engine="litellm"`; engine and API-family choices are independent.

```python
# Before: implicit Chat in 3.4.1
lm = dspy.LM("openai/gpt-4o")

# After: pin a Chat-only route
lm = dspy.LM("openai/gpt-4o", model_type="chat")
```

Saved explicit `"chat"` stays Chat. Auto string calls still return lists of strings; explicit `"responses"` retains its dictionary convention. Routing is chosen before inference, not retried through another backend after a provider failure.

## Breaking changes and removals

### Removed APIs

| Removed | Use instead |
| --- | --- |
| `lm(messages=...)`, `BaseLM.forward` / `aforward`, `LegacyEngine` / `AsyncLegacyEngine`, `complete_legacy` and legacy LM response-processing hooks | Canonical `Request` / `Response` engines; see below. |
| `Tool.format_as_litellm_function_call()` | `Tool.as_function_tool()` |
| `CodeAct`, `ProgramOfThought` | Rebuild with `RLM` and re-optimize; old saved programs have no compatibility alias. |
| `Image.from_file()`, `Audio.from_file()` | `from_path()` |
| `Image.from_PIL()` | `Image(pil_image)` |
| Image constructor `download` / `verify` options | Explicit `from_path()` / `from_url()`, or a non-downloading `Image(url)` reference. |
| Experimental `TypeSafe` client/module/extra | `dspy.LM("typesafe:jev-latest", ...)`; retain your model, endpoint and timeout explicitly. |
| Bundled lm15 callable-tool derivation helpers (`tool`, `derive_tool`, related config/result types) | `FunctionTool` schemas plus callable execution in your application. `dspy.Tool` remains available. |

### Custom engines and adapters

**Integration authors:** implement `complete(Request) -> Response`, or `async def complete` for an async engine. There is no new `generate()` API. Import canonical types from bundled `dspy.lm15`, not a separately installed lm15 package.

- `Adapter.format()` returns `dspy.adapters.Prompt`, not OpenAI message dictionaries.
- `dspy.Type.format()` returns text/content parts; `parse_lm_response` receives a canonical `Response`.
- Custom streams must emit canonical events, not raw LiteLLM chunks. Complete-only engines need no streaming methods; sync/async need their own engines.
- Put settings in `Request.config` for direct typed calls. These calls return one `Response` and do not inherit LM generation defaults; string calls still return lists.

Migrate engines, adapters, custom types and raw-history consumers together. See the [LM migration contract](normalized-lm-api-migration.md) and [custom engine examples](../tutorials/custom_lm_engines/index.md).

### ReActV2 replaces ReAct

**Existing ReAct users:** `dspy.ReAct` now uses structured History and a typed `submit` tool, not a trajectory dictionary plus extractor.

```python
# Before
trace = result.trajectory

# After: a different representation
messages = result.history.messages
termination = result.termination_reason
```

Recompile old optimized agents. Multiple tools run sequentially per turn; `max_iters` counts turns, not calls. Failed final submission raises, and automatic trajectory truncation is gone.

Rename conflicting inputs `history`, `tools`, `turn_budget`; outputs `history`, `termination_reason`; or a tool named `submit`. See the [ReAct migration details](../diving-deeper/react.md#migration-to-dspy-35).

### Dependencies and embeddings

**Users relying on transitive SDKs:** LiteLLM and the OpenAI SDK are optional. Install `dspy[litellm]` for compatibility routes or `dspy[openai]` for SDK-dependent features.

**Hosted embedding users:** install `dspy[numpy,openai]`. Embedder now uses the OpenAI SDK for OpenAI, Azure or compatible endpoints; other protocols need a callable. Installing LiteLLM or registering an LM provider does not restore embedding routing. Preserve model, dimensions, normalization and query/document modes when keeping an index.

Embedding caches now distinguish endpoint/API-version/organization settings. Explicit base64 outputs are decoded, rows are sorted by index, and uncached batches close their SDK clients. See [Embedder](../api/models/Embedder.md).

### Provider options, media and streams

**Provider-specific integrations:** compatibility calls now pass canonical validation too. Replace deprecated `functions` / `function_call` shapes with tools; per-tool `strict=True` is rejected. Strict response schemas remain supported. Unknown options must be JSON-safe; `None` removes a default. Review aliases and adaptations in [LM migration](normalized-lm-api-migration.md#behaviour-changes-worth-knowing).

Review media/citation assumptions: native Chat supports text/images; Responses supports files, not audio. File names and native citation offsets/indexes are not preserved; Document no longer automatically enables Anthropic citations. Inline file data needs a base64 data URI. Use `provider_data` for raw telemetry. Unsupported modalities may be refused rather than dropped.

Move OpenRouter `models`/`route` into generation options; inference provider/privacy restrictions belong in predictor config, not LM constructor `provider=`. Text endpoints flatten roles and reject media. LiteLLM text/Responses calls inside `streamify` can complete without token chunks; use native Responses for token streaming.

## State, costs and runtime boundaries

### RLM History and interpreters

**RLM users:** ordered History replaces flat action context. Re-evaluate saved internal predictors and prompt fixtures. Continue conversations using returned History:

```python
saved = result.history.model_dump_json()
history = dspy.History.model_validate_json(saved)
result = rlm(question="What changes next?", history=history)  # Supply other required inputs too.
```

History replays conversation evidence; it does **not** restore live Python variables. The public factory creates an interpreter per invocation and shuts it down afterward. Startup snapshots are prepared environments, not saved user sessions. Input History is not mutated.

RLM reserves `history`, `repl_entry`, `variables_info`, `iteration` as input/output names, plus input `history_processor`. REPL text rendering and saved entry tags changed; display truncation does not redact stored History.

**Custom interpreters/Flex users:** guests must import real DSPy, replacing the host predictor shim. Flex's `max_predictor_calls` now counts LM calls, including nested calls/candidates. Review budgets and guest dependencies.

**Image users:** typed Image inputs and mixed text/Image subqueries are supported; arbitrary user classes/methods and nested image JSON are not automatically reconstructed in the guest. Guest packages are separate from host packages; `sandbox_packages=[]` omits Pillow. ReAct and RLM are not drop-in replacements: check tools, outputs, modalities and runtime requirements.

### Caches regenerate and load restrictively

**Persistent-cache users:** ordinary old entries and typed v1 records miss under canonical v2. Budget recomputation. Restricted disk loading is now the default; unlisted custom types and old NumPy pickles regenerate. Plain arrays use data-only storage; object arrays/subclasses are not persisted in restricted mode.

Keep restrictions enabled and allow only trusted `safe_types`. LM credentials/endpoints still do not partition cache keys; separate cache/rollout identities when results depend on them. Sync/async caches remain separate. Windows directory protections remain best-effort; load only trusted pickle checkpoints.

### Candidates and accounting

**Latency/cost-sensitive users:** `n` candidates now use sequential single-response calls on every backend, with seed + candidate index. Budget repeated input tokens. Adapter candidates cache independently; direct-string `n` remains grouped, without partial caching on failure.

Canonical Usage totals include cached/tool-use input and reasoning/audio output. Stop adding those counters again. Low-level `InferencePricing.estimate` takes `Usage`; unknown cost remains `None`, not zero or a partial estimate. Review manually constructed Usage and pricing integrations.

`num_retries=0` disables LM retries, not adapter fallback: Chat/XML can make an extra JSON-format call after a parse failure. Set `use_json_adapter_fallback=False` when a strict call budget requires it.

## Deprecations

`ReActV2` and `TwoStepAdapter` remain available in 3.5, deprecated for removal in 3.6. Switch to `dspy.ReAct`; replace TwoStepAdapter with `JSONAdapter()` or `XMLAdapter()` and re-evaluate its former extraction-LM pipeline.

## Other changes to review

- **Optimization/data:** `metric_threshold=0` now accepts scores ≥ 0; use `None` for metric truthiness. `seed=0`, callable-tool demo sampling and failed-CoT scoring are fixed. Rebaseline demos/scores. Majority can return non-strings, pandas integers retain precision, CSV exports include all fields, and hidden-only Example copies retain data.
- **Tool evidence:** new serialized ToolCalls/History retains IDs/results/continuations; previously lost IDs are not repaired. Replay retains errors, and nested schemas, date RootModels, sandbox null/defaults and async results are corrected. Review code relying on old coercions.
- **Low-level lm15 users:** the bundle moves from 1.0.0rc3 to patched 2.2.0. Set explicit Claude output caps and SSE limits where needed; omitted caps increase and SSE caps become unbounded. Review structured `Response.text`, stateful stream readers, stricter errors/schema/options and changed subscription/Vertex credential resolution. Pin the intended billing project; ordinary inference does not initiate login. See the [LM contract](normalized-lm-api-migration.md).
- **Install constraints:** base `certifi>=2026.10.19`; development/CI `uv>=0.13.4`; GEPA `>=0.1.4,<0.2.0`.

Reviewed range: [3.4.1 → 471b3fb0](https://github.com/stanfordnlp/dspy/compare/3.4.1...471b3fb0e4efad45dbac980e42b1258f55a245cf). This guide does not cover unmerged proposals or deferred live-interpreter/session APIs.
