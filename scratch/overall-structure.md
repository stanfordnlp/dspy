# DSPy Documentation Rebuild — Overall Structure

Working artifact for the documentation rebuild. Not published.

Output location: `docs/docs/` (replaces existing prose; tutorials/notebooks left in place).

## Selected primary use case

A **haiku-writing program** that grows through the tutorial — chosen because each new feature is motivated by a real shortcoming of the previous step, not bolted on for demonstration.

Reference implementation: `/Users/dbreunig/Development/cmpnd/dspy-doc-redux/haiku_bench/haiku_doc_example.ipynb` plus the JSON-Lines dataset under `haiku_bench/data/` (1,656 train / 552 val / 552 test, each example a `{location, season, mood}` triple). The tutorial mirrors this notebook's progression. We copy the dataset and the metric module into the docs repo (under `docs/docs/getting-started/`) so the tutorial is self-contained.

The reader finishes with a saved program that:
1. Writes a haiku from a string-form signature + `Predict`
2. Grows the signature to accept multiple inputs and inline types
3. Promotes the signature to a class-based form with instructions, `InputField`/`OutputField`, and a `Literal` season
4. Reasons before writing by swapping `Predict` for `ChainOfThought`
5. Grounds its writing in real facts by calling Wikipedia tools through `ReAct`
6. Composes a custom `Module` that pairs the haiku-bot with a titler sub-predictor
7. Is evaluated against a rule-based metric (syllables, line count, present tense, first-person absence, …)
8. Is optimized by `GEPA` against that metric
9. Is re-optimized by `GEPA` against an LLM-judge metric for aesthetic quality
10. Is saved to disk and reloaded

## Getting Started — outline

Single file: `docs/docs/getting-started.md`. Companion assets (dataset, metric module) live under `docs/docs/getting-started/`.

| Step | Title | Components introduced | Notes |
|---|---|---|---|
| 0 | Install & configure an LM | `dspy.LM`, `dspy.configure`, direct-call callout `lm("...")` | One provider in the main flow; alternative-providers tabbed callout (OpenAI / Anthropic / Gemini / Databricks / Ollama / vLLM) |
| 1 | Your first DSPy program | String `Signature` (`"subject -> haiku"`), `dspy.Predict`, the `Prediction` return type | Smallest possible win; show `result.haiku` field access |
| 2 | Add inputs and inline types | Multi-input string signatures (`"location, mood -> haiku"`), inline typed fields (`"location, mood, is_humorous: bool -> haiku"`) | Shows that the string mini-language already has range |
| 3 | Promote to a class-based `Signature` | `class HaikuBot(dspy.Signature)`, docstring-as-instructions, `InputField`, `OutputField`, `Literal["spring", …]` for `season`, the type-mismatch warning | This is the `HaikuBot` from the reference notebook; use the warning on `season="fall"` to motivate types |
| 4 | Swap in reasoning with one line | `dspy.ChainOfThought` | Same signature, different module — the key "modules are interchangeable" beat |
| 5 | Give the program tools | `dspy.ReAct` with plain-Python tool functions: `wikipedia_search(query)`, `get_wikipedia_page(title)` (using the `wikipedia` library; show the trace) | Lift the tool functions from the notebook; demonstrate that any typed Python function is a tool |
| 6 | Compose a custom Module | Subclass `dspy.Module`, define `__init__` + `forward()`, hold sub-predictors, return `dspy.Prediction(...)` | The `HaikuRobot` example: ReAct haiku-bot + a `dspy.Predict("haiku -> title")` titler. This is the "programming-not-prompting" payoff. |
| 7 | Score the baseline | `dspy.Example`, `.with_inputs()`, loading the JSON-Lines splits, a rule-based metric returning `dspy.Prediction(score=..., feedback=...)`, `dspy.Evaluate` | Ship `metric.py` alongside the doc — show its shape and one or two example checks inline, link to the full file. Stress the `(gold, pred, trace=None, pred_name=None, pred_trace=None)` metric signature because GEPA needs it. |
| 8 | Optimize with GEPA (rule metric) | `dspy.GEPA(metric=..., reflection_lm=..., auto="light")`, `.compile(student, trainset=..., valset=...)` | Use a cheaper student LM + a stronger `reflection_lm`. Re-run `Evaluate`. Report the score lift and show one before/after haiku diff. |
| 9 | Upgrade the metric to an LLM judge | A `JudgeHaiku` `dspy.Signature` (`haiku, location, season, mood -> score: float, feedback: str`) wrapped as a metric function returning `dspy.Prediction(score, feedback)` | Motivates GEPA's reflective feedback channel and shows the rule/judge metric duality |
| 10 | Re-optimize with the judge | `dspy.GEPA` again with the judge metric | Compare scores under each metric; discuss cost/time |
| 11 | Save and reload | `program.save(path)`, `dspy.load(path)` | Closes the loop — the optimized program is now a portable artifact |

Each step ends with the smallest possible runnable snippet. The "why" sits in one or two sentences before the code; the "how" is the code itself. The full rule-based metric (~120 lines of spaCy + pyphen) is shipped as a module and referenced rather than inlined.

## Diving Deeper — candidate topics

One file per topic under `docs/docs/diving-deeper/`. The asterisked items are the ones I recommend writing in this pass; the rest are reasonable but cuttable. You decide.

| # | Topic | Recommend? | Why it's DD (not GS / not Ref) |
|---|---|---|---|
| 1 | **Signatures in depth** — class form, typed fields, `Literal`, structured outputs, docstring-as-instructions, `with_updated_fields`, `make_signature`, `ensure_signature` | ★ | Customization test — more than a sentence of intent |
| 2 | **Modules: composing your own** — subclassing `dspy.Module`, multi-step pipelines, sub-predictors, `forward()` semantics, parameter introspection | ★ | Mental-model test — composition is the whole point |
| 3 | **Built-in module variants** — when to reach for `BestOfN`, `Refine`, `MultiChainComparison`, `ProgramOfThought`, `CodeAct`, `Parallel`; how they differ from `Predict`/`ChainOfThought` | ★ | Auxiliary-use-case test — each has its own intent |
| 4 | **Tools, ReAct, and MCP** — wrapping Python functions with `dspy.Tool`, async tools, ToolCalls, how ReAct loops, MCP integration | ★ | Customization test — many surfaces |
| 5 | **Adapters: how signatures become prompts** — Chat vs. JSON vs. XML vs. TwoStep, when each helps, structured-output guarantees, custom type wrappers (Image, Audio, File, Code) | ★ | Mental-model test |
| 6 | **Optimizers: choosing one** — the selection guide. BootstrapFewShot, BootstrapFewShotWithRandomSearch, COPRO, MIPROv2, GEPA, BootstrapFinetune, SIMBA — what each does, what it needs, when it wins | ★ | Mental-model test (this is the single most-asked DSPy question) |
| 7 | **GEPA in depth** — reflection_lm, feedback metrics (`ScoreWithFeedback`), auto modes, predictor-level feedback, custom proposers | ★ | Customization test — covered shallowly in GS |
| 8 | ~~MIPROv2 in depth~~ — *dropped in Phase 3 kickoff* | — | — |
| 9 | **BootstrapFewShot family** — the simplest optimizer, RandomSearch wrapper, LabeledFewShot baseline, KNNFewShot, InferRules | ★ | Customization test |
| 10 | **Fine-tuning with BootstrapFinetune** — when to fine-tune instead of prompt-optimize; data prep; supported providers | ★ | Auxiliary-use-case test |
| 11 | **Metrics and evaluation** — anatomy of a metric, built-in metrics (`EM`, `answer_exact_match`, `SemanticF1`, `CompleteAndGrounded`), LLM-as-judge patterns, `Evaluate` knobs (threads, displaying, return_outputs) | ★ | Customization test |
| 12 | ~~RAG with DSPy~~ — *dropped in Phase 3 kickoff* | — | — |
| 13 | **Settings and `context()`** — global vs. thread-local config, how `dspy.context()` propagates to child threads, mixing async and threads | ★ | Mental-model test |
| 14 | **Caching** — disk + memory caches, `configure_cache`, `rollout_id`, when caching hurts (sampling) | ★ | Customization test |
| 15 | **Saving and loading** — `.save()` / `dspy.load()`, JSON vs. pickle (`allow_pickle`), `load_state`, what's actually serialized | ★ | Customization test |
| 16 | **Async, streaming, and parallel** — `asyncify`/`syncify`, `streamify`, `StreamListener`, `StatusMessageProvider`, `dspy.Parallel` | ★ | Alternative-path test |
| 17 | **Observability and debugging** — `dspy.inspect_history()`, callbacks, usage tracking, logging configuration | ★ | Customization test (debugging surface) |
| 18 | Multimodal inputs (Image, Audio, File) | maybe | Auxiliary-use-case test — could fold into Adapters or stand alone |
| 19 | Ensembles and BetterTogether | maybe | Auxiliary; could fold into Optimizer-choice topic |

**Default plan:** write the 17 starred topics, skip 18–19 unless you want them. We can also batch — Phase 3 will let you re-order or cut.

## Reference — modules to document

One file per public module under `docs/docs/reference/`. Pure API spec — class/function names, signatures, one-or-two-sentence purposes.

| File | Covers |
|---|---|
| `reference/signatures.md` | `Signature`, `InputField`, `OutputField`, `make_signature`, `ensure_signature`, `infer_prefix` |
| `reference/predict.md` | `Predict`, `ChainOfThought`, `ReAct`, `BestOfN`, `Refine`, `MultiChainComparison`, `ProgramOfThought`, `CodeAct`, `Parallel`, `KNN`, `RLM`, `majority` |
| `reference/primitives.md` | `Module`, `BaseModule`, `Example`, `Prediction`, `Completions`, `PythonInterpreter`, `CodeInterpreter`, `SandboxSerializable` |
| `reference/adapters.md` | `Adapter`, `ChatAdapter`, `JSONAdapter`, `XMLAdapter`, `TwoStepAdapter` and the type wrappers (`Image`, `Audio`, `File`, `History`, `Type`, `Tool`, `ToolCalls`, `Code`, `Reasoning`) |
| `reference/clients.md` | `LM`, `BaseLM`, `Embedder`, `Provider`, `inspect_history`, `configure_cache`, `DSPY_CACHE` |
| `reference/teleprompt.md` | Every optimizer: `BootstrapFewShot`, `BootstrapFewShotWithRandomSearch`/`BootstrapRS`, `BootstrapFinetune`, `COPRO`, `MIPROv2`, `GEPA`, `SIMBA`, `Ensemble`, `KNNFewShot`, `LabeledFewShot`, `InferRules`, `BetterTogether`, `AvatarOptimizer` |
| `reference/evaluate.md` | `Evaluate`, `EvaluationResult`, `answer_exact_match`, `answer_passage_match`, `SemanticF1`, `CompleteAndGrounded`, `EM`, `normalize_text` |
| `reference/retrievers.md` | `Retrieve`, `Embeddings`, `EmbeddingsWithScores`, `ColBERTv2` |
| `reference/streaming.md` | `streamify`, `StreamListener`, `StatusMessage`, `StatusMessageProvider`, `streaming_response` |
| `reference/utils.md` | `asyncify`, `syncify`, `track_usage`, `load`, `configure_dspy_loggers`, `disable_logging`, `enable_logging`, `ContextWindowExceededError` |
| `reference/settings.md` | `configure`, `context`, `settings` singleton, `load_settings` |

## Open decisions for you

1. Are the 17 starred Diving Deeper topics the right cut? Anything to drop, add, merge, or reorder?
2. The Getting Started arc has 10 steps — is that the right level of ambition? Could be tightened to ~7 by collapsing 5+6 and 8+9.
3. Reference is currently 11 files. Acceptable, or do you want to merge (e.g., fold `settings.md` into `clients.md`)?
4. After approval, suggest clearing context before starting Phase 2 (Getting Started). This file is the durable handoff.
