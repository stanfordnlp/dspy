# Deferred topics — sidecar for Phase 2

Digressions noticed while drafting Getting Started. Each entry names the temptation and where it's expected to be covered.

## Reviewed against `overall-structure.md`'s Diving Deeper plan

Every digression below maps cleanly to a DD topic already on the list. No new DD candidates surfaced from this phase.

| Digression | Where it appeared in GS | Lands in |
|---|---|---|
| `dspy.context(lm=...)` for per-call LM overrides | §10 metric uses it without explanation | DD: "Settings and `context()`" |
| ReAct `max_iters`, tool-call truncation | §6 doesn't mention iteration control | DD: "Tools, ReAct, and MCP" |
| `dspy.Tool` (explicit wrapping vs plain Python functions) | §6 treats tools as plain functions | DD: "Tools, ReAct, and MCP" |
| Cache behavior, `rollout_id` | Implicit throughout; never explained | DD: "Caching" |
| LM config knobs (temperature, max_tokens, retries, api_base, model_type) | §1 names `dspy.LM` but skips the knobs | Reference: `clients.md` (mostly lookup-grade) |
| Adapter layer (Chat/JSON/XML/TwoStep) | Never mentioned in GS | DD: "Adapters" |
| Other optimizers (BootstrapFewShot, MIPROv2, COPRO, BootstrapFinetune) | §8/§9 name in passing only | DD: "Optimizers: choosing one" (+ per-optimizer DD topics) |
| Other predict modules (BestOfN, Refine, MCC, ProgramOfThought, CodeAct, Parallel) | Never mentioned in GS | DD: "The predictor zoo" |
| Structured output beyond `Literal` (Pydantic models as fields) | §3/§4 only show str/bool/Literal | DD: "Signatures in depth" |
| `dspy.inspect_history()` deeper usage | §2 mentions it; nothing more | DD: "Observability and debugging" |
| Callbacks | Not mentioned | DD: "Observability and debugging" |
| Async / streaming for ReAct or modules | Not mentioned | DD: "Async, streaming, and parallel" |
| `Example` and `Prediction` internals (fields, copying, with_inputs semantics) | §7/§8 use them; minimal explanation | DD: "Signatures in depth" or a new minor sub-topic |

## Conclusion

`overall-structure.md` does not need changes. Phase 2 closed cleanly with the existing DD plan intact.
