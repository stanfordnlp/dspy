# Saving and loading

## Intent

Saving a DSPy program preserves what an optimizer produced: the rewritten signature instructions, the few-shot demos, the LM config, and whatever state each predictor needs to recreate its behavior. There are two paths — save the state alone and re-load it into a freshly-instantiated program, or pickle the whole program so the loading side doesn’t need your class definitions at all.

Read this when you’re shipping an optimized program to another team, version-controlling optimizer output, or trying to figure out why a `.json` save round-trips but a `.pkl` save needs an extra flag to load.

The examples below use this two-predictor program:

```python
import dspy


class HaikuEnsemble(dspy.Module):
    def __init__(self):
        super().__init__()
        self.draft = dspy.ChainOfThought("topic -> haiku")
        self.critique = dspy.Predict("haiku -> improved_haiku")

    def forward(self, topic):
        draft = self.draft(topic=topic)
        return self.critique(haiku=draft.haiku)
```

## Design decisions

### 1. Two paths: state-only and full-program

State-only saves the optimizer’s work — demos, signature instructions, LM config — and assumes the loading side has your class definitions. Full-program saves the whole module via cloudpickle, so the loader doesn’t need to import your code. Use state-only by default; reach for full-program when shipping to a process that doesn’t have your source.

### 2. JSON by default

State-only saves to JSON when the path ends in `.json`. Human-readable, diff-able in version control, no code-execution risk on load. The `.pkl` state form is there for state that doesn’t serialize cleanly to JSON (custom Pydantic objects in demos, for instance), but the JSON path is the one you’ll reach for almost every time.

### 3. `save_program=True` exists for the “no class on the loading side” case

When the loader doesn’t have your `class HaikuEnsemble(dspy.Module)` definition available — a different repo, a different team, a different service — full-program mode bundles the class with the state. The loader calls `dspy.load(path)` and gets a usable module back without ever importing your code. The price is cloudpickle, which means the load path must trust the file.

### 4. `allow_pickle` is deliberate friction

Loading any pickle — state-PKL or full-program — requires `allow_pickle=True`. The default is `False` so an unsuspecting caller can’t deserialize an arbitrary file and silently execute code. The flag is a forcing function: it makes the trust decision explicit at the call site, not buried in defaults.

### 5. Endpoint configuration in saved state refuses to load by default

When loading state, LM-config keys that carry endpoint routing — `api_base`, `base_url`, `model_list`, and the `engine` block — cause the load to fail with a typed `dspy.LMStateError` unless you opt in. The reasoning: a tampered file could reroute requests (and the prompts inside them) to an attacker's endpoint, and silently dropping the keys instead would reroute a local-server program to the provider's default endpoint — working, but differently. The error names both exits, and both are fully supported:

**The trusted path — "just works", any number of LMs.** For a file you trust (you saved it, or it comes from a source you control), pass `allow_unsafe_lm_state=True`. Every saved LM — however many the program has — is reconstructed exactly as saved: same endpoints, same engine configuration, same declared capabilities. This is the right call for your own checkpoints and optimization artifacts.

```python
program = HaikuEnsemble()
program.load("haiku_ensemble.json", allow_unsafe_lm_state=True)
```

**The safe path — routes come from your code.** For a file you don't fully trust, supply the LMs yourself with `lm=`. A single LM applies to every predictor; a dict keyed by predictor name (the names `program.named_predictors()` returns) covers multi-LM programs. Predictors you leave out of the dict load their saved LM state normally, so you only need to supply LMs for the ones that carried endpoint configuration.

```python
program = HaikuEnsemble()
program.load(
    "haiku_ensemble.json",
    lm={
        "draft.predict": dspy.LM("openai/local-model", api_base="http://localhost:8000/v1"),
        "critique": dspy.LM("anthropic/claude-sonnet-5"),
    },
)
```

This is deliberately manual: the safe path's whole point is that endpoint routes are typed into your code, where you can read them, not carried by a file you didn't write.

### 6. API keys and call history are never serialized — on either path

`LM.dump_state` explicitly excludes `api_key` from the saved kwargs, and there’s no flag to re-enable it. The full-program pickle path applies the same hygiene: while `save_program=True` pickles the module, LMs are serialized with credentials (string keys, credential callables, sensitive kwargs like `azure_ad_token`, and sensitive headers like `Authorization`) and `lm.history` scrubbed. The LM always needs its credentials configured fresh on the loading side. Anything else would be a credential leak waiting to happen — a saved artifact travels, and everything embedded in it travels too.

This scrubbing applies only to saved artifacts: in-process `copy.deepcopy` and `lm.copy()` (which optimizers rely on) keep working credentials and history.

### 7. `load_state` is transactional

When you call `Module.load_state(state)`, DSPy runs the load against a deep copy of the module first, and only commits the change to the live module if the trial succeeded. If a state file is corrupt or incompatible, the original module is left untouched — no half-loaded state, no module stuck between two configurations.

### 8. Callbacks and history are excluded from saved state

Both are runtime-only: callbacks are hooks the caller registers per process, and history is a growing log of LM calls. They’re dropped in `__getstate__` so they don’t ride along with pickled programs and don’t bloat state files. After loading, re-register callbacks if you need them; history starts fresh.

### 9. Metadata sits in a separate file

When you save a full program, the directory contains `program.pkl` and `metadata.json`. The metadata file holds dependency versions (Python, DSPy, cloudpickle) so you can read them without unpickling. State-only JSON saves embed the same metadata under a `"metadata"` key in the JSON dict.

### 10. Version mismatches warn, don’t block

If you load a program saved under an older DSPy, you get a warning logged with the version delta — but the load proceeds. The save format aims for backward compatibility, and a hard version check would force users to keep stale virtualenvs around to load old programs.

### 11. `modules_to_serialize` embeds user-defined classes by value

By default, cloudpickle serializes user classes by import path (`mymodule.MyClass`). If the loading side doesn’t have `mymodule` importable, the load fails. Passing `modules_to_serialize=[MyClass.__module__]` (or the module object) registers it with `cloudpickle.register_pickle_by_value`, embedding the class code in the pickle. Useful when you’re saving a program defined in a script rather than a package.

## API walkthrough

Grouped by what you’re trying to do.

### Saving

Three call shapes; the path’s suffix (or lack of one) chooses the mode.

**`Module.save("path.json")`**  
State-only JSON. Writes `Module.dump_state(json_mode=True)` plus a `metadata` block of dependency versions. Pretty-printed via `orjson`. Diff-friendly.

**`Module.save("path.pkl")`**  
State-only PKL. Cloudpickle of the same state dict. Use when the state contains objects that don’t round-trip through JSON. Logs a warning at save time noting that loading requires `allow_pickle=True`.

**`Module.save("path/", save_program=True)`**  
Full program. Writes two files into the directory: `program.pkl` (cloudpickle of `self`) and `metadata.json` (dependency versions). The path must be directory-shaped — passing a `path.suffix` raises an error. Creates the directory if it doesn’t exist.

**`modules_to_serialize=[...]` (full-program mode only)**  
Registers each entry with `cloudpickle.register_pickle_by_value` before pickling. Pass the modules that define your custom `Module` subclasses; otherwise the pickle stores them by import path, which breaks when the loader can’t import them.

### Loading

Two entry points, matching the two save modes.

**`Module.load(path, allow_pickle=False, allow_unsafe_lm_state=False, lm=None)`**  
Loads state into an existing module instance. You instantiate the program the same way you built it, then call `.load()` on it. JSON paths load freely; `.pkl` paths require `allow_pickle=True`. If the saved LM state carries endpoint configuration (`api_base`, `base_url`, `model_list`, `engine`), the load raises `dspy.LMStateError` unless you pass `allow_unsafe_lm_state=True` (trusted file) or `lm=` (a single LM for every predictor, or a dict of predictor names to LMs; matching predictors ignore their saved LM state).

```python
program = HaikuEnsemble()           # same construction as when saved
program.load("haiku_ensemble.json") # state slots in
```

**`dspy.load(path, allow_pickle=False)`**  
Loads a full-program directory and returns the rehydrated module. No prior instantiation needed — cloudpickle reconstructs the object graph. Always requires `allow_pickle=True` in practice (pickle is the only loading path here); the flag is the user’s acknowledgment that the directory is trusted.

```python
program = dspy.load("haiku_ensemble/", allow_pickle=True)
```

### Underlying state surface

You rarely call these directly — `Module.save` / `Module.load` are the user-facing pair — but knowing what they round-trip helps when debugging a state file.

**`Module.dump_state(json_mode=True)` → `dict`**  
Returns `{name: parameter.dump_state(...)}` for every named parameter in the tree. `json_mode=True` (the default) forces JSON-serializable shapes; `False` allows pickle-only objects through (used internally by the `.pkl` save path).

**`Module.load_state(state, *, allow_unsafe_lm_state=False, lm=None)`**  
Applies a state dict. Runs the load on a deep copy first to validate, then commits to the live module. Failure on the trial leaves the live module unchanged.

**`Predict.dump_state(json_mode=True)`**  
A single predictor’s state: `{"traces": [...], "train": [...], "demos": [...], "signature": {...}, "lm": {...}}`. Demos are serialized via a `serialize_object` helper that recursively converts Pydantic objects to plain dicts.

**`Signature.dump_state()`**  
`{"instructions": str, "fields": [{"prefix": str, "description": str}, ...]}`. The instructions are the docstring — what optimizers like GEPA rewrite. Field metadata (prefix, description) round-trips too; field names and types are reconstructed from the live Signature class on load.

**`LM.dump_state()`**  
Model name, `model_type`, cache flag, retry count, kwargs (minus `api_key`), and finetuning-related fields. The omission of `api_key` is hard-coded; there’s no flag to opt back in.

### Security flags

Two flags, two different concerns.

**`allow_pickle=False` (default)**  
Refuses to load any `.pkl` or full-program directory. Loading a pickle can execute arbitrary code; the flag forces the caller to acknowledge that the file is trusted. Applies to both `Module.load` and `dspy.load`.

**`allow_unsafe_lm_state=False` (default)**  
On state load, refuses LM config containing `api_base`, `base_url`, `model_list`, or an `engine` block, raising a typed `dspy.LMStateError`. Pass `True` to honor the original endpoint configuration from a trusted file, or pass `lm=` to supply the route from code instead. The refusal exists because a saved program's endpoint may not be one the loader should talk to — and because a program must never load "successfully" onto a different endpoint than it was saved with.

API keys are never re-enabled by either flag. The loading side configures credentials fresh.

### Metadata and versioning

Every save writes dependency versions: Python, DSPy, cloudpickle. Full-program saves put them in `metadata.json` next to `program.pkl`; state-only saves embed them under a `"metadata"` key in the JSON / PKL state.

On load, the runtime compares versions to the current process. A mismatch logs a warning and proceeds. The save format aims for backward compatibility, so old saves load against newer DSPy versions — but a warning is your hint to check the release notes if something behaves differently.

## Cross-links

- [Modules: composing your own](modules.md) — `Module.save` / `load` are inherited from `BaseModule`; the tree-walk that gathers state is the same one optimizers use.
- [Signatures in depth](signatures-in-depth.md) — `Signature.dump_state` / `load_state` are what carry an optimizer’s rewritten instructions through a round-trip.
- [Settings and `context()`](settings-and-context.md) — `dspy.settings.save` / `dspy.load_settings` are a parallel surface for the settings singleton.
