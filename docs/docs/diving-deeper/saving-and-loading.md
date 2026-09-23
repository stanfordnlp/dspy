# Saving and loading

## Intent

Saving a DSPy program preserves what an optimizer produced: the rewritten signature instructions, the few-shot demos, the LM config, and whatever state each predictor needs to recreate its behavior. There are two paths — save the state alone and re-load it into a freshly-instantiated program, or pickle the whole program so the loading side doesn’t need your class definitions at all.

Read this when you’re shipping an optimized program to another team, version-controlling optimizer output, or trying to figure out why a `.json` save round-trips but a `.pkl` save needs an extra flag to load.

## Design decisions

### 1. Two paths: state-only and full-program

State-only saves the optimizer’s work — demos, signature instructions, LM config — and assumes the loading side has your class definitions. Full-program saves the whole module via cloudpickle, so the loader doesn’t need to import your code. Use state-only by default; reach for full-program when shipping to a process that doesn’t have your source.

### 2. JSON by default

State-only saves to JSON when the path ends in `.json`. Human-readable, diff-able in version control, no code-execution risk on load. The `.pkl` state form is there for state that doesn’t serialize cleanly to JSON (custom Pydantic objects in demos, for instance), but the JSON path is the one you’ll reach for almost every time.

### 3. `save_program=True` exists for the “no class on the loading side” case

When the loader doesn’t have your `class HaikuEnsemble(dspy.Module)` definition available — a different repo, a different team, a different service — full-program mode bundles the class with the state. The loader calls `dspy.load(path)` and gets a usable module back without ever importing your code. The price is cloudpickle, which means the load path must trust the file.

### 4. `allow_pickle` is deliberate friction

Loading any pickle — state-PKL or full-program — requires `allow_pickle=True`. The default is `False` so an unsuspecting caller can’t deserialize an arbitrary file and silently execute code. The flag is a forcing function: it makes the trust decision explicit at the call site, not buried in defaults.

### 5. `allow_unsafe_lm_state` strips endpoints by default

When loading state, three LM-config keys are dropped unless you opt in: `api_base`, `base_url`, and `model_list`. The reasoning: a saved program might point at an internal endpoint that the loading side shouldn’t be talking to, or that’s no longer reachable. Pass `allow_unsafe_lm_state=True` when you want the original endpoint configuration to ride along.

### 6. API keys are never serialized

`LM.dump_state` explicitly excludes `api_key` from the saved kwargs, and there’s no flag to re-enable it. The LM client always needs its credentials configured fresh on the loading side. Anything else would be a credential leak waiting to happen.

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

**`Module.load(path, allow_pickle=False, allow_unsafe_lm_state=False)`**  
Loads state into an existing module instance. You instantiate the program the same way you built it, then call `.load()` on it. JSON paths load freely; `.pkl` paths require `allow_pickle=True`. `allow_unsafe_lm_state=True` keeps `api_base`, `base_url`, and `model_list` instead of stripping them.

```python
program = HaikuEnsemble(n=5)        # same construction as when saved
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

**`Module.load_state(state, *, allow_unsafe_lm_state=False)`**  
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
On state load, drops `api_base`, `base_url`, and `model_list` from the LM config. Pass `True` to restore the original endpoint configuration. The split exists because a saved program’s endpoint may not match where the loader wants to run — an internal-only URL, a deprecated host, a model list that’s no longer available.

API keys are never re-enabled by either flag. The loading side configures credentials fresh.

### Metadata and versioning

Every save writes dependency versions: Python, DSPy, cloudpickle. Full-program saves put them in `metadata.json` next to `program.pkl`; state-only saves embed them under a `"metadata"` key in the JSON / PKL state.

On load, the runtime compares versions to the current process. A mismatch logs a warning and proceeds. The save format aims for backward compatibility, so old saves load against newer DSPy versions — but a warning is your hint to check the release notes if something behaves differently.

## Cross-links

- [Modules: composing your own](modules.md) — `Module.save` / `load` are inherited from `BaseModule`; the tree-walk that gathers state is the same one optimizers use.
- [Signatures in depth](signatures-in-depth.md) — `Signature.dump_state` / `load_state` are what carry an optimizer’s rewritten instructions through a round-trip.
- [Settings and `context()`](settings-and-context.md) — `dspy.settings.save` / `dspy.load_settings` are a parallel surface for the settings singleton.
