# Flex: Optimizable module code

## Intent

`dspy.Flex` is for tasks where you don't know the right *shape* of the solution up front. Every other module fixes its structure when you construct it — `Predict` is one call, `ChainOfThought` is one call with reasoning, `ReAct` is a tool loop — and optimization only tunes the prompt around that fixed structure. `Flex` moves the structure itself into the search space. You give it a signature; it starts as a `dspy.Predict` or `dspy.RLM` baseline; and `dspy.GEPA` rewrites its entire implementation — how many predictors, which primitives, what runs in Python instead of an LM — against your metric. Reach for it when the best decomposition is something you'd rather discover and optimize.

## Design decisions

### 1. A Flex is a module whose source code is the optimizable parameter

An ordinary module's tunable surface is its predictors' instructions. A `Flex`'s tunable surface is a whole `dspy.Module` subclass, held as a source string and exposed as `module_src`. That class has the usual two methods: `__init__` constructs whatever predictors it needs, and `forward` calls them and returns a `dspy.Prediction`. Optimizing the module means replacing that source with a better version — different predictors, different control flow, more or less Python. The prompt is no longer the unit of optimization; the program code is.

### 2. It drops into any signature and starts from a simple baseline

You construct `Flex` from any `dspy.Signature`, and it's immediately runnable. With no tools, its baseline source is a single `dspy.Predict` over the whole signature; with tools, a `dspy.RLM` so the baseline can call them. The baseline is the simplest thing that works as the starting point of a search.

### 3. GEPA discovers `Flex` by type and optimizes code instead of text

When GEPA compiles a program, it enumerates the `Flex` submodules and splits its work: each `Flex` becomes a **code component**, every other predictor stays an **instruction component**. Code components are seeded with their current `module_src` and evolved by a dedicated code proposer; instruction components are seeded with their current instructions and evolved by GEPA's usual instruction proposer. A custom `instruction_proposer` replaces the instruction proposer only; code components stay on the code proposer.

### 4. The code proposer reflects on whole-program behavior

Instruction optimization in GEPA is per-predictor: it looks at one predictor's inputs, outputs, and feedback. Code optimization can't work that way, because the predictors *are* part of what's being rewritten — they may not exist in the next candidate. So a code component reflects on the whole program's I/O instead: the module's inputs, its final prediction, and the metric's feedback per example. The proposer is given the signature, any available tools, a catalog of allowed primitives, the current source, and a batch of failing examples, and asked to return a full revised module class.

### 5. Predictors inside a Flex are owned by its code, not tuned in parallel

When you mix a `Flex` with ordinary modules in one program, GEPA optimizes the `Flex`'s code and the other predictors' instructions — but never the instructions of predictors that live *inside* the `Flex`. Those predictors are constructed by the current `module_src` and will be replaced wholesale by the next code candidate, so tuning their instructions would be optimizing something that may be overwritten.

`Flex` is a subclass of `Parameter` and `Module` (like `dspy.Predict`), so a parent program's `named_parameters()` yields it as one leaf and never recurses into it. A parent's `named_predictors()`, `predictors()`, `set_lm()`, and demo/instruction optimizers like `BootstrapFewShot` don't see the `Flex`'s internals. The `Flex` reports the same about itself: `flex.named_predictors()` is empty — its update unit is its code, not the predictors the code constructs. A `Flex` can carry its own LM (`flex.set_lm(...)`), which `forward` applies as the ambient LM for the whole call; otherwise its bridged predictor calls resolve the ambient LM at call time (`dspy.configure(lm=...)`, or a caller's `dspy.context(lm=...)`). `reset_copy()` resets a `Flex` the way it resets a `Predict`: the LM is cleared while `module_src` is kept, just as tuned instructions are.

### 6. A broken candidate scores as a failure, not a crash

The reflection model authors code, and code can be wrong. A candidate that doesn't parse raises when GEPA binds it to build the program for evaluation; `Flex` optimization catches that, logs it, and scores the whole batch at the failure score, letting the search continue and simply not select the broken candidate. Code that parses but breaks when it runs — an import that doesn't resolve, an edge case that throws on some inputs — fails per example at `forward` instead: each crashed example is scored at the failure score in its own slot, by example index, so the surviving scores stay aligned to the batch and GEPA's per-instance bookkeeping stays intact. Either way, the optimization is robust to the reflection model's mistakes by construction.

### 7. Generated code always runs in an interpreter, never in-process

`Flex` runs `module_src` in a sandbox: `interpreter_factory` defaults to `dspy.PythonInterpreter` (Deno/Pyodide) and must be a *zero-argument factory* — a bare instance or `None` is rejected. Since the code is authored by the reflection model, it never gets the host's full permissions: everything it does stays in the sandbox except provided-tool calls, predictor construction, and predictor calls, which bridge back to the host. The factory creates a fresh interpreter for each sandbox session; a forward owns an outer session and nested code-executing modules may request separate sessions. Source portability between custom interpreters is not guaranteed. `max_predictor_calls` caps how many predictor calls the generated code can make in one `forward`.

### 8. The declared output types are enforced at the sandbox boundary

Everything the generated code returns crosses back as JSON, so a field declared `int`, `list[str]`, or a pydantic model would arrive as a bare string or dict. `Flex` parses each declared output field against its annotation on the way out, so a `Flex` returns the same types as a `dspy.Predict` over the same signature and stays substitutable for one. Two things follow. Type names have to survive the round trip in *both* directions: the baseline renders custom types into its signature string (`text: str -> person: Person`) and the host resolves those names from the `Flex`'s own signature, because a signature crosses the boundary as text and dspy's usual caller-frame type lookup runs inside dspy, where your module is out of scope. And a candidate whose output doesn't match — wrong shape, or a declared field missing entirely — raises a `CodeInterpreterError` naming the field, which GEPA scores at the failure score for that example rather than handing the metric a `Prediction` that breaks somewhere else. An annotation that can't round-trip through a signature string at all (a `Callable`, say) is emitted untyped rather than rendered into source that wouldn't bind.

### 9. The code is state; the interpreter is a runtime dependency

A saved `Flex` is `{"module_src": ..., "lm": ...}` — the code, plus any LM set directly on the module. The internal predictors are not saved: they are derived from the code, and each `forward` reconstructs them from the bound source. The interpreter is a live runtime resource and is not serialized; `save(path, save_program=True)` cloudpickles the program with the bridge excluded, and loading rebuilds it. Reconstructing with `dspy.Flex(signature)` restores the default sandbox, so you only re-supply the `interpreter_factory` before `load` if you optimized with a custom one.

### 10. Flex is experimental and the interface is in flux

The class carries the `@experimental` decorator. Treat the API and serialization format as subject to change between minor releases and pin a version if you depend on it.

## API walkthrough

### Defining and running a Flex

**`dspy.Flex(signature, *, tools=None, interpreter_factory=PythonInterpreter, max_predictor_calls=100)`**
Parses the signature and binds the baseline source — a single `dspy.Predict` over the signature, or a `dspy.RLM` when `tools` are given. Validates `interpreter_factory` (a zero-arg factory, defaulting to `dspy.PythonInterpreter`), and sets up the sandbox bridge.

**`__call__(**inputs)` / `forward(**inputs)`**
Runs the currently bound source inside the interpreter, bridging predictor calls back to the host. Returns a `dspy.Prediction` over the signature's output fields, and accepts keyword inputs only.

**`module_src`**
A read-only property holding the current implementation as source — one `dspy.Module` subclass. This is the value GEPA reads as the seed and overwrites with each accepted candidate.

### Optimizing with GEPA

**`dspy.GEPA(metric=..., reflection_lm=..., ...).compile(flex_program, trainset=..., valset=...)`**
Compiling a program that contains one or more `Flex` submodules optimizes each `Flex`'s code and every non-flex predictor's instructions together, under one budget. Returns a new program whose `module_src` (per flex submodule) is the best code found. The auto-budget counts each flex submodule as one component alongside the instruction predictors.

**The metric's `feedback`**
As with any GEPA metric, the feedback string goes into the prompt handed to the proposer — here, the *code* proposer. Feedback that diagnoses *why* an output was wrong and hints at structure steers rewrites far better than a bare score.

**Trace-aware metrics (`program_trace`)**
Add `program_trace=None` as a sixth parameter to your metric and GEPA passes the execution trace at scoring time, so you can score against how the answer was produced. For instance, you may use `len(program_trace)` as an LM-call count to fold a small per-call penalty into the score.

### Tools and sandboxing

**`tools=[...]`**
Plain functions or `dspy.Tool` instances, referenced by name in the generated code, so each name must be a valid Python identifier. Providing tools makes the baseline a `dspy.RLM` and tells the code proposer they are in scope — it can wire them into `dspy.RLM`/`dspy.ReAct`, call them directly, or supplement them with its own inline helpers.

**`interpreter_factory=...`**
Defaults to `dspy.PythonInterpreter` (sandboxed, needs Deno). Must be a zero-argument callable returning a fresh `CodeInterpreter` for each sandbox session; parallel evaluations and nested code-executing modules can therefore receive isolated sessions. As in `dspy.RLM`, a bare interpreter instance is not accepted. This low-level hook does not guarantee source or standard-library portability between different interpreters.

**`max_predictor_calls`**
The maximum number of predictor calls the generated code can make in one `forward`. It guards against runaway loops. `None` removes the limit.

### What the generated code can use

The optimizer-authored code does not run against the real `dspy` package. Inside the sandbox, `dspy` is a small shim whose job is to hand predictor construction and predictor calls back to the host — so what it exposes is narrower than the library, and it is the same surface the code proposer is told about.

**Available**

- `dspy.Module` — the base class the generated source subclasses. `__init__` assigns predictors; `forward` runs in the sandbox.
- `dspy.Predict`, `dspy.ChainOfThought`, `dspy.ReAct`, `dspy.ReActV2`, `dspy.RLM` — constructed in the sandbox but *built on the host*, where the real LM calls happen. Constructor kwargs cross as JSON.
- `dspy.Signature("inputs -> outputs", "instructions")` — the string form only, used to give an inner predictor its instructions. It is a marker the host turns back into a real `Signature`, not the class: it has no methods, no `with_instructions()`, and no `dspy.InputField` / `dspy.OutputField` class form.
- `dspy.Prediction(**fields)` — the return value of `forward`. Holds fields; it is not the host `Prediction` type.
- `dspy.Tool(func)` — a pass-through wrapper. Only tools you passed to `dspy.Flex(tools=...)` can be handed to a bridged sub-predictor; they are already in scope by name, so wrapping is optional.
- The Python standard library, imported *inside* `forward` (the interpreter's own sandboxing still applies — `dspy.PythonInterpreter` has no filesystem or network access by default).

**Not available**

- Adapters (`dspy.ChatAdapter`, `dspy.JSONAdapter`, …). The host formats and parses every bridged call; the generated code never sees a prompt or a raw completion.
- `dspy.settings`, `dspy.context`, `dspy.configure`, `dspy.LM`. The LM is whatever the host has configured; generated code cannot select or reconfigure one.
- `dspy.Example`, `dspy.Evaluate`, the optimizers, retrievers, and `dspy.Flex` itself — no nesting.
- Class-based signatures and typed field declarations.
- Host objects generally: values cross as JSON, so a tool you define inside `forward` cannot be passed to a bridged sub-predictor, and a predictor field that isn't JSON-serializable raises rather than silently degrading.

Anything outside this surface fails when the candidate runs, and GEPA scores it at the failure score — a missing name costs a search step rather than crashing the run. The shim lives in `dspy/predict/flex/_sandbox_shim.py` and the proposer-facing version of this list is `dspy/predict/flex/primitives_doc.py`.

### Saving and loading

**`save(path)` / `load(path)` / `dump_state()` / `load_state(state)`**
`module_src` travels in the serialized state (along with any LM set on the module), so loading restores the optimized code; the predictors are rebuilt from it on each `forward`. The interpreter is not serialized; reconstructing with `dspy.Flex(signature)` restores the default sandbox, so re-supply it in the constructor before `load` only if you used a custom `interpreter_factory`.

## Cross-links

- [`dspy.Flex` API reference](../api/modules/Flex.md) — constructor table, examples, and the full method list.
- [Built-in module variants](built-in-module-variants.md) — where `Flex` sits among the other non-`Predict` modules.
- [GEPA in depth](gepa-in-depth.md) — the reflective optimizer that drives `Flex`'s code search.
- [RLM: exploring large contexts with code](rlm.md) — the module `Flex` uses as its tool-enabled baseline.
