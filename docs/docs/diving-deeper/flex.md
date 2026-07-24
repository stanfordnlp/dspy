# Flex: LLM-generating a module's code

## Intent

`dspy.Flex` is for tasks where you don't know the right *shape* of the solution up front. Every other module fixes its structure when you construct it — `Predict` is one call, `ChainOfThought` is one call with reasoning, `ReAct` is a tool loop — and optimization only tunes the text around that fixed structure. `Flex` moves the structure itself into the search space. You give it a signature; it starts as a trivial baseline; and `dspy.GEPA` rewrites its entire implementation — how many predictors, which primitives, what runs in Python instead of an LM — against your metric. Reach for it when the best decomposition is something you'd rather discover and optimize.

## Design decisions

### 1. A Flex is a module whose source code is the optimizable parameter

An ordinary module's tunable surface is its predictors' instructions. A `Flex`'s tunable surface is a whole `dspy.Module` subclass, held as a source string and exposed as `module_src`. That class has the usual two methods: `__init__` constructs whatever predictors it needs, and `forward` calls them and returns a `dspy.Prediction`. Optimizing the module means replacing that source with a better version — different predictors, different control flow, more or less Python. The prompt is no longer the unit of optimization; the program is.

### 2. It drops into any signature and starts from a simple baseline

You construct `Flex` from the same signature you'd give `Predict`, and it's immediately runnable. With no tools, its baseline source is a single `dspy.Predict` over the whole signature; with tools, a single `dspy.RLM` so the baseline can call them. The baseline is deliberately the simplest thing that works — one call, no decomposition — because it's meant to be the starting point of a search, not the answer. This is what lets you adopt `Flex` by changing one line and revert just as cheaply.

### 3. GEPA discovers the marker and optimizes code instead of text

`Flex` carries a `_code_optimizable` flag. When GEPA compiles a program, it enumerates the flex submodules by that marker and splits its work: each `Flex` becomes a **code component**, every other predictor stays an **instruction component**. Code components are seeded with their current `module_src` and evolved by a dedicated code proposer; instruction components are seeded with their current instructions and optimized using GEPA. Only module instances of `Flex` can have their code optimized by GEPA.

### 4. The code proposer reflects on whole-program behavior

Instruction optimization in GEPA is per-predictor: it looks at one predictor's inputs, outputs, and feedback. Code optimization can't work that way, because the predictors *are* part of what's being rewritten — they may not exist in the next candidate. So a code component reflects on the whole program's I/O instead: the module's inputs, its final prediction, and the metric's feedback per example. The proposer is given the signature, any available tools, a catalog of allowed primitives, the current source, and a batch of failing examples, and asked to return a full revised module class.

### 5. Predictors inside a Flex are owned by its code, not tuned in parallel

When you mix a `Flex` with ordinary modules in one program, GEPA optimizes the `Flex`'s code and the other predictors' instructions — but never the instructions of predictors that live *inside* the `Flex`. Those predictors are constructed by the current `module_src` and will be replaced wholesale by the next code candidate, so tuning their instructions would be optimizing something about to be overwritten. `Flex` excludes them from the instruction-component set by object identity, keeping the two optimization surfaces from fighting each other.

### 6. A broken candidate scores as a failure, not a crash

The reflection model authors code, and code can be wrong — a syntax error, a missing method, an import that doesn't resolve. Binding a candidate happens when GEPA builds the program to evaluate it, so a bad candidate raises there. `Flex` optimization catches that, logs it, and scores the whole batch at the failure score, letting the search continue and simply not select the broken candidate. The optimization is robust to the reflection model's mistakes by construction.

### 7. Scores stay aligned to the batch even when a candidate crashes at runtime

A code candidate can bind cleanly and still raise mid-`forward` on some inputs — the model wrote code that works for most examples and throws on an edge case. Those examples drop out of trace capture, so `Flex` rebuilds the score list to full length by example index, scoring each dropped example at the failure score in its own slot. Without this, a gap would shift every later score onto the wrong example and corrupt GEPA's per-instance bookkeeping. Runtime-fragile candidates are penalized honestly, not silently misattributed.

### 8. Generated code always runs in an interpreter, never in-process

`Flex` runs `module_src` in a sandbox: the `interpreter_factory` defaults to `dspy.PythonInterpreter` (Deno/Pyodide), matching `dspy.RLM`, and — like `dspy.RLM` — must be a *zero-argument factory* (a bare instance or `None` is rejected). Since the code is authored by the reflection model, isolating it keeps it from running with the host's full permissions: the optimizer-authored glue runs isolated, and only predictor construction and predictor calls bridge back to the host to make real LM calls. Sandbox sessions are stateful and not thread-safe, so the factory is called per rollout and each parallel evaluation gets its own; pass your own factory to customize the sandbox (grant filesystem/network access, or use another backend). `max_predictor_calls` caps bridged LM calls per `forward` as a runaway guard.

### 9. The code is state; the interpreter is a runtime dependency

`module_src` is part of the module's serialized state, so saving an optimized program and loading it elsewhere restores the discovered code and its predictors. The interpreter is treated like the LM: a live runtime resource that isn't serialized. Reconstructing with `dspy.Flex(signature)` restores the default sandbox, so you only re-supply the `interpreter_factory` before `load` if you optimized with a customized one. This mirrors how DSPy handles LMs — configuration travels with the program, live resources are wired up at load time.

### 10. Flex is experimental and the interface is in flux

The class carries the `@experimental` decorator. The moving parts — the code proposer's prompt, the sandbox bridge, trace-aware scoring, the failure-handling contracts — are still settling. Treat the API as subject to change between releases and pin a version if you depend on it.

## API walkthrough

### Defining and running a Flex

**`dspy.Flex(signature, *, tools=None, interpreter_factory=PythonInterpreter, max_predictor_calls=100)`**
Parses the signature and binds the baseline source — a single `dspy.Predict` over the signature, or a `dspy.RLM` when `tools` are given. Marks the instance `_code_optimizable`, validates `interpreter_factory` (a zero-arg factory, defaulting to `dspy.PythonInterpreter`, exactly as `dspy.RLM`), and sets up the sandbox bridge. One instance carries one configuration.

**`__call__(**inputs)` / `forward(**inputs)`**
Runs the currently bound source inside the interpreter, bridging predictor calls back to the host. Returns a `dspy.Prediction` over the signature's output fields, and accepts keyword inputs only.

**`module_src`**
A read-only property holding the current implementation as source — one `dspy.Module` subclass. This is the value GEPA reads as the seed and overwrites with each accepted candidate. Print it to see what the optimizer discovered.

**`close()` / context-manager use**
Shuts down any interpreter sessions the `Flex` created. Safe to call repeatedly. Since `Flex` always runs sandboxed, use `with dspy.Flex(...) as f:` or call `close()` explicitly to release the Deno subprocess.

### Optimizing with GEPA

**`dspy.GEPA(metric=..., reflection_lm=..., ...).compile(flex_program, trainset=..., valset=...)`**
Compiling a program that contains one or more `Flex` submodules optimizes each `Flex`'s code and every non-flex predictor's instructions together, under one budget. Returns a new program whose `module_src` (per flex submodule) is the best code found. The auto-budget counts each flex submodule as one component alongside the instruction predictors.

**The metric's `feedback`**
As with any GEPA metric, the feedback string is the prompt handed to the proposer — here, the *code* proposer. Feedback that diagnoses *why* an output was wrong and hints at structure steers rewrites far better than a bare score.

**Trace-aware metrics (`program_trace`)**
Add `program_trace=None` as a sixth parameter to your metric and GEPA passes the execution trace at scoring time, so you can score against how the answer was produced — most commonly, `len(program_trace)` as an LM-call count to fold a small per-call penalty into the score. Metrics without the parameter are unaffected. GEPA still requires the five standard metric arguments `(gold, pred, trace, pred_name, pred_trace)`, so `program_trace` is a sixth, not a replacement — if you don't use `pred_name`/`pred_trace`, absorb them with `*args`: `def metric(gold, pred, trace=None, *args, program_trace=None)`.

### Tools and sandboxing

**`tools=[...]`**
Plain functions or `dspy.Tool` instances, referenced by name in the generated code, so each name must be a valid Python identifier. Providing tools makes the baseline a `dspy.RLM` and tells the code proposer the tools are in scope — to wire into `dspy.RLM`/`dspy.ReAct`, call directly, or supplement with its own inline helpers.

**`interpreter_factory=...`**
Defaults to `dspy.PythonInterpreter` (sandboxed, needs Deno), like `dspy.RLM`. Must be a zero-argument callable returning a fresh `CodeInterpreter`; each parallel evaluation gets its own session. As in `dspy.RLM`, a bare interpreter instance is not accepted — pass a factory.

**`max_predictor_calls`**
Caps bridged LM calls per `forward` as a runaway guard. `None` disables it.

### Saving and loading

**`save(path)` / `load(path)` / `dump_state()` / `load_state(state)`**
`module_src` travels in the serialized state, so loading restores the optimized code and rebuilds its predictors. The interpreter is not serialized; reconstructing with `dspy.Flex(signature)` restores the default sandbox, so re-supply it in the constructor before `load` only if you customized the `interpreter_factory`.

## Cross-links

- [`dspy.Flex` API reference](../api/modules/Flex.md) — constructor table, worked examples, and the full method list.
- [Built-in module variants](built-in-module-variants.md) — where `Flex` sits among the other non-`Predict` modules.
- [GEPA in depth](gepa-in-depth.md) — the reflective optimizer that drives `Flex`'s code search.
- [RLM: exploring large contexts with code](rlm.md) — the module `Flex` uses as its tool-enabled baseline, and a close cousin in how it runs generated code.
