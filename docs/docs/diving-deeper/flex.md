# Flex: Optimizable module code

## Intent

`dspy.Flex` is for tasks where you don't know the right *shape* of the solution up front. Every other module fixes its structure when you construct it — `Predict` is one call, `ChainOfThought` is one call with reasoning, `ReAct` is a tool loop — and optimization only tunes the prompt around that fixed structure. `Flex` moves the structure itself into the search space. You give it a signature; it starts as a `dspy.Predict` or `dspy.RLM` baseline; and `dspy.GEPA` rewrites its entire implementation — how many predictors, which primitives, what runs in Python instead of an LM — against your metric. Reach for it when the best decomposition is something you'd rather discover and optimize.

## Design decisions

### 1. A Flex is a module whose source code is the optimizable parameter

An ordinary module's tunable surface is its predictors' instructions. A `Flex`'s tunable surface is a whole `dspy.Module` subclass, held as a source string and exposed as `module_src`. That class has the usual two methods: `__init__` constructs whatever predictors it needs, and `forward` calls them and returns a `dspy.Prediction`. Optimizing the module means replacing that source with a better version — different predictors, different control flow, more or less Python. The prompt is no longer the unit of optimization; the program code is.

### 2. It drops into any signature and starts from a simple baseline

You construct `Flex` from any `dspy.Signature`, and it's immediately runnable. With no tools, its baseline source is a single `dspy.Predict` over the whole signature; with tools, a `dspy.RLM` so the baseline can call them. The baseline is the simplest thing that works as the starting point of a search.

### 3. GEPA discovers `Flex` by type and optimizes code instead of text

When GEPA compiles a program, it enumerates the `Flex` submodules and splits its work: each `Flex` becomes a **code component**, every other predictor stays an **instruction component**. Code components are seeded with their current `module_src` and evolved by a dedicated code proposer; instruction components are seeded with their current instructions and evolved by GEPA's usual instruction proposer. Each side has its own override: a custom `instruction_proposer` replaces the instruction proposer, a custom `code_proposer` replaces the code proposer, and whichever you leave unset keeps its default.

### 4. The code proposer reflects on whole-program behavior

Instruction optimization in GEPA is per-predictor: it looks at one predictor's inputs, outputs, and feedback. Code optimization can't work that way, because the predictors *are* part of what's being rewritten — they may not exist in the next candidate. So a code component reflects on the whole program's I/O instead: the module's inputs, its final prediction, and the metric's feedback per example. The proposer is given the signature, any available tools, a catalog of allowed primitives, the current source, and a batch of failing examples, and asked to return a full revised module class.

### 5. Predictors inside a Flex are owned by its code, not tuned in parallel

When you mix a `Flex` with ordinary modules in one program, GEPA optimizes the `Flex`'s code and the other predictors' instructions — but never the instructions of predictors that live *inside* the `Flex`. Those predictors are constructed by the current `module_src` and will be replaced wholesale by the next code candidate, so tuning their instructions would be optimizing something that may be overwritten.

`Flex` is a subclass of `Parameter` and `Module` (like `dspy.Predict`), so a parent program's `named_parameters()` yields it as one leaf and never recurses into it. A parent's `named_predictors()`, `predictors()`, `set_lm()`, and demo/instruction optimizers like `BootstrapFewShot` don't see the `Flex`'s internals. The `Flex` reports the same about itself: `flex.named_predictors()` is empty — its update unit is its code, not the predictors the code constructs. A `Flex` can carry its own LM (`flex.set_lm(...)`), which `forward` applies as the ambient LM for the whole call; otherwise the LM calls its code makes resolve the ambient LM at call time (`dspy.configure(lm=...)`, or a caller's `dspy.context(lm=...)`). `reset_copy()` resets a `Flex` the way it resets a `Predict`: the LM is cleared while `module_src` is kept, just as tuned instructions are.

### 6. A broken candidate scores as a failure, not a crash

The reflection model authors code, and code can be wrong. A candidate that doesn't parse raises when GEPA binds it to build the program for evaluation; `Flex` optimization catches that, logs it, and scores the whole batch at the failure score, letting the search continue and simply not select the broken candidate. Code that parses but breaks when it runs — an import that doesn't resolve, an edge case that throws on some inputs — fails per example at `forward` instead: each crashed example is scored at the failure score in its own slot, by example index, so the surviving scores stay aligned to the batch and GEPA's per-instance bookkeeping stays intact. Either way, the optimization is robust to the reflection model's mistakes by construction.

### 7. Generated code always runs through an interpreter

`Flex` takes its interpreter from `interpreter_factory`, else from `dspy.settings.interpreter_factory`, else from `dspy.PythonInterpreter` (Deno/Pyodide). Each of those must be a *zero-argument factory*: a bare instance or `None` is rejected. The generated code imports the real `dspy` in that interpreter. With the default, everything stays in the sandbox except provided-tool calls and LM calls, which go to the host. The host builds each request from its own LM, admits only generation options, counts the call against `max_predictor_calls`, and runs it, so the model, credentials, cache, and retries stay on the host. Each code-executing module that the generated code builds, like `dspy.RLM`, gets a fresh interpreter from the same factory, so its code shares nothing with the module that built it. Functions of that module it is given, like an `llm_query` or a helper, still run in the module's sandbox. Custom factories define their own trust boundary: `LocalInterpreter` separates process memory and stdout but retains the host user's filesystem, environment, credentials, network, and process authority. An in-process interpreter (one that runs code with `exec()` in the host, such as `dspy_interpreters.InProcessInterpreter`) works but isolates nothing: generated code uses the host's own `dspy` module, so `max_predictor_calls` and the LM-option allowlist bind only well-behaved code. Use one only for code you trust. Any other interpreter must let the generated code import the host's `dspy`. The factory creates a fresh interpreter for each forward, and one for each code-executing module the generated code builds, all shut down when the forward ends. Source portability between custom interpreters is not guaranteed.

### 8. The declared output types are enforced at the sandbox boundary

Everything the generated code returns crosses back as JSON, so a field declared `int`, `list[str]`, or a pydantic model would arrive as a bare string or dict. `Flex` parses each declared output field against its annotation on the way out, so a `Flex` returns the same types as a `dspy.Predict` over the same signature and stays substitutable for one. Two things follow. The custom types of the `Flex`'s signature have to exist in the sandbox too, so the generated code can name them (`text: str -> person: Person`, or `person: Person = dspy.OutputField()` in a class). dspy's own types, the standard library's and those of the packages dspy uses are imported there. Your own cross as their shape, read off the live class: a model's, dataclass's, `NamedTuple`'s or `TypedDict`'s fields, with their types, defaults, aliases and constraints, and an enum's members. The sandbox builds plain classes of the same names from that, so the generated code can read and build them. No code of yours crosses: methods and validators stay on the host, which validates with your real types what the sandbox sends back, and a default only your own factory function makes is made there too. An adapter class of your own runs on the host as well: predictors in the sandbox send it the signature, demos and inputs, and it formats, calls the metered LM and parses there, as without a sandbox. A type the generated code defines itself reaches it as text, which is parsed into that type in the sandbox: the host never builds a class from what the sandbox sends. Values cross as JSON, by field name, so `bytes` have to be UTF-8 text (send other bytes base64-encoded). A field excluded from serialization (`exclude=True`) doesn't cross, so a type that requires one reaches the generated code as a plain dict, as it can't be rebuilt from its own dump on the host either. The host parses the outputs back into your real types. And a candidate whose output doesn't match — wrong shape, or a declared field missing entirely — raises a `CodeInterpreterError` naming the field, which GEPA scores at the failure score for that example rather than handing the metric a `Prediction` that breaks somewhere else. An annotation that can't round-trip through a signature string at all (a `Callable`, say) is emitted untyped rather than rendered into source that wouldn't bind.

### 9. The code is state; the interpreter is a runtime dependency

A saved `Flex` is `{"module_src": ..., "lm": ...}` — the code, plus any LM set directly on the module. The internal predictors are not saved: they are derived from the code, and each `forward` reconstructs them from the bound source. The interpreter is a live runtime resource and is not serialized; `save(path, save_program=True)` cloudpickles the program with the bridge excluded, and loading rebuilds it. Reconstructing with `dspy.Flex(signature)` restores the configured sandbox, so you only re-supply the `interpreter_factory` before `load` if you optimized with one passed to the constructor.

### 10. Flex is experimental and the interface is in flux

The class carries the `@experimental` decorator. Treat the API and serialization format as subject to change between minor releases and pin a version if you depend on it.

## API walkthrough

### Defining and running a Flex

**`dspy.Flex(signature, *, tools=None, interpreter_factory=PythonInterpreter, max_predictor_calls=100)`**
Parses the signature and binds the baseline source — a single `dspy.Predict` over the signature, or a `dspy.RLM` when `tools` are given. Validates `interpreter_factory` as a zero-arg factory and sets up the sandbox bridge. `dspy.configure(interpreter_factory=...)` replaces the default. A factory passed to `Flex` wins, unless it is `PythonInterpreter`.

**`__call__(**inputs)` / `forward(**inputs)`**
Runs the currently bound source inside the interpreter, where every LM call goes to the host. Returns a `dspy.Prediction` over the signature's output fields, and accepts keyword inputs only.

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
Defaults to `dspy.PythonInterpreter` (sandboxed, needs Deno). `dspy.configure(interpreter_factory=...)` replaces that default on each interpreter session, including for a `Flex` built before the call. Must be a zero-argument callable returning a fresh `CodeInterpreter` for each interpreter session, so parallel evaluations receive isolated sessions. Any interpreter other than `dspy.PythonInterpreter` must let the generated code import the host's `dspy`. As in `dspy.RLM`, a bare interpreter instance is not accepted. This low-level hook does not guarantee source, standard-library portability, or a particular security boundary between different interpreters.

**`max_predictor_calls`**
The maximum number of LM calls the generated code can make in one `forward`, counted by the host as each call arrives. Every predictor call makes at least one. It guards against runaway loops. `None` removes the limit.

### What the generated code can use

The optimizer-authored code runs against the real `dspy` package, the same source the host runs. Only a few things cross to the host:

- LM calls. Every predictor uses the host's LM through a stand-in that sends each request to the host. Generation parameters (`temperature`, `max_tokens`, `n`, …) apply, while LM routing and credential options (`api_base`, `api_key`, headers, `model`) are rejected by the host. The host runs each call on its LM's sync path, an async one in the sandbox too, since the caller's loop waits while the sandbox runs. An LM that only implements async can't serve a sandbox.
- Provided tools. Each tool passed to `dspy.Flex(tools=...)` is in scope by name as a `dspy.Tool` carrying its host name, description, and arguments, so the code can call it directly or pass it to `dspy.ReAct` or `dspy.RLM`. It runs on the host. Functions the code defines run in the sandbox and can be passed to those modules as well.
- Custom types of the signature, as described in section 8.
- The trace. Predictor calls made in the sandbox are added to the host's `dspy.settings.trace`, which is how `program_trace` metrics see them.

The code should build its modules from the primitives the proposer is told about (`dspy.Predict`, `dspy.ChainOfThought`, `dspy.ReAct`, `dspy.RLM`, string or class-based `dspy.Signature`s, `dspy.Tool`, `dspy.Prediction`) and the Python standard library, imported *inside* `forward`. It should not construct LMs or change settings, since the host owns the LM. `dspy.PythonInterpreter` has no filesystem, network, or threads, so `dspy.Parallel` and batched calls run one after another there. Anything that fails when the candidate runs is scored at the failure score, so a mistake costs a search step rather than crashing the run. The proposer-facing version of this list is `dspy/predict/flex/primitives_doc.py`.

**The sandbox bundle**
The first time a machine runs a `Flex` forward (or a `dspy.RLM` call) on `dspy.PythonInterpreter`, the host reads dspy's dependencies from its installed metadata, takes Pyodide's builds of those Pyodide ships (from the Pyodide CDN, checked against Pyodide's lock file) and this environment's copies of the other pure-Python ones, and byte-compiles them together with dspy's own source into `$DSPY_CACHEDIR/sandbox` (`~/.dspy_cache/sandbox` by default). That takes a few seconds, once, and again whenever the dspy source changes. After that, a sandbox starts and imports dspy in well under a second.

### Saving and loading

**`save(path)` / `load(path)` / `dump_state()` / `load_state(state)`**
`module_src` travels in the serialized state (along with any LM set on the module), so loading restores the optimized code; the predictors are rebuilt from it on each `forward`. The interpreter is not serialized; reconstructing with `dspy.Flex(signature)` restores the configured sandbox, so re-supply it in the constructor before `load` only if you passed a custom `interpreter_factory` there.

## Cross-links

- [`dspy.Flex` API reference](../api/modules/Flex.md) — constructor table, examples, and the full method list.
- [Built-in module variants](built-in-module-variants.md) — where `Flex` sits among the other non-`Predict` modules.
- [GEPA in depth](gepa-in-depth.md) — the reflective optimizer that drives `Flex`'s code search.
- [RLM: exploring large contexts with code](rlm.md) — the module `Flex` uses as its tool-enabled baseline.
