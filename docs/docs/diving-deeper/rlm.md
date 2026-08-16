# RLM: exploring large contexts with code

## Intent

`RLM`, or “Recursive Language Model”, is for tasks whose context is too large, too messy, or too unevenly relevant to feed to a model directly. Instead of putting the whole context in the prompt, `dspy.RLM` hands the model a Python REPL where the context lives as variables, then lets the model write code and call sub-LLMs to explore and manipulate the inputs. Try `dspy.RLM` when your program is suffering from context rot and/or you want the model to decide how to decompose the problem.

## Design decisions

### 1. RLM splits context into a variable space and a token space

A normal `Predict` call puts the whole context into the prompt, so every token competes for the model’s attention and accuracy falls as the context grows. `dspy.RLM` keeps the context in the REPL as variables and shows the model only metadata: each variable’s name, type, length, and a short preview. The model pulls pieces into the prompt on demand by writing `print()` and code. The model reads what it needs, when it needs it.

### 2. RLM drops into any signature as an inference-time strategy

`dspy.RLM` takes the same signature you’d hand `Predict` or `ChainOfThought`. The input fields become REPL variables and the output fields define what the model must submit and how each value is typed. Nothing about your task definition changes when you swap the module. You can try `dspy.RLM` on an existing program by editing one line and revert just as cheaply, which is what makes it an inference-time strategy rather than a rewrite.

### 3. The LLM drives an iterative REPL loop

`dspy.RLM` is not a single call. `forward()` runs up to `max_iters` turns. On each turn the internal `generate_action` predictor sees the variable metadata and the prior REPL history, then emits reasoning plus one Python code block. The code runs, its output joins the `REPLHistory`, and the next turn begins. State persists across turns because the whole run shares one interpreter session. This is why the instructions push the model to explore first and take small steps: each turn is meant to build on what the last one printed.

### 4. Built-in `llm_query` tools give the loop its recursion

The recursion in an RLM is the model’s ability to call a model from inside its own code. `dspy.RLM` injects two functions into the sandbox, `llm_query(prompt)` for one call and `llm_query_batched(prompts)` for concurrent calls. For example, the outer model might use code to locate and slice relevant text, then hands a focused snippet to a sub-LLM for the semantic read. Results come back as Python values it can store and combine rather than as a text blob forced into the context window. One long-context question becomes many short-context ones.

### 5. A shared counter caps sub-LLM calls per run

Recursion is the expensive part, so `dspy.RLM` bounds it. `max_llm_calls` caps how many sub-LLM calls the model can make over a whole run; each prompt in a `llm_query_batched` list is counted individually, so `llm_query_batched([p1, p2, p3])` costs three budget units. Exceeding the cap raises an error that is fed back to the model, nudging it to aggregate in plain Python instead of spending more calls. The budget resets on each `forward()`, so a runaway loop cannot quietly fan out into hundreds of paid calls.

### 6. A separate `sub_lm` can handle the recursive calls

The model steering the loop and the model answering snippets need not be the same. `sub_lm` sets the model for `llm_query`, falling back to `dspy.settings.lm` when unset. A common split pairs a strong model for planning with a cheap model for extraction, since reading one snippet is simpler than orchestrating the whole search. You pay for the strong model only on the calls that need judgment.

### 7. Generated code runs in a pluggable sandboxed interpreter

The model’s code is untrusted, so it runs in a sandbox. The default `PythonInterpreter` executes Python in a Deno and Pyodide WASM runtime with no filesystem or network access. Each `forward()` creates one interpreter, uses it for the whole REPL loop, and shuts it down afterward. Supply `interpreter_factory=` to configure another `CodeInterpreter`, such as an adapter for a remote sandbox.

### 8. `SandboxSerializable` loads large inputs into the sandbox once

Some inputs are not plain strings. A DataFrame, a parsed corpus, or a binary blob would be wasteful and lossy to re-marshal into code on every turn. `SandboxSerializable` lets an input define how it crosses into the sandbox through four hooks: `sandbox_setup()` for imports, `to_sandbox()` for the payload, `sandbox_assignment()` for the reconstruction, and `rlm_preview()` for the metadata the model sees. `dspy.RLM` runs this once at the start of `forward()`, so the value is rebuilt under its original name and stays live across turns. The model works with a real pandas DataFrame rather than a stringified copy.

### 9. An extract pass salvages outputs when the loop ends early

A model can exhaust `max_iters` without ever submitting final outputs. Rather than return nothing, `dspy.RLM` falls back to a second predictor, the `extract` step, which reads the variable metadata and the full REPL history and produces the signature’s output fields directly. You get the best answer the trajectory supports even when the model never declared itself done.

### 10. RLM optimizes like any other module

The action and extract steps are ordinary `dspy.Predict` instances; `dspy.RLM` exposes them through `named_predictors` and compiles like any module. The task instructions, output-field descriptions, and tool docs are formatted into the action prompt when you construct the module, which is the surface an optimizer tunes. Running GEPA or MIPROv2 against a metric improves the loop’s behavior, not just the task instructions.

### 11. RLM is marked experimental and the interface is in flux

The class carries the `@experimental` decorator. The hard parts are still settling: call budgeting, sandbox lifetime, error recovery, and the action prompt itself. Treat the API as subject to change between releases and pin a version if you depend on it.

## API walkthrough

### Defining and running an RLM

**`dspy.RLM(signature, max_iters=20, max_llm_calls=50, max_output_chars=10_000, verbose=False, tools=None, sub_lm=None, interpreter_factory=PythonInterpreter)`**
The constructor parses the signature and builds the two internal predictors, formatting your task instructions, input names, and output-field types into the action prompt and creating a separate extract signature for the fallback. The budgets and the sandbox configuration are all fixed here, so one instance carries one configuration.

**`__call__([interpreter], **inputs)` / `acall([interpreter], **inputs)`**
The public call validates the inputs against the signature, builds the variable list, opens the interpreter, and runs the loop. Each turn asks the action predictor for code, runs it, and appends the result to history until the model submits or the loop reaches `max_iters`. `acall()` is the async twin and uses `acall` on the predictors. Both return a `Prediction`. An interpreter passed as the first positional argument is caller-owned: RLM updates its execution context but does not shut it down.

### Programming the loop with built-in tools

The model writes these into its code blocks. You don’t call them, but knowing them tells you what the model can do.

**`llm_query(prompt)`**
One sub-LLM call. It sends the prompt to `sub_lm` or the configured LM, increments the counter, and returns the response text. It raises if the prompt is empty or the budget is spent.

**`llm_query_batched(prompts)`**
Concurrent sub-LLM calls over a list, run on an eight-worker thread pool and returned in input order. A failed call comes back as an `[ERROR] ...` string in its slot rather than aborting the batch. This beats a Python loop of `llm_query` when the model has many independent snippets to read.

**`SUBMIT(...)`**
Ends the run and returns the final outputs. RLM validates the submitted dict against the signature’s output fields and parses each value to its declared type. On a type error or a missing field it feeds the message back to the model for another attempt instead of failing the call.

**`print(...)`**
The only way the model sees a result. REPL stdout is captured, truncated to `max_output_chars`, and shown in the next turn’s history. A code block that computes without printing returns `(no output - did you forget to print?)`, which is why the instructions stress printing.

### Budgeting iterations and sub-LLM calls

**`max_iters`, `max_llm_calls`, `max_output_chars`**
Three independent limits. `max_iters` bounds REPL turns before the extract fallback fires. `max_llm_calls` bounds recursive sub-LLM calls across the whole run. `max_output_chars` bounds how much of each REPL output reaches the model, keeping a noisy print from flooding the prompt.

**`sub_lm`**
The model for `llm_query` and `llm_query_batched`. Left unset it falls back to `dspy.settings.lm`, so by default the loop and its sub-calls share one model.

### Extending the sandbox with tools and inputs

**`tools=[...]`**
A list of plain functions or `dspy.Tool` objects. RLM normalizes each to a `Tool`, rejects names that aren’t valid identifiers or that collide with the built-ins, and documents their signatures in the action prompt. The model calls them as ordinary Python inside its code.

**`interpreter_factory=...`**
A zero-argument callable that returns a fresh `CodeInterpreter` for one invocation. RLM may call the factory concurrently, and it always shuts down the returned interpreter. A class such as `PythonInterpreter` is already a factory; use `functools.partial` or a callable provider object when construction needs configuration. RLM adds invocation-scoped tools to the returned interpreter's mutable `tools` dictionary, so remote sandboxes need a `CodeInterpreter` adapter that supports that protocol.

**`__call__(interpreter, **inputs)` / `acall(interpreter, **inputs)`**
An escape hatch for a caller-owned interpreter, supplied as the first positional argument. RLM mutates its `tools` dictionary and, when supported, its output-field metadata, but does not shut down or restore the instance. Reuse is supported only for sequential calls to the same RLM instance, so retained variables and tool registrations stay within one program and trust boundary. Use `interpreter_factory` for concurrent invocations. A `PythonInterpreter` override must also stay on the thread where it was first used.

**`dspy.SandboxSerializable`**
The base class for inputs that need custom loading. Implement `sandbox_setup`, `to_sandbox`, `sandbox_assignment`, and `rlm_preview`. It also defines a Pydantic schema hook, so a subclass can be a typed field in a signature, as in `data: DataFrame = dspy.InputField()`.

### Inspecting the trajectory

**`Prediction` fields: your output fields, `trajectory`, `final_reasoning`**
`forward()` returns a `Prediction` carrying the signature’s output fields plus two debugging fields. `trajectory` is a list of `{reasoning, code, output}` dicts, one per turn. `final_reasoning` is the model’s reasoning on the closing step. Read `trajectory` to see exactly what code ran and why.

**`RLM.tools`**
A property returning the user-provided tools as a name-to-`Tool` dict, excluding the built-in `llm_query` and `llm_query_batched`. Use it to confirm what the model can call.

## Cross-links

- [Built-in module variants](built-in-module-variants.md) — where RLM sits among the other non-`Predict` modules.
- [Tools, ReAct, and MCP](tools-react-and-mcp.md) — the tool-wrapping machinery RLM reuses.
- [`dspy.RLM` API reference](../api/modules/RLM.md) — full parameter table, built-in tool list, and worked examples.
