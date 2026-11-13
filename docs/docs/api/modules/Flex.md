# dspy.Flex

`Flex` is a DSPy module whose implementation is *optimizable code* rather than a fixed prompt. You construct it from a signature, and it defaults to a thin baseline over that signature. What makes it different is what an optimizer is allowed to do with it: instead of only rewriting instructions, `dspy.GEPA` can rewrite the module's entire source — splitting the task into multiple predictors, folding deterministic steps into plain Python, and authoring its own helper functions. Being a `Flex` is what tells GEPA that the module's code is an optimizable parameter.

## When to Use Flex

Reach for `Flex` when you'd rather have the optimizer discover the program's structure than hand-write it. That's the case when:

- The best decomposition is **unknown or worth searching** — you have a metric and a dataset to judge candidate structures against.
- Parts of the task are **deterministic** and shouldn't cost an LM call — arithmetic, parsing, lookups, normalization.
- You want the optimizer to **trade accuracy against cost** — e.g. rewarding programs that answer clear cases in code and reserve the LM for genuinely hard ones.

## Basic Usage

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-5"))

# Construct Flex from a signature, like any module.
solve = dspy.Flex("invoice: str -> total_cents: int")

# Runs the baseline (a single dspy.Predict).
result = solve(invoice="2 widgets @ $3.50, shipping $1.00")
print(result.total_cents)
```

Out of the box, `solve` is just a `dspy.Predict` over the signature, wrapped in a module (with `tools`, it starts as a `dspy.RLM` instead — see [Tools](#tools)). The point of `Flex` is what happens when you optimize it (see [Optimizing with GEPA](#optimizing-with-gepa)): GEPA can replace that baseline with, say, a predictor that only extracts quantities and unit prices, and a line of Python that multiplies and sums them.

The generated code always runs through a `CodeInterpreter` (`interpreter_factory` defaults to the sandboxed `dspy.PythonInterpreter`), so the example above needs [Deno](https://deno.land/) installed — see [Interpreter Execution](#interpreter-execution).

## How Optimization Works

`dspy.GEPA` discovers `Flex` submodules by type. When GEPA compiles a program containing one or more `Flex` submodules, it treats each one as a **code component**: rather than proposing a new instruction string, its reflection model proposes a new *whole module source*, guided by the signature, any available tools, and your metric's feedback on failing examples. GEPA binds the candidate source, evaluates it, and keeps it if it advances the Pareto frontier — the same search GEPA runs for prompts, applied to code.

A broken candidate can't crash the optimization run. If the reflection model emits source that fails to bind, GEPA scores that candidate as a failure and moves on, rather than aborting the optimization.

## Optimizing with GEPA

You optimize a `Flex` the same way you optimize any DSPy program — hand it to `dspy.GEPA` with a metric and a trainset:

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-5-mini"))  # runs the program

def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    correct = getattr(pred, "total_cents", None) == gold.total_cents
    fb = "Correct." if correct else (
        f"Wrong total: got {getattr(pred, 'total_cents', None)}, expected {gold.total_cents}. "
        "Have the LM extract line items, then sum them in Python."
    )
    return dspy.Prediction(score=1.0 if correct else 0.0, feedback=fb)

solve = dspy.Flex("invoice: str -> total_cents: int")

optimized = dspy.GEPA(
    metric=metric,
    reflection_lm=dspy.LM("openai/gpt-5", temperature=1.0, max_tokens=8000),
    max_metric_calls=60,
).compile(solve, trainset=trainset, valset=valset)

print(optimized.module_src)  # the discovered program
```

The `metric` returns a `dspy.Prediction(score=..., feedback=...)` — a scalar plus natural-language feedback that GEPA reflects on to revise the module. For how to write an effective feedback metric, see [Implementing Feedback Metrics](../optimizers/GEPA/overview.md#implementing-feedback-metrics) in the GEPA guide and the [dspy.GEPA tutorials](../../tutorials/gepa_ai_program/index.md).

### Rewarding leaner programs with a trace-aware metric

A common goal with `Flex` is to push work out of the LM and into deterministic code. To optimize for that, your metric needs to see *how* an answer was produced, not just whether it was right. Declare a `program_trace` parameter and GEPA will pass the execution trace to the metric at scoring time, letting you penalize LM calls:

```python
LLM_CALL_PENALTY = 0.15

def metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None):
    correct = getattr(pred, "total_cents", None) == gold.total_cents
    n_calls = len(program_trace) if program_trace else 0
    score = max(0.0, (1.0 if correct else 0.0) - LLM_CALL_PENALTY * n_calls)
    fb = f"{'Correct' if correct else 'Wrong'} — used {n_calls} LM call(s). Settle clear cases in Python."
    return dspy.Prediction(score=score, feedback=fb)
```

The `program_trace` parameter is opt-in *by declaration*: only metrics that name it receive the trace. Keep the penalty small relative to correctness, so a decomposition has to *hold* accuracy to win.

## Interpreter Execution

`Flex` always runs its generated code through a `CodeInterpreter`. `interpreter_factory` defaults to `dspy.PythonInterpreter` (Deno/Pyodide) and must be a **zero-argument factory** returning a fresh interpreter; a bare instance is not accepted, so parallel evaluations receive isolated sessions. `dspy.configure(interpreter_factory=...)` replaces that default. A factory passed to `Flex` wins, unless it is `PythonInterpreter`. The factory is called once per forward, and once more for each code-executing module the generated code builds, which gets a fresh interpreter of its own. With the default interpreter, the optimizer-authored code runs inside the sandbox against the real `dspy`, and only provided-tool calls and LM calls go back to the host, which makes the real LM calls. Custom factories define their own trust boundary: `dspy.LocalInterpreter`, for example, separates process memory and stdout but retains the host user's filesystem, environment, credentials, network, and process authority. An in-process interpreter that runs code with `exec()` in the host works, but isolates nothing; use one only for trusted code.

Because the default builds a `PythonInterpreter`, *running* a `Flex` needs [Deno](https://deno.land/) installed; without it, the call raises.

```python
solve = dspy.Flex(
    "invoice: str -> total_cents: int",
    interpreter_factory=lambda: dspy.PythonInterpreter(),  # the default; swap in your own CodeInterpreter factory here
)
```

One call puts the same interpreter behind every code-executing module in the program — a
remote sandbox, say, or one for a host that cannot run Deno:

```python
dspy.configure(interpreter_factory=MyInterpreter)  # every module with no factory of its own
```

Each call owns and shuts down every interpreter session it creates, so a `Flex` holds no live sessions between calls.

## Tools

Pass `tools` and the baseline starts as a `dspy.RLM` instead of a `dspy.Predict`.

```python
def lookup_sku(code: str) -> dict:
    """Look up a product by SKU."""
    return catalog[code]

solve = dspy.Flex("order: str -> total_cents: int", tools=[lookup_sku])
```

The optimizer can then wire your tools into `dspy.RLM(..., tools=[...])` / `dspy.ReAct(..., tools=[...])`, or call them directly from `forward`.

## Your Own Libraries

The code can also import your own code, such as your project's abstractions or your company's SDK. Pass it as `libraries`:

```python
import shopkit                      # a package of your project
from shopkit.cart import Cart

class Checkout(dspy.Signature):
    """Total an order."""
    order: str = dspy.InputField()
    cart: Cart = dspy.InputField()
    total_cents: int = dspy.OutputField()

checkout = dspy.Flex(Checkout, libraries=[shopkit])
```

A library is a module, or a single class or function, to point the optimizer at one part of a large package (`libraries=[Cart, shopkit.vat_rate]`). A name works too, for a library that only the interpreter has: `"shopkit.pricing"`, or `"shopkit.cart:Cart"` for a class.

The interpreter makes the libraries importable, and everything else follows from what it has:

- **The optimizer** gets an API reference of each library, read in the interpreter: how to import it, its docstring, and its public functions and classes with their signatures and docstrings. Notes on how a library is meant to be used belong in its module docstring, where they stay with the code. The optimizer writes the imports at the top of the module, and uses your library for what it does and a DSPy module where a language model is needed. To start from code of your own, such as a module that only uses your libraries, pass it as `module_src`.
- **Values** of a library's classes are objects of those classes in the interpreter, so the code can call `cart.add(...)` on a `Cart` you pass in. A value the code returns comes back as plain data, which `Flex` reads as the type your signature declares.
- **The signature stays the contract**: the outputs are parsed and checked against their declared types, and your metric scores them as usual.

Where the code runs decides what a library can do there:

- `dspy.PythonInterpreter` (the default) copies what the libraries reach into its sandbox, and no more: the Python source files of your own code that importing them leads to, and the installed packages those import, as Pyodide builds them or from your environment if they are pure Python. Imports are read from the source, so one made by a computed name is not seen: name such a module among the libraries. Of your own code, only code is copied. A data file next to it may hold anything, so it stays on the host: a library that reads data files of its own gets them as an input or through a tool, or runs in another interpreter. A library with compiled code that Pyodide doesn't ship is refused with an error. The sandbox runs Pyodide's Python and Pyodide's versions of the packages it ships, which can differ from your environment's, and it has no network or host files: anything that needs them goes through your [tools](#tools).
- `dspy.LocalInterpreter` imports the libraries from your environment, so any library works, with your credentials, network and files. It isolates nothing.
- An interpreter of your own is taken to have the libraries in its environment. If it has to do something to provide them, it defines `provide_libraries(modules)`.

A `dspy.InterpreterPool` needs no setup for libraries: a `Flex` tells the pool it uses, which starts its interpreters with them from then on.

### What the code was validated against

A library changes, and code written against it may stop working. `Flex` records what its code was validated against, read in the interpreter: its Python version, each package the libraries reach with its version there, and a hash of each of your own source files they reach.

```python
optimized = dspy.GEPA(...).compile(checkout, trainset=trainset, valset=valset)
optimized.environment
# {"python": "3.13",
#  "requires": ["regex==2024.11.6"],
#  "sources": {"shopkit": {"shopkit/cart.py": "ece11f4c56706929", "shopkit/vat.py": "cd0a1b3c5df5ba96", ...}}}
```

`dspy.GEPA` records it for the code it returns. It is saved with the program, and forgotten when the code changes. It covers what the libraries you declared reach, so declare everything the code imports. The first call of a program whose recorded environment differs from the one it now runs in warns, once, and says what differs:

```text
dspy.Flex Checkout: its code was validated against another environment (shopkit/vat.py changed). Run your
evaluation again, then call pin(), or optimize it again.
```

`check_environment()` returns the same as a list, which a CI job can assert is empty. When your evaluation still passes, `pin()` records the environment as it is now. When it doesn't, optimizing again starts from the current code, so the optimizer only has to repair it. Call `pin()` as well after writing or editing `module_src` yourself.

### Tools for other SDKs

Code that hands your tools to another SDK, like an agent framework, can pass `tool.as_function()`: a plain function with the tool's name, description and typed arguments, which SDKs that build tools from a function's signature read like any other. Calling it calls the tool.

### Side effects

An SDK can do much more than a tool, and an optimizer runs your program many times. During optimization, give the program a test double, a sandbox account or test credentials for anything that changes the world. Inside the sandbox, a library can't reach the network, so side effects can only happen through your tools.

Each prediction lists the calls its forward made to the provided tools, in `prediction.tool_calls`. A metric can score them and say why in its feedback, which the optimizer reads:

```python
def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    refunds = [call for call in prediction.tool_calls if call["name"] == "refund"]
    if refunds and not example.should_refund:
        return dspy.Prediction(score=0.0, feedback=f"It refunded {refunds[0]['args']}, but the customer only asked a question.")
    return dspy.Prediction(score=float(prediction.reply == example.reply), feedback="")
```

The metric runs after the forward, so it can't stop a side effect. To stop one, check in the tool itself, or swap in a test double while optimizing.

## Saving and Loading

A `Flex` serializes its `module_src`, so saving and loading a program restores the optimized code:

```python
optimized.save("solver.json")

restored = dspy.Flex("invoice: str -> total_cents: int")
restored.load("solver.json")  # rebinds the saved module_src
```

The interpreter is a **runtime dependency and is not serialized**. Reconstructing with `dspy.Flex(signature)` restores the configured sandbox automatically; if you optimized with an `interpreter_factory` passed to the constructor, pass the same one when you reconstruct the module before calling `load`.

The saved state also holds the [environment the code was validated against](#what-the-code-was-validated-against), if one is recorded: what the libraries resolved to in the interpreter, and a fingerprint of each tool. A program loaded from JSON or from a pickle warns on its first call when that differs from the environment it now runs in.

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| Signature` | required | Declares the module's inputs and outputs (e.g. `"invoice -> total_cents: int"`). |
| `tools` | `list[Callable \| dspy.Tool]` | `None` | Tools the generated code may call. With tools, the baseline is a `dspy.RLM`; without, a `dspy.Predict`. |
| `libraries` | `list[module \| class \| function \| str]` | `None` | What the generated code may import besides dspy and the standard library. See [Your Own Libraries](#your-own-libraries). |
| `module_src` | `str` | `None` | Source to start from instead of the baseline. |
| `interpreter_factory` | `Callable[[], CodeInterpreter]` | `PythonInterpreter` | Zero-arg factory returning a fresh `CodeInterpreter` for each interpreter session; defaults to `dspy.PythonInterpreter` (needs Deno), and `dspy.configure(interpreter_factory=...)` replaces that default. A bare interpreter instance is not accepted. Supported Python, libraries, and security boundaries are interpreter-dependent. |
| `max_predictor_calls` | `int \| None` | `100` | Maximum number of LM calls the generated code can make in one `forward`, a guard against runaway loops. `None` removes the limit. |

## Notes

!!! note "Interpreter Requirements"
    `Flex` defaults to `dspy.PythonInterpreter`, which requires [Deno](https://deno.land/) for its Pyodide WASM sandbox — see the [RLM page](RLM.md#deno-installation) for installation notes. `dspy.LocalInterpreter` requires no additional runtime but is not a security sandbox.

## API Reference

<!-- START_API_REF -->
::: dspy.Flex
    handler: python
    options:
        members:
            - __init__
            - __call__
            - forward
            - module_src
            - signature
            - environment
            - pin
            - check_environment
            - deepcopy
            - dump_state
            - get_lm
            - inspect_history
            - load
            - load_state
            - named_parameters
            - named_predictors
            - named_sub_modules
            - parameters
            - predictors
            - reset
            - reset_copy
            - save
            - set_lm
        show_source: true
        show_root_heading: true
        heading_level: 3
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
<!-- END_API_REF -->
