# dspy.Flex

`Flex` is a DSPy module whose implementation is *optimizable code* rather than a fixed prompt. You construct it from a signature, and it defaults to a thin baseline over that signature. What makes it different is what an optimizer is allowed to do with it: instead of only rewriting instructions, `dspy.GEPA` can rewrite the module's entire source — splitting the task into multiple predictors, folding deterministic steps into plain Python, and authoring its own helper tools. `Flex` acts as a marker to indicate that code is an optimizable parameter.

Reach for `Flex` when you don't yet know the right *shape* of a solution — how many LM calls it needs, where code should replace a call, how the work should decompose — and you'd rather have the optimizer discover that structure than hand-write it.

## When to Use Flex

- The best decomposition is **unknown or worth searching** — you have a metric and examples, and you'd rather optimize the program's structure than hand-write it.
- Parts of the task are **deterministic** and shouldn't cost an LM call — arithmetic, parsing, lookups, normalization.
- You want the optimizer to **trade accuracy against cost** — e.g. rewarding programs that answer clear cases in code and reserve the LM for genuinely hard ones.
- You generally don't care much about the concrete implementation of the module.

If the structure is already clear, a hand-written `dspy.Module` (optionally optimized with GEPA or MIPROv2 on its instructions) is simpler. `Flex` earns its keep when the module's forward method itself is the thing you want to learn.

## Basic Usage

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-5"))

# Construct it from a signature, like any module.
solve = dspy.Flex("invoice: str -> total_cents: int")

# Runs using baseline (a single dspy.Predict).
result = solve(invoice="2 widgets @ $3.50, shipping $1.00")
print(result.total_cents)
```

Out of the box, `solve` is just a `dspy.Predict` over the signature, wrapped in a module. The point of `Flex` is what happens when you optimize it (see [Optimizing with GEPA](#optimizing-with-gepa)): GEPA can replace that baseline with, say, a predictor that only extracts quantities and unit prices, and a line of Python that multiplies and sums them.

The generated code always runs in a sandbox (`interpreter_factory` defaults to `dspy.PythonInterpreter`), so the example above needs [Deno](https://deno.land/) installed — see [Sandboxed Execution](#sandboxed-execution).

## How Optimization Works

`Flex` modules are marked `_code_optimizable`, a flag `dspy.GEPA` looks for. When GEPA compiles a program containing one or more `Flex` submodules, it treats each one as a **code component**: rather than proposing a new instruction string, its reflection model proposes a new *whole module source*, guided by the signature, any available tools, and your evals. GEPA binds the candidate source, evaluates it, and keeps it if it scores better — the same Pareto-based search GEPA runs for prompts, applied to code.

One thing to note is that a broken GEPA candidate can't crash the run. If the reflection model emits source that fails to import or bind, GEPA scores that candidate as a failure and moves on, rather than aborting the optimization.

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

The `program_trace` parameter is opt-in *by declaration*: only metrics that name it receive the trace, so existing GEPA metrics keep their usual scoring semantics. Keep the penalty small relative to correctness, so a decomposition has to *hold* accuracy to win — it can never beat a strictly-more-accurate program.

!!! note "Metric signature"
    `program_trace` is the optional **sixth** parameter of the GEPA metric contract: `(gold, pred, trace, pred_name, pred_trace, program_trace=None)`. It defaults to `None`, so ordinary five-argument GEPA metrics remain valid and unchanged — you only add it when you want the execution trace at scoring time. GEPA still requires the five standard arguments (it passes all five at feedback time), so `program_trace` is an addition, not a replacement for them.

## Sandboxed Execution

`Flex` always runs its generated code in a sandbox — like `dspy.RLM`, it never runs it in the host Python process. `interpreter_factory` defaults to `dspy.PythonInterpreter` (Deno/Pyodide) and must be a **zero-argument factory** returning a fresh `CodeInterpreter`; a bare instance is not accepted, so each parallel evaluation during optimization gets its own session. The code is authored by the reflection model, so isolating it keeps it from running with your host's full permissions. The optimizer-authored glue — control flow, string work, arithmetic, imports — runs inside the sandbox, and only predictor construction and predictor calls bridge back to the host, which makes the real LM calls.

Because the default builds a `PythonInterpreter`, constructing a `Flex` needs [Deno](https://deno.land/) installed and raises with install instructions otherwise. To customize the sandbox — grant filesystem or network access, or use another `CodeInterpreter` backend — pass your own factory:

```python
solve = dspy.Flex(
    "invoice: str -> total_cents: int",
    interpreter_factory=lambda: dspy.PythonInterpreter(),  # the explicit form of the default; configure it here
)
```

Because the interpreter holds live sessions, use the `Flex` as a context manager or call `close()` when you're done:

```python
with dspy.Flex("invoice: str -> total_cents: int") as solve:
    result = solve(invoice=text)
```

## Tools

Pass `tools` and the baseline starts as a `dspy.RLM` (so it can call them) instead of a `dspy.Predict`, and the code proposer is told the tools are in scope by name:

```python
def lookup_sku(code: str) -> dict:
    """Look up a product by SKU."""
    return catalog[code]

solve = dspy.Flex("order: str -> total_cents: int", tools=[lookup_sku])
```

The optimizer can then wire your tools into `dspy.RLM(..., tools=[...])` / `dspy.ReAct(..., tools=[...])`, call them directly from `forward`, or author its own helper functions inline. Tool names must be valid Python identifiers, since the generated code references them by name.

## Saving and Loading

A `Flex` serializes its `module_src`, so saving and loading a program restores the optimized code:

```python
optimized.save("solver.json")

restored = dspy.Flex("invoice: str -> total_cents: int")
restored.load("solver.json")  # rebinds the saved module_src
```

The interpreter, like the LM, is a **runtime dependency and is not serialized**. Reconstructing with `dspy.Flex(signature)` restores the default sandbox automatically; if you optimized with a customized `interpreter_factory`, pass the same one when you reconstruct the module before calling `load`.

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| Signature` | required | Declares the module's inputs and outputs (e.g. `"invoice -> total_cents: int"`). |
| `tools` | `list[Callable \| dspy.Tool]` | `None` | Tools the generated code may call. With tools, the baseline is a `dspy.RLM`; without, a `dspy.Predict`. |
| `interpreter_factory` | `Callable[[], CodeInterpreter]` | `PythonInterpreter` | Zero-arg factory returning the sandbox that runs the generated code; defaults to `dspy.PythonInterpreter` (needs Deno), like `dspy.RLM`. A bare interpreter instance is not accepted. |
| `max_predictor_calls` | `int` | `100` | Cap on bridged LM calls per `forward` (a runaway guard). `None` disables it. |

## Notes

!!! warning "Experimental"
    `Flex` is marked experimental. The API and the optimization behavior may change between releases; pin a version if you depend on it.

!!! note "Interpreter Requirements"
    `Flex` always runs generated code in a sandbox (`interpreter_factory` defaults to `dspy.PythonInterpreter`), which requires [Deno](https://deno.land/) for its Pyodide WASM sandbox — see the [RLM page](RLM.md#deno-installation) for installation notes.

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
            - close
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
