# dspy.Flex

`Flex` is a DSPy module whose implementation is *optimizable code* rather than a fixed prompt. You construct it from a signature, and it starts life as a thin baseline over that signature. What makes it different is what an optimizer is allowed to do with it: instead of only rewriting instructions, `dspy.GEPA` can rewrite the module's entire source — splitting the task into focused predictors, folding deterministic steps into plain Python, and authoring its own helper tools.

Reach for `Flex` when you don't yet know the right *shape* of a solution — how many LM calls it needs, where code should replace a call, how the work should decompose — and you'd rather have the optimizer discover that structure than hand-write it.

## When to Use Flex

A normal module fixes its structure at construction time: `dspy.Predict("invoice -> total")` will always be one LM call, and optimization can only improve the prompt around that call. But many tasks are better solved by a *program* than a single call — extract the line items with an LM, then sum them in Python; settle the clear cases with a rule and only escalate the ambiguous ones to a model.

Use `Flex` when:

- The best decomposition is **unknown or worth searching** — you have a metric and examples, and you'd rather optimize the program's structure than guess it.
- Parts of the task are **deterministic** and shouldn't cost an LM call — arithmetic, parsing, lookups, normalization.
- You want the optimizer to **trade accuracy against cost** — e.g. rewarding programs that answer clear cases in code and reserve the LM for genuinely hard ones.
- You generally don't care about the concrete implementation of the module.

If the structure is already clear, a hand-written `dspy.Module` (optionally optimized with GEPA or MIPROv2 on its instructions) is simpler. `Flex` earns its keep when the structure itself is the thing you want to learn.

## Basic Usage

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-5"))

# Construct it from a signature, like any module.
solve = dspy.Flex("invoice: str -> total_cents: int")

# It runs immediately, using its baseline (a single dspy.Predict).
result = solve(invoice="2 widgets @ $3.50, shipping $1.00")
print(result.total_cents)

# Inspect the current implementation — a full dspy.Module subclass as source.
print(solve.module_src)
```

Out of the box, `solve` is just a `dspy.Predict` over the signature, wrapped in a module. The point of `Flex` is what happens when you optimize it (see [Optimizing with GEPA](#optimizing-with-gepa)): GEPA can replace that baseline with, say, a predictor that only extracts quantities and unit prices, and a line of Python that multiplies and sums them — no arithmetic left to the LM.

## How Optimization Works

Under the hood, a `Flex` holds its implementation as a string of Python source — one `dspy.Module` subclass with an `__init__` (which constructs the predictors it needs) and a `forward` (which calls them and returns a `dspy.Prediction`). This is exposed as the read-only `module_src` property.

The module is marked `_code_optimizable`, a flag `dspy.GEPA` looks for. When GEPA compiles a program containing one or more `Flex` submodules, it treats each one as a **code component**: rather than proposing a new instruction string, its reflection model proposes a new *whole module source*, guided by the signature, any available tools, and a batch of failing examples with feedback. GEPA binds the candidate source, evaluates it, and keeps it if it scores better — the same Pareto-based search GEPA runs for prompts, applied to code.

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

As with any GEPA metric, the `feedback` string matters as much as the score: it's the text GEPA hands its code proposer, so feedback that *diagnoses* the failure ("the LM did the arithmetic in its head — extract and sum in Python instead") steers the rewrite far better than a bare score.

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

By default, a `Flex` binds and runs its generated code in-process with `exec`. During optimization the code is authored by the reflection model, so you may prefer to run it in a sandbox. Pass an `interpreter`:

```python
solve = dspy.Flex(
    "invoice: str -> total_cents: int",
    interpreter=lambda: dspy.PythonInterpreter(),  # zero-arg factory
)
```

With an interpreter set, the optimizer-authored glue — control flow, string work, arithmetic, imports — runs isolated inside the sandbox. Only predictor construction and predictor calls bridge back to the host, which makes the real LM calls. Any `CodeInterpreter` backend works.

Pass a **zero-argument factory** (as above) rather than a bare instance, so parallel evaluations during optimization each get their own session. A shared instance is stateful and not thread-safe; `Flex` warns if you pass one. Because the interpreter holds live sessions, use the `Flex` as a context manager or call `close()` when you're done:

```python
with dspy.Flex(sig, interpreter=lambda: dspy.PythonInterpreter()) as solve:
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

The interpreter, like the LM, is a **runtime dependency and is not serialized**. If you optimized with a sandbox, re-supply the `interpreter` when you reconstruct the module before calling `load`.

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| Signature` | required | Declares the module's inputs and outputs (e.g. `"invoice -> total_cents: int"`). |
| `tools` | `list[Callable \| dspy.Tool]` | `None` | Tools the generated code may call. With tools, the baseline is a `dspy.RLM`; without, a `dspy.Predict`. |
| `interpreter` | `CodeInterpreter \| Callable[[], CodeInterpreter]` | `None` | When set, generated code runs in a sandbox. Prefer a zero-arg factory for isolation under parallel evaluation. |
| `max_predictor_calls` | `int` | `100` | Cap on bridged LM calls per sandboxed `forward`. `None` disables it; ignored without an `interpreter`. |

## Notes

!!! warning "Experimental"
    `Flex` is marked experimental. The API and the optimization behavior may change between releases; pin a version if you depend on it.

!!! note "Interpreter Requirements"
    The default `PythonInterpreter` requires [Deno](https://deno.land/) for its Pyodide WASM sandbox. See the [RLM page](RLM.md#deno-installation) for installation notes. No interpreter is needed for the default in-process execution.

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
