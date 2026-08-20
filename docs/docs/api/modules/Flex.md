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

The generated code always runs in a sandbox (`interpreter_factory` defaults to `dspy.PythonInterpreter`), so the example above needs [Deno](https://deno.land/) installed — see [Sandboxed Execution](#sandboxed-execution).

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

## Sandboxed Execution

`Flex` always runs its generated code in a sandbox — never in the host Python process. `interpreter_factory` defaults to `dspy.PythonInterpreter` (Deno/Pyodide) and must be a **zero-argument factory** returning a fresh `CodeInterpreter`; a bare instance is not accepted, so parallel evaluations receive isolated sessions. The factory is called once per sandbox session, including separate sessions requested by nested code-executing modules. The code is authored by the reflection model, so isolating it keeps it from running with your host's full permissions. With the default interpreter, optimizer-authored control flow, string work, arithmetic, and supported imports run inside the sandbox, and only provided-tool calls, predictor construction, and predictor calls bridge back to the host, which makes the real LM calls.

Because the default builds a `PythonInterpreter`, *running* a `Flex` needs [Deno](https://deno.land/) installed; without it, the call raises.

```python
solve = dspy.Flex(
    "invoice: str -> total_cents: int",
    interpreter_factory=lambda: dspy.PythonInterpreter(),  # the default; swap in your own CodeInterpreter factory here
)
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

## Saving and Loading

A `Flex` serializes its `module_src`, so saving and loading a program restores the optimized code:

```python
optimized.save("solver.json")

restored = dspy.Flex("invoice: str -> total_cents: int")
restored.load("solver.json")  # rebinds the saved module_src
```

The interpreter is a **runtime dependency and is not serialized**. Reconstructing with `dspy.Flex(signature)` restores the default sandbox automatically; if you optimized with a custom `interpreter_factory`, pass the same one when you reconstruct the module before calling `load`.

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| Signature` | required | Declares the module's inputs and outputs (e.g. `"invoice -> total_cents: int"`). |
| `tools` | `list[Callable \| dspy.Tool]` | `None` | Tools the generated code may call. With tools, the baseline is a `dspy.RLM`; without, a `dspy.Predict`. |
| `interpreter_factory` | `Callable[[], CodeInterpreter]` | `PythonInterpreter` | Zero-arg factory returning a fresh `CodeInterpreter` for each sandbox session; defaults to `dspy.PythonInterpreter` (needs Deno). A bare interpreter instance is not accepted. Supported Python and libraries are interpreter-dependent. |
| `max_predictor_calls` | `int \| None` | `100` | Maximum number of predictor calls the generated code can make in one `forward` — a guard against runaway loops. `None` removes the limit. |

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
