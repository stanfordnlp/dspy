---
title: Hard Dollar Limit on an Evaluation or Optimization Run
---

# Hard Dollar Limit on an Evaluation or Optimization Run

When you run `dspy.Evaluate` or an optimizer with `num_threads > 1`, concurrent LM calls can spend more than you expect before a post-call total is checked. This short tutorial shows a pattern that refuses a call *before* it reaches the provider if it would exceed a hard dollar budget for that run.

The approach uses an OpenAI-compatible gateway (for example [Inferrail](https://github.com/inferrail/inferrail)) that understands two custom headers:

- a run identifier
- a remaining budget in dollars

Every LM call reserves its worst-case cost before it is sent. If the reservation does not fit, the gateway must return an error that DSPy surfaces as a non-retryable billing error (typically `dspy.LMBillingError` or a plain `Exception` depending on configuration). Verify the exact response format against your DSPy version, because rate-limit style errors are often retried.

## Setup

```python
import dspy
from dspy.evaluate import Evaluate

# Point DSPy at a gateway that enforces the budget.
# The gateway must implement the reservation logic and return a non-retryable error.
lm = dspy.LM(
    "openai/gpt-4o-mini",
    api_base="http://localhost:8000/v1",  # your gateway
    api_key="not-needed",
    extra_headers={
        "X-Run-Id": "eval-run-001",
        "X-Budget-USD": "0.50",  # hard limit for this run
    },
)
dspy.configure(lm=lm)
```

## Running an evaluation under the budget

```python
# Simple metric and dataset for illustration
def metric(example, pred, trace=None):
    return float(example.answer == pred.answer)

devset = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    # ... more examples
]

program = dspy.ChainOfThought("question -> answer")

evaluate = Evaluate(
    devset=devset,
    metric=metric,
    num_threads=4,
    max_errors=3,  # after this many failures, Evaluate raises
)

try:
    score = evaluate(program)
    print("Score:", score)
except Exception as e:
    # With max_errors, Evaluate raises a plain Exception after the threshold.
    # Individual LMBillingError (or equivalent) failures are recorded as failed
    # examples and scored with failure_score before the threshold is reached.
    print("Evaluation stopped (budget or other errors):", e)
```

Per-example billing refusals are treated as failed examples and scored with the configured `failure_score`. Only after `max_errors` failures does `Evaluate` raise a plain `Exception` that cancels the run. This is different from a single top-level `LMBillingError`.

## Optimizer runs

Binding only the compile call is not sufficient. Optimizers such as MIPROv2 store their own task and prompt models at construction time, and a program may have an attached model that takes precedence over the outer context. To keep every call behind the gateway:

MIPROv2 requires the optional Optuna dependency. Install it before running the example:

```bash
pip install "dspy[optuna]"
```

```python
# Provide a training set (required by most optimizers)
trainset = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="What is 3+5?", answer="8").with_inputs("question"),
    # ... more training examples
]

# Create the optimizer and any programs with the gateway LM
optimizer = dspy.MIPROv2(metric=metric, auto="light")
optimizer.task_model = lm
optimizer.prompt_model = lm

# Or attach the LM to the program before compiling
program.set_lm(lm)

compiled = optimizer.compile(program, trainset=trainset)
```

Always verify with a small budget that no calls escape to a direct provider. Without the Optuna extra, compilation can make gateway calls before raising `ImportError`.

## Notes

- The gateway must implement the reservation logic and a verified non-retryable error response; DSPy only forwards the headers.
- A running total checked *after* a call returns cannot prevent overspend under concurrency.
- Test with a small budget first and confirm the exact exception type raised by your DSPy version.
- This tutorial is a pattern, not a complete turnkey solution. You must supply and test the gateway.
- Install optional extras (such as `dspy[optuna]` for MIPROv2) before running optimizer examples so missing dependencies do not consume budget.
