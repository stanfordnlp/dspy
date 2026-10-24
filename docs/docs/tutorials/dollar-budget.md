---
sidebar_position: 10
title: Hard Dollar Limit on an Evaluation or Optimization Run
---

# Hard Dollar Limit on an Evaluation or Optimization Run

When you run `dspy.Evaluate` or an optimizer with `num_threads > 1`, concurrent LM calls can spend more than you expect before a post-call total is checked. This short tutorial shows a pattern that refuses a call *before* it reaches the provider if it would exceed a hard dollar budget for that run.

The approach uses an OpenAI-compatible gateway (for example Inferrail) that understands two custom headers:

- a run identifier
- a remaining budget in dollars

Every LM call reserves its worst-case cost before it is sent. If the reservation does not fit, the gateway returns an error that DSPy surfaces as `LMBillingError` and does not retry.

## Setup

```python
import dspy
from dspy.evaluate import Evaluate

# Point DSPy at a gateway that enforces the budget
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
    max_errors=3,  # stop the run after a few refused calls
)

try:
    score = evaluate(program)
    print("Score:", score)
except dspy.LMBillingError as e:
    print("Budget exhausted:", e)
```

With `max_errors` set, a few refused calls cause `Evaluate` to stop instead of continuing to spend. The same `extra_headers` pattern works for optimizer runs when you wrap the compile call in `dspy.context(lm=lm)`.

## Notes

- The gateway must implement the reservation logic; DSPy only forwards the headers.
- A running total checked *after* a call returns cannot prevent overspend under concurrency.
- Test with a small budget first to confirm refusals appear as `LMBillingError` and are not retried.
