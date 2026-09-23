# dspy.experimental.ReAnchor

!!! warning "Experimental API"
    `ReAnchor` is experimental and may change without warning.

`ReAnchor` fits the numeric decision settings in a program to your metric. These
settings are the `threshold`, `cuts`, and `weights` entries in each predictor's
`fields`. They are described in [Decision types and System One models](DecisionTypes.md).
ReAnchor does not change any instructions, descriptions, or demos.

You give ReAnchor a metric, a program, and training examples. The program can be
a single `dspy.Predict` or any module that holds predictors with decision
outputs. A decision output is a `Noul`, `Score`, or `Choice` output, or a native
`bool` or `Literal` output.

```python
import dspy
from dspy.experimental import ReAnchor, TypeSafe


class Match(dspy.Signature):
    """Decide whether two product listings describe the same item."""

    pair: str = dspy.InputField(desc="Two listings.")
    match: bool = dspy.OutputField(desc="Are they the same item?")


def metric(example, prediction, trace=None):
    return float(prediction.match == example.match)


dspy.configure(lm=TypeSafe("jev-latest"))
optimizer = ReAnchor(metric)
tuned = optimizer.compile(dspy.Predict(Match), trainset=trainset, valset=valset)
print(optimizer.report)
print(ReAnchor.source(tuned))
```

## What ReAnchor fits

ReAnchor tries each setting and keeps the one with the best mean metric score on
the training examples.

- For a Boolean output, it tries each `threshold` from 0.05 to 0.95 in steps of
  0.05. The current threshold stays unless another value scores strictly
  better. When two values score the same, ReAnchor picks the one nearer to 0.5.
- For a Score output, it tries `cuts` between the levels and keeps them in
  order. The cuts choose `.level` and do not change `.value`. When two settings
  score the same, ReAnchor keeps the default cuts.
- For a Choice output, it tries a multiplier from 0.1 to 10 for each option.
  When two settings score the same, ReAnchor picks the multipliers nearer to 1.

Before ReAnchor tries settings for an output, it checks whether your metric
reads that output. It scores the output at its current setting. Then it pushes
every answer to one end and scores it again, e.g., every answer True and then
every answer False. When no example's score changes, ReAnchor skips the output
and records the skip in `report`.

## Requests and the cache

The settings are not part of the request. The backend answers each training
example once, and each later pass reads those answers from the cache. For this
reason, `compile` raises an error when a predictor's client has its cache turned
off. Pass `require_cache=False` to run anyway. Each setting ReAnchor tries then
sends a new request for every training example.

## Native outputs on a generative LM

A generative LM answers a native `bool` or `Literal` output with a single value.
It does not report probabilities, so ReAnchor has nothing to fit. When an output
has an entry in the predictor's `fields`, `Predict` asks the LM for
probabilities instead. `Predict` then applies the threshold or weights to pick
the value, and the output still returns a `bool` or a `Literal` member.

ReAnchor adds this entry for each native output on a generative LM. It keeps the
entry only when the fitted setting scores strictly better than the native value
did. Otherwise it removes the entry, and the output keeps its native behavior.
The `report` row for a kept entry has `"promoted": True`. The first pass with
probabilities sends new requests, because the request changes.

```python
dspy.configure(lm=dspy.LM("openai/gpt-6-luna"))
tuned = ReAnchor(metric).compile(dspy.Predict(Match), trainset=trainset)
```

A System One model such as Jev always returns probabilities, so this step does
not apply there.

## Results

`compile` returns a copy of the program and leaves your program unchanged. After
`compile`, `report` holds:

- `train_score_before` and `train_score`, the mean metric score on the training
  examples before and after calibration.
- `val_score_before` and `val_score`, the same scores on `valset` when you pass
  one. ReAnchor never fits settings on `valset`.
- `fitted`, one row per output with the fitted value, or the reason ReAnchor
  skipped it.

Your metric may return a number or a `dspy.Prediction` with a `score`.

`ReAnchor.source(program)` writes each predictor's signature as a class, and
after it the line that sets that predictor's `fields`. You can paste this into
your code. When you set `log_dir`, ReAnchor writes `report.json` and
`source.py` to that folder.

The fitted settings are part of each predictor's `fields`, so the tuned program
saves and loads like any other `Predict` program.

::: dspy.experimental.ReAnchor
    options:
        members: [__init__, compile, source]
