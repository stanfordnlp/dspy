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
`bool` or `Literal` output. `RLM` does not support decision outputs, so ReAnchor
does not support RLM programs either.

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

ReAnchor tries settings for each output and keeps the one with the best mean
metric score on the training examples.

ReAnchor builds the settings to try from the training examples. First it runs
the program once and records the probabilities behind each output on every
call. A threshold anywhere between two neighboring P(True) values makes the
same decisions, so ReAnchor tries the midpoint of each gap between them. For
example, when the model only returns P(True) of 0.98 and 1.0, ReAnchor tries
0.5 and 0.99.

- For a Boolean output, ReAnchor tries each midpoint between the observed
  P(True) values.
- For a Score output, the level depends on the mean level index. ReAnchor tries
  each `cut` at the midpoints between the observed mean indexes, and it keeps
  the cuts in order. The cuts choose `.level` and do not change `.value`.
- For a Choice output, ReAnchor moves one option's multiplier at a time. It
  tries the multipliers that fall between the points where that option's pick
  would flip on some call.

The current setting stays unless another setting scores strictly better. When
two settings score the same, ReAnchor picks the one in the widest gap, because
it leaves the most room on either side. When an output returns many distinct
values, ReAnchor thins the list to at most 40 settings, spaced evenly through
the observed values.

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

## Errors

ReAnchor stops at the first error from the program or the metric. On a
generative LM, a malformed answer counts as an error. For example, `Predict`
raises when a `Score` answer leaves out a probability for any level. Use a
model that follows JSON schemas reliably, and set a client `timeout` that fits
your backend's load.

## Results

`compile` returns a copy of the program and leaves your program unchanged. After
`compile`, `report` holds:

- `train_score_before` and `train_score`, the mean metric score on the training
  examples before and after calibration.
- `val_score_before` and `val_score`, the same scores on `valset` when you pass
  one. ReAnchor never fits settings on `valset`.
- `fitted`, one row per output with the fitted value, or the reason ReAnchor
  skipped it. Each fitted row has an `observed` entry with the number of calls,
  the number of distinct values, and the number of settings tried.

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
