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
```

## What ReAnchor fits

ReAnchor searches each output's numeric settings against the whole-program
metric. A new setting must improve the training score and pass a fold check.

Before fitting each output, ReAnchor runs the program and records that output's
probabilities on every call. A threshold anywhere between two neighboring P(True) values makes the
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

The current setting stays unless another setting scores strictly better and
passes a fold check. The check splits the training examples into up to five parts.
For each part, ReAnchor picks a setting using the remaining parts and scores
that pick on the held-out part. A new setting is kept only when the combined held-out
score beats the current setting. The check discourages gains confined to small
portions of the dataset, but can accept them when they recur across folds.
Among improving candidates with equal scores, ReAnchor prefers the widest gap,
which leaves the most room on either side. Each search step tries at most
40 gap midpoints, thinning large lists to settings spaced evenly through
the observed values. For Boolean outputs, it also tries threshold zero when
P(True)=0 is observed, since that boundary is the only way to classify those
answers as True.

## Requests and the cache

The numeric settings are not part of the request. Repeated identical requests
reuse cached answers. In a composed program, changing an upstream decision can
change downstream inputs or which predictors run, producing new requests and
additional backend calls. Native-output promotion also changes the request.

`compile` requires caching by default to avoid repeating identical backend
calls; it does not guarantee a fixed request count. Pass `require_cache=False`
to run without this check.

## Native outputs on a generative LM

A generative LM answers a native `bool` or `Literal` output with a single value.
It does not report probabilities, so ReAnchor has nothing to fit. When an output
has an entry in the predictor's `fields`, `Predict` asks the LM for
probabilities instead. `Predict` then applies the threshold or weights to pick
the value, and the output still returns a `bool` or a `Literal` member.

ReAnchor adds this entry for each native output on a generative LM. It keeps the
entry only when the fitted setting beats the native value under the same fold
check. Otherwise it removes the entry, and the output keeps its native behavior.
The `report` row for a kept entry has `"promoted": True`. The first pass with
probabilities sends new requests, because the request changes.

```python
dspy.configure(lm=dspy.LM("your-provider/your-model"))  # Use your model's identifier.
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
  skipped it. Each fitted row has an `observed` entry with the number of calls
  and settings tried; threshold and cut reports also summarize observed values. Its
  `fold_check` entry counts the search steps whose better training score passed
  or failed the fold check.

Your metric may return a number or a `dspy.Prediction` with a `score`.

When you set `log_dir`, ReAnchor writes `report.json` to that folder.

The fitted settings are part of each predictor's `fields`, so the tuned program
saves and loads like any other `Predict` program.

::: dspy.experimental.ReAnchor
    options:
        members: [__init__, compile]
