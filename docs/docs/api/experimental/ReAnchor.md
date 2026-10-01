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

For every compatible output, ReAnchor saves the original configuration and score,
enables probability-based execution, and fits the numeric settings. It then compares
the fitted behavior against the original using a fold check. If rejected, it restores
the exact original configuration, including removing an entry that was originally absent.
The adapter handles the backend used on each call; calibration does not inspect it.

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
answers as True. When two observed probabilities are adjacent floats, it tries
the upper value as a threshold: `p >= threshold` separates them without a midpoint.

## Flex programs

A [`dspy.Flex`](../modules/Flex.md) builds its predictors from its code on every forward, so
they do not exist until the program runs. ReAnchor finds them during its first pass over the
training set and names each by its attribute in the Flex's code, such as `judge`, or
`triage.judge` for a Flex at `triage`. The fitted settings go into the Flex's
`predictor_fields`, which the Flex applies to each predictor it builds, on top of any `fields`
its code sets. They are saved with the Flex, and cleared when the Flex's code changes.

```python
flex = dspy.Flex(Match)  # starts as one Predict over the signature, named `predict`
tuned = ReAnchor(metric).compile(flex, trainset=trainset, valset=valset)
print(tuned.predictor_fields)  # {'predict': {'match': {'threshold': 0.8}}}
```

An output with no question to ask, meaning no description and no `instructions` entry, cannot
be decided from probabilities, so ReAnchor reports it as skipped. A predictor that a Flex builds
with different decision outputs on different calls is also left out.

### Decomposing the code

Pass a generative `proposer` LM to also rewrite each Flex's code. Each round, an RLM on the
proposer reads every training example: its inputs and expected outputs, the calibrated program's
outputs and metric score, and the probability behind each decision. It can run drafts on chosen
examples to see the probabilities its questions get. It then writes new code that splits the
judgment into narrower decisions and combines them in Python. The predictors keep running on
the Flex's LM or the configured one. On a System One model such as Jev, the proposer is told that
all outputs of one predictor share one billed request, so asking several questions of the same
input costs about the same as one.

ReAnchor calibrates each rewrite as above and keeps it only when the calibrated program scores
higher on `valset`, or the same with fewer calls from the Flex's predictors per example. Without a `valset`, the
choice falls back to the training set, which the proposer has read. A rewrite that fails on any
training example is rejected, and its error goes to the next round.

```python
proposer = dspy.LM("openai/gpt-5.6-sol", max_tokens=32000)
optimizer = ReAnchor(metric, proposer=proposer, rounds=4)
tuned = optimizer.compile(dspy.Flex(Match), trainset=trainset, valset=valset)
print(tuned.module_src)
print(optimizer.report["decomposition"])  # every rewrite, its scores and calls per example, or its error
```

Every calibration pass reruns the Flex's code, and the default `dspy.PythonInterpreter`
starts a new sandbox for every forward. For long searches, a faster `interpreter_factory` such
as `dspy.LocalInterpreter` cuts the time sharply, but it is not a security sandbox.

## Requests and the cache

The numeric settings are not part of the request. Repeated identical requests
reuse cached answers. In a composed program, changing an upstream decision can
change downstream inputs or which predictors run, producing new requests and
additional backend calls. Native-output promotion also changes the request.

`compile` requires caching by default to avoid repeating identical backend
calls; it does not guarantee a fixed request count. Pass `require_cache=False`
to run without this check. The cache check inspects statically bound or globally
configured clients. For programs selecting clients inside `forward()`, disable
the check and manage caching on those clients.

## Native outputs on a generative LM

A generative LM answers a native `bool` or `Literal` output with a single value.
It does not report probabilities, so ReAnchor has nothing to fit. When an output
has an entry in the predictor's `fields`, `Predict` asks the LM for
probabilities instead. `Predict` then applies the threshold or weights to pick
the value, and the output still returns a `bool` or a `Literal` member.

The first pass with probabilities sends new requests, because the request changes.
If the fitted behavior is rejected, restoring the original configuration returns
an unconfigured native field to direct generation.

```python
dspy.configure(lm=dspy.LM("your-provider/your-model"))  # Use your model's identifier.
tuned = ReAnchor(metric).compile(dspy.Predict(Match), trainset=trainset)
```

A System One model such as Jev always returns probabilities. Enabling probability
execution leaves its request unchanged; restoring a field restores its original
decoding settings, not direct generation.

## Errors

ReAnchor stops at the first error from the program or the metric, and raises that error. On a
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
- `decomposition`, with a `proposer`: every code tried, starting with the
  original, with its score on the selection set, predictor calls per example,
  fitted rows, whether it was kept, or the error it failed with.
- `fitted`, one row per output with the fitted value, or the reason ReAnchor
  skipped it. Each fitted row has an `observed` entry with the number of calls
  and settings tried; threshold and cut reports also summarize observed values. Its
  `fold_check` entry counts the search steps whose better training score passed
  or failed the fold check. Per-field `train_score_original` is the score before
  enabling probabilities, `train_score_at_start` is the probability-based baseline,
  and `train_score` describes the retained behavior. Rejected configurations have
  a `skipped` reason instead of a fitted `value`.

Your metric may return a number or a `dspy.Prediction` with a `score`.

When you set `log_dir`, ReAnchor writes `report.json` to that folder.

The fitted settings are part of each predictor's `fields`, so the tuned program
saves and loads like any other `Predict` program.

::: dspy.experimental.ReAnchor
    options:
        members: [__init__, compile]
