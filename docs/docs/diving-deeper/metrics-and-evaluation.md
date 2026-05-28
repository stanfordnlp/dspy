# Metrics and evaluation

## Intent

A DSPy metric is the function that turns a prediction into a number an optimizer can chase, and `dspy.Evaluate` is the harness that runs that metric across a dataset in parallel. This page covers the metric contract optimizers expect, the built-in string and LLM-judge metrics that ship with DSPy, and how `Evaluate` orchestrates a scoring run.

Read this when you’re writing your own metric, swapping in an LLM-as-judge for a rule-based one, choosing knobs on `Evaluate`, or troubleshooting why an optimizer reads your metric the way it does.

## Design decisions

### 1. A metric is any callable that takes `(gold, pred, ...)` and returns a score

The contract is duck-typed. No base class to subclass, no decorator to register a metric, no `@metric` marker. `Evaluate` calls whatever you pass it; the optimizer does the same. The benefit: an existing scoring function from any NLP or QA library wraps as a DSPy metric in one line. The price: the contract isn’t enforced — a callable that returns the wrong type fails at evaluation time, not at registration.

### 2. The full optimizer-facing signature is `(gold, pred, trace=None, pred_name=None, pred_trace=None)`

Most metrics declare only `(gold, pred)` and Python’s default-arg rules let optimizers pass the rest harmlessly. The optimizer fills in `trace` when it wants the metric to see how the program reached its answer, and fills in `pred_name` / `pred_trace` when it wants per-predictor scoring (GEPA does both). A metric written today against `(gold, pred)` keeps working when a future optimizer wants more — that’s why the extra args are tail-positioned defaults.

### 3. The return type can be `bool`, `float`, or `dspy.Prediction(score, feedback)` — each is handled differently

`Evaluate` aggregates booleans into a percentage and floats into a mean. A `Prediction` return is unpacked: `score` is averaged like a float, `feedback` is read only by GEPA. The three-shape return matters because the simplest metrics shouldn’t have to ceremoniously wrap their output in `Prediction`, but the optimizers that need richer signals shouldn’t be locked out either.

### 4. The `feedback` channel is for GEPA, not for `Evaluate`‘s score

A metric can attach a natural-language explanation to every score; only GEPA’s reflective loop reads it. `Evaluate` ignores it. The split keeps two concerns clean: “how good was this prediction” is one signal, “what’s a useful critique” is a different signal that costs nothing to omit when no optimizer is listening.

### 5. `trace` is the lever that lets a metric behave one way inside an optimizer and another way at eval time

The `SemanticF1` and `CompleteAndGrounded` judges return the raw F1 when `trace is None` (evaluation) and a binarized `score >= threshold` when `trace is not None` (optimization). The pattern: at eval time you want a continuous score; at optimization time you want a clean pass/fail signal the search can chase. The `trace` argument tells the metric which mode it’s in without forcing two separate functions.

### 6. Built-in string metrics centralize on `normalize_text`

`EM`, `F1`, `HotPotF1`, and the metric-shaped wrappers all call `normalize_text` on both prediction and reference: NFD-unicode, lowercase, drop English articles (`a` / `an` / `the`), strip punctuation, collapse whitespace. A single canonical normalization keeps token-level F1 and exact match operating on identical inputs — which matters because `answer_exact_match` switches between them based on its `frac` argument.

### 7. LLM judges are `dspy.Module` subclasses, not plain functions

`SemanticF1` and `CompleteAndGrounded` subclass `dspy.Module` so they can be inspected, traced, and even optimized like any other DSPy program. You could write a judge as a free function around `dspy.Predict`, but you’d lose the trace and history a Module gives you for free. The pattern: write a `dspy.Signature` whose outputs include `score: float` (and optionally `feedback: str`), wrap it in a Module whose `forward` returns a `Prediction(score, feedback)`.

### 8. `Evaluate` parallelizes through `ParallelExecutor`, not its own thread pool

The same executor that `dspy.Parallel` and `Module.batch` use. The benefit: settings propagation, error counting, straggler detection, and progress reporting are one implementation. A `dspy.context(lm=...)` block around an `Evaluate` call is honored inside every worker, because `ParallelExecutor` snapshots and re-applies `thread_local_overrides`.

### 9. Metric failures get a `failure_score`, not an exception

When the metric raises (or returns `None`), `Evaluate` substitutes `failure_score` (default `0.0`) for that example and keeps going. The reason: in a long evaluation, one bad example shouldn’t lose the score on the other 999. A `max_errors` knob caps how many failures the harness will tolerate before stopping.

### 10. `EvaluationResult` is the single return shape

`.score` is the aggregate (percent of `True` for booleans, mean for floats and `Prediction.score`), `.results` is the list of `(example, prediction, score)` triples. The legacy `return_outputs` / `return_all_scores` constructor kwargs were removed; passing them now raises `ValueError` with a migration message. One return shape means downstream code doesn’t have to switch on which keyword the caller used.

## API walkthrough

Grouped by what you’re trying to do.

### Metric anatomy

How DSPy calls your metric and what it expects back.

**The call signature** — `metric(gold, pred, trace=None, pred_name=None, pred_trace=None)`
`gold` is the labeled `dspy.Example`; `pred` is the `dspy.Prediction` the program produced. `trace` is a list of `(predictor, inputs, outputs)` triples covering the whole execution — filled in by optimizers when they want the metric to see how the program got there. `pred_name` and `pred_trace` narrow the same view to one predictor; GEPA uses these when scoring a single step of a multi-step program.

**`dspy.Prediction(score, feedback)`** — the score-with-feedback return shape
A `Prediction` with at least a `score` field, optionally a `feedback` field. `Evaluate` reads only `score`; GEPA reads both. Returning `Prediction(score=...)` from a plain metric works fine — `feedback` is optional, and the wrapper costs almost nothing when you only want a score.

### Built-in metrics for strings and tokens

The standard library of rule-based metrics, in `dspy/evaluate/metrics.py`.

**`dspy.evaluate.normalize_text(s: str)` → `str`**
NFD-unicode, lowercase, strip the English articles `a` / `an` / `the`, strip punctuation, collapse whitespace. The single canonical normalization shared by every string-comparison metric below.

**`dspy.evaluate.EM(prediction, answers_list)` → `bool`**
Exact match after `normalize_text` is applied to both sides. Returns `True` if any reference in `answers_list` matches. Raises if `answers_list` isn’t a list. The smallest metric primitive; useful as a building block in custom wrappers.

**`dspy.evaluate.answer_exact_match(example, pred, trace=None, frac=1.0)` → `bool`**
The metric-shaped wrapper. Reads `pred.answer` and `example.answer`, normalizes both, and dispatches to `EM` when `frac >= 1.0` or to F1-thresholded matching when `frac < 1.0`. Handles `example.answer` being a single string or a list of acceptable references.

**`dspy.evaluate.answer_passage_match(example, pred, trace=None)` → `bool`**
Retrieval evaluation. Returns `True` if any passage in `pred.context` contains any reference from `example.answer`. Uses a DPR-style normalizer for passages (which preserves more text) while still using `normalize_text` for answers.

**`F1` / `HotPotF1`** — token-level scorers (internal)
`F1` does token-level F1 over normalized strings, picking the max across references. `HotPotF1` adds one HotPotQA-specific rule: if either normalized side is `yes` / `no` / `noanswer` and the two disagree, return `0`. Both feed into `answer_exact_match` and are rarely called directly.

### LLM-as-judge metrics

For tasks where rule-based scoring can’t capture the right notion of correctness. Both live in `dspy/evaluate/auto_evaluation.py` and are `dspy.Module` subclasses.

**`dspy.evaluate.SemanticF1(threshold=0.66, decompositional=False)`**
A Module. `forward(example, pred, trace=None)` runs a `ChainOfThought` over `SemanticRecallPrecision` (or the decompositional variant) and asks the LM for `precision` and `recall` against `example.response` and `pred.response`. Computes `f1_score(precision, recall)`. Returns `Prediction(score=f1)` at eval time and `Prediction(score=(f1 >= threshold))` at optimization time — the same instance serves both modes.

**`dspy.evaluate.CompleteAndGrounded(threshold=0.66)`**
Runs two `ChainOfThought` calls: one for completeness (does `pred.response` cover `example.response`?), one for groundedness (is `pred.response` supported by `pred.context`?). Returns `Prediction(score=f1_score(groundedness, completeness))` with the same trace-based binarization as `SemanticF1`. Designed for retrieval-augmented programs.

**The general LLM-judge pattern.** Write a `dspy.Signature` whose outputs include `score: float` (and optionally `feedback: str`). Wrap it in a `dspy.Module` whose `forward(example, pred, trace=None)` calls the judge predictor and returns `Prediction(score, feedback)`. Use the `trace is None` toggle when you want different behavior inside optimization. That’s the whole recipe — both built-in judges are 20-line modules over this shape.

### The Evaluate harness

The runner that takes a program + metric + dataset and produces a score.

**`dspy.Evaluate(*, devset, metric=None, num_threads=None, display_progress=False, display_table=False, max_errors=None, provide_traceback=None, failure_score=0.0, save_as_csv=None, save_as_json=None)`**
Keyword-only constructor. `metric` can be passed here or deferred to the call. `display_table=N` truncates the displayed DataFrame to `N` rows; `display_table=True` shows the whole thing; `False` shows nothing. `save_as_csv` / `save_as_json` write per-example results to disk for later inspection. Passing the removed `return_outputs` kwarg raises a `ValueError`.

**`Evaluate.__call__(program, metric=None, devset=None, ...)`** → `EvaluationResult`
Submits `(program, example)` pairs to a `ParallelExecutor`. Each worker re-applies the parent’s `thread_local_overrides` and runs `program(**example.inputs())`, then calls the metric with `(example, prediction)`. The aggregate becomes `.score`: percent of `True` for boolean returns, mean for float returns, mean of `score` for `Prediction` returns. Per-example results land in `.results` as `(example, prediction, score)` triples.

**`dspy.evaluate.EvaluationResult`**
A `dspy.Prediction` subclass with two fields: `.score` (aggregate) and `.results` (list of triples). `repr` shows the score and the result count rather than dumping the full list, so logging an `EvaluationResult` doesn’t print a megabyte of output.

## Cross-links

- [Settings and `context()`](settings-and-context.md) — how `Evaluate`‘s workers inherit a `dspy.context(...)` override around the call.
- [Built-in module variants](built-in-module-variants.md) — `dspy.Parallel` and `Module.batch` use the same `ParallelExecutor` under the hood.
- [Optimizers: choosing one](choosing-an-optimizer.md) — every optimizer compiles against a metric defined here; the `Prediction(score, feedback)` shape GEPA expects is documented above.
