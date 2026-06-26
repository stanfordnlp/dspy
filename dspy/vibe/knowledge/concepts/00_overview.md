# Improving a dspy.vibe module

You are improving the *code* of a `dspy.vibe` module. The starting point is almost
always a single trivial `dspy.RLM(...)` call that delegates the whole task to one
recursive-LM black box. Your job is to spend compute to turn that into a better, more
reliable program — decomposing the task, moving determinism into plain Python, and
using focused predictors only where an LM is genuinely needed.

A `dspy.Vibe` module's implementation is a single `dspy.Module` subclass you author, with
two methods:

- `__init__(self)`: calls `super().__init__()` and assigns each predictor it needs to an
  attribute (`self.<name> = dspy.Predict / dspy.ChainOfThought / dspy.ReAct / dspy.RLM(...)`).
  Define a predictor ONLY for a step that genuinely needs a language model. If the whole task
  is deterministic, define no predictors.
- `forward(self, **inputs)`: calls those predictors (`self.<name>(...)`) and/or runs arbitrary
  deterministic Python, then returns `dspy.Prediction(<output fields>=...)` matching the parent
  signature.

## What a GOOD optimized module looks like

- **Decomposed:** the task is split into the smallest predictors that each do one clear
  thing (extract, classify, judge, generate). Each predictor has a tight, well-named
  sub-signature.
- **Mostly Python:** arithmetic, parsing, normalization, sorting, aggregation, validation,
  and control flow are done in plain Python in `forward`. The LM is reserved for genuine
  judgment / extraction / generation.
- **Robust:** it coerces predictor outputs to the declared types, unwraps every
  `dspy.Prediction` before composing, and handles obvious edge cases (empty input, missing
  fields) without raising.
- **Deterministic where possible:** if a step can be done without an LM, it is. A fully
  deterministic task ends up with no predictors in `__init__` and a pure-Python `forward`.
- **Self-contained:** only uses `dspy`, the declared inputs, the context tools available by
  name, and small helpers nested inside `forward`. No module-scope helpers, no hidden
  globals, no I/O.
