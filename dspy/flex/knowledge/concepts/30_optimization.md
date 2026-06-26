## How flex optimization works (and how to use feedback)

A `dspy.Flex` module is marked *code-optimizable*. When the program is compiled with
`dspy.GEPA`, the optimizer may rewrite this module's source (`module_src` — the whole
`dspy.Module` subclass) — decomposing the task into focused predictors and plain Python —
instead of only tuning a fixed prompt's instructions. GEPA reflects on the program's execution traces and
on the metric's textual feedback to propose better candidates, keeps the ones that score
higher on the validation set, and writes the winner back to the persisted file.

This is the "spend compute to write better code" loop. Two regimes:

- **With data + a metric:** each candidate rewrite is *scored* on real examples and only kept
  if it does better. Treat the metric's feedback as your primary signal.
- **No data:** the reflection LM rewrites the RLM baseline using this knowledge base + the
  signature, and the last candidate that binds cleanly is accepted (no empirical selection).
  Here, correctness rests entirely on following these principles.

### Diagnosing from failing examples

When you are given failing examples with feedback, do not rewrite everything. Diagnose the
*specific* failure, then make the **smallest change** that fixes the observed failures while
keeping the rest of the working structure intact. Common diagnoses:

- The feedback says the answer is off by a factor or wrong arithmetically → an LM step is doing
  computation that should be plain Python. Have the LM only *extract* the operands, then compute
  in `forward`.
- Wrong field read / `AttributeError` on a predictor result → the sub-signature's output names
  don't match what the code reads. Fix one or the other so they agree.
- A `Prediction` is being returned where a scalar was declared → unwrap (`...=r.field`) and cast
  to the declared type.
- The label/format is subtly wrong → push the exact allowed set / format into the relevant
  predictor's instructions, and normalize the output in Python (`.strip().lower()`, map synonyms).

Generalize from the failures — never hardcode the specific failing inputs' answers.
