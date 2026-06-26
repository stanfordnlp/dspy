## Patterns to follow

1. **Unwrap every predictor output.** A predictor call returns a `dspy.Prediction`; read its
   declared output fields off as attributes (`r.foo`) before composing further calls or building
   the final return. Never pass a whole `dspy.Prediction` as a field value.

2. **Coerce to the declared output type.** If the parent signature declares `total_cents: int`,
   cast in Python (`int(...)`, `float(...)`, `str(...)`, `bool(...)`, comprehensions for
   `list[T]`) before returning. LM string outputs do not auto-cast.

3. **Let the LM extract/judge; let Python compute.** Do arithmetic, sorting, filtering, and
   aggregation in plain Python in `forward`. This is more reliable and easier to debug than
   asking one LM call to both read *and* calculate.

4. **Compose by passing unwrapped attributes.** `self.next(x=prev.x)` — never `self.next(x=prev)`.

5. **A fully-deterministic task needs no LM.** Emit `PREDICTORS = {}` and implement `forward` in
   plain Python, with any helpers nested inside it.

6. **Push task rules into predictor instructions.** Allowed labels, domain rules, and output
   formats go inside the predictor's `dspy.Signature(..., instructions=...)`, then normalize the
   result in Python.

## Anti-patterns to AVOID

- Asking one LM call to both extract AND compute (e.g. "read the invoice and return the total").
  Split it: LM extracts the numbers, Python computes. LMs are unreliable at arithmetic.
- Leaving everything inside a single `dspy.RLM` when the task fits a couple of direct
  `dspy.Predict` calls — RLM's REPL loop is expensive and overkill for small, well-structured
  inputs. Keep RLM only when an input field is a large/structured blob that truly needs iterative
  exploration.
- Returning a whole `dspy.Prediction` as an output value (`dspy.Prediction(x=self.p(...))`)
  instead of unwrapping (`...=self.p(...).x`).
- Not casting to the declared output type, so a string `"42"` is returned where `int` was declared.
- Hardcoding answers, magic constants, or example values lifted from the task description /
  failing examples — generalize, don't memorize.
- Unbounded `while` loops (bound every loop with a `break` or rewrite as `for _ in range(N)`),
  accessing dunder attributes (`__*__`), defining helpers at module scope, or doing any I/O.
- Raising from `forward` except when the input itself is malformed — always return a
  `dspy.Prediction`.
