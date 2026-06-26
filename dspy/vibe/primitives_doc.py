PRIMITIVES_CATALOG: str = """\
You are authoring two code artifacts for a `dspy.Flex` module:

1) A `PREDICTORS` dict literal at module scope mapping attribute names to predictor
   instances. Each entry is a constructed `dspy.Predict` / `dspy.ChainOfThought` /
   `dspy.ReAct` / `dspy.RLM` over a sub-signature string. Include a predictor ONLY
   for a step that genuinely needs a language model. If the task needs no LM at
   all, emit an empty `PREDICTORS = {}`.

2) A `def forward(self, **inputs):` function whose body returns
   `dspy.Prediction(<output_fields...>)` matching the declared signature. The body
   may call the predictors (`self.<name>(...)`) AND/OR run arbitrary deterministic
   Python — parsing, arithmetic, regex, string and data-structure manipulation,
   control flow, and small helper functions defined *inside* `forward`. A module
   may be fully deterministic (no predictors), fully LM-driven, or a mix. Prefer
   plain Python for anything that doesn't require an LM's judgment.

Available DSPy primitives:

- `dspy.Predict("a, b -> c, d")`, one LM call. `result = p(a=x, b=y)` returns a
  `dspy.Prediction` object; the declared outputs are attributes on it:
  `result.c`, `result.d`. Signature strings use snake_case names.

- `dspy.ChainOfThought("a -> b")`, like Predict but adds an implicit `reasoning`
   output field; useful when the answer benefits from explicit step-by-step thought.

- `dspy.ReAct(signature_str, tools=[tool1, tool2])`, multi-step reasoning + tool
   use. `tools` is a list of plain Python callables or `dspy.Tool` instances.

- `dspy.RLM("context, query -> output", max_iterations=10)`, Recursive Language
   Model: the LLM writes Python code in a sandboxed REPL to programmatically
   explore very large or structured inputs (long documents, big JSON blobs,
   tables) and build up an answer iteratively. Use when one of the input fields
   is too large or too structured for a single direct prompt. Optional kwargs:
   `max_iterations` (REPL steps), `max_llm_calls` (sub-LLM calls), `sub_lm`
   (cheaper LM for sub-queries), `tools` (extra callables for the REPL).

- `dspy.Tool(func)`, wraps a callable as a tool (only used inside ReAct).

- `dspy.Prediction(field_a=value_a, field_b=value_b)`, construct the final return
   value. The keyword names MUST match the parent signature's output_fields
   exactly, and each `value_*` must be a plain value of the declared output
   type — never a `dspy.Prediction` you got back from a predictor call.

Rules you MUST follow:

- Reference every predictor as `self.<name>` (e.g. `self.classify`). The PREDICTORS
  entries will be attached to the `self` instance before `forward` is called.

- Do NOT access dunder attributes (anything matching `__*__`). Use the public API.

- Every `while` loop must be bounded, either with an explicit `break`, or rewrite
  to `for _ in range(N):`. Pick a small `N` (e.g. 5).

- Always return `dspy.Prediction(...)` from `forward`. Never raise from `forward`
  unless the input itself is malformed.

- Use predictors only where an LM is genuinely needed (extraction, classification,
  generation, open-ended reasoning); do everything else — math, parsing, formatting,
  validation, control flow — in plain Python. When no step needs an LM, write a
  pure-Python `forward` with `PREDICTORS = {}`.

- Define any helper functions NESTED inside `forward` (not at module scope), so the
  persisted module file and the live module execute identically.

- Keep the design composable: prefer two short predictors with clear signatures
  over one all-knowing prompt.

- Do not hardcode example outputs or magic constants from the task description.

- Pick the right primitive for each step. Rough guide:
    * `dspy.Predict` for direct, single-call transformations.
    * `dspy.ChainOfThought` when the answer benefits from explicit step-by-step
      reasoning.
    * `dspy.ReAct` when external tools must be called during reasoning.
    * `dspy.RLM` when an input field is a large blob (long document, big JSON,
      multi-table data) that doesn't fit naturally in a single prompt — RLM
      lets the LLM explore it iteratively via a sandboxed REPL.

Common patterns (study these before writing `forward`):

1) Unwrap every predictor output. A predictor call returns a `dspy.Prediction`
   whose declared output fields are attributes on it. Always read those
   attributes before composing further calls or the final return.

   ```
   PREDICTORS = {
       "parse":   dspy.Predict("invoice -> qty: int, unit_price: float"),
       "shipping": dspy.Predict("invoice -> shipping_dollars: float"),
   }

   # RIGHT — pull fields off each predictor result, then build the final Prediction
   def forward(self, **inputs):
       parsed = self.parse(invoice=inputs["invoice"])
       ship = self.shipping(invoice=inputs["invoice"])
       cents = round((parsed.qty * parsed.unit_price + ship.shipping_dollars) * 100)
       return dspy.Prediction(total_cents=int(cents))

   # WRONG — passes the whole inner Prediction as the field value, producing
   # Prediction(total_cents=Prediction(total_cents=...)) at runtime
   def forward(self, **inputs):
       result = self.parse(invoice=inputs["invoice"])
       return dspy.Prediction(total_cents=result)            # bug: no `.field`
   ```

2) Coerce to the declared output type. If the parent signature declares
   `total_cents: int`, cast in Python (`int(...)`, `float(...)`, `str(...)`,
   `bool(...)`, list comprehensions for `list[T]`) before returning. LM
   string outputs do NOT auto-cast to the declared type.

3) Compose predictor outputs into the next call by passing the unwrapped
   attribute, never the whole Prediction. `self.next(x=prev.x)` — not
   `self.next(x=prev)`.

4) Do arithmetic, sorting, filtering, and aggregation in plain Python in
   `forward()`. Use the LM to extract / classify / generate; let Python do
   the math. This is more reliable and easier to debug than asking the LM
   to do both at once.

5) A fully-deterministic task needs no LM: emit `PREDICTORS = {}` and implement
   `forward` in plain Python, with any helpers nested inside it.

   ```
   PREDICTORS = {}

   def forward(self, **inputs):
       def slugify(text):
           cleaned = "".join(c.lower() if c.isalnum() else "-" for c in text)
           return "-".join(part for part in cleaned.split("-") if part)

       return dspy.Prediction(slug=slugify(inputs["title"]))
   ```
"""


KNOWLEDGE_BASE: str = """\
You are improving the *code* of a dspy.Flex module. The starting point is almost
always a single trivial `dspy.RLM(...)` call that delegates the whole task to one
recursive-LM black box. Your job is to spend compute to turn that into a better,
more reliable program — decomposing the task, moving determinism into plain Python,
and using focused predictors only where an LM is genuinely needed.

What a GOOD optimized module looks like:

- Decomposed: the task is split into the smallest predictors that each do one
  clear thing (extract, classify, judge, generate). Each predictor has a tight,
  well-named sub-signature.
- Mostly Python: arithmetic, parsing, normalization, sorting, aggregation,
  validation, and control flow are done in plain Python in `forward`. The LM is
  reserved for genuine judgment/extraction/generation.
- Robust: it coerces predictor outputs to the declared types, unwraps every
  `dspy.Prediction` before composing, and handles obvious edge cases (empty input,
  missing fields) without raising.
- Deterministic where possible: if a step can be done without an LM, it is. A fully
  deterministic task ends up with `PREDICTORS = {}` and a pure-Python `forward`.
- Self-contained: only uses `dspy`, the declared inputs, and small helpers nested
  inside `forward`. No module-scope helpers, no hidden globals, no I/O.

Giving a predictor task-specific instructions (allowed labels, domain rules, output
format): build its signature WITH instructions and pass that to the predictor —
`dspy.Predict(dspy.Signature("text -> intent", "Classify into exactly one of: ..."))`
(same for `dspy.ChainOfThought`/`dspy.RLM`). Carry over any important guidance (e.g. the
exact allowed label set) from the task description into the predictor's instructions so the
model still sees it. Do NOT pass instructions as a second positional argument to
`dspy.Predict(...)` / `dspy.ChainOfThought(...)` — that argument is not the instructions and
will raise; instructions belong inside the signature.

Worked example — evolving the RLM baseline for an invoice-total task:

```
# BASELINE (what you start from): one opaque RLM call.
PREDICTORS = {"rlm": dspy.RLM("invoice: str -> total_cents: int")}

def forward(self, **inputs):
    result = self.rlm(invoice=inputs["invoice"])
    return dspy.Prediction(total_cents=result.total_cents)
```

```
# GOOD (a strong optimization): LM extracts the line items, Python does the math.
PREDICTORS = {
    "extract": dspy.Predict("invoice: str -> qty: int, unit_price_cents: int, shipping_cents: int"),
}

def forward(self, **inputs):
    e = self.extract(invoice=inputs["invoice"])
    total = int(e.qty) * int(e.unit_price_cents) + int(e.shipping_cents)
    return dspy.Prediction(total_cents=int(total))
```

Anti-patterns to AVOID (these are common ways an optimization goes wrong):

- Asking one LM call to both extract AND compute (e.g. "read the invoice and return
  the total"). Split it: LM extracts the numbers, Python computes. LMs are
  unreliable at arithmetic.
- Leaving everything inside a single `dspy.RLM` when the task fits a couple of
  direct `dspy.Predict` calls — RLM's REPL loop is expensive and overkill for small,
  well-structured inputs. Only keep RLM when an input field is a large/structured
  blob that truly needs iterative exploration.
- Returning a whole `dspy.Prediction` as an output value
  (`dspy.Prediction(x=self.p(...))`) instead of unwrapping (`...=self.p(...).x`).
- Not casting to the declared output type, so a string `"42"` is returned where
  `int` was declared.
- Hardcoding answers, magic constants, or example values lifted from the task
  description / failing examples — generalize, don't memorize.
- Unbounded `while` loops, accessing dunder attributes, or defining helpers at
  module scope.

When you are given failing examples, diagnose the specific failure (wrong field
read, missing coercion, an LM step that should be Python, an under-decomposed
prompt) and make the smallest change that fixes the observed failures while keeping
the rest of the working structure intact.
"""
