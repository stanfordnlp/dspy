PRIMITIVES_CATALOG: str = """\
You are authoring ONE Python source string: a single `dspy.Module` subclass that
implements the user's signature. It has exactly two methods:

1) `def __init__(self):` — calls `super().__init__()` and assigns each predictor it
   needs to an attribute, e.g. `self.classify = dspy.Predict("text -> label")`.
   Each predictor is a constructed `dspy.Predict` / `dspy.ChainOfThought` /
   `dspy.ReAct` / `dspy.RLM` over a sub-signature string. Define a predictor ONLY
   for a step that genuinely needs a language model. If the task needs no LM at
   all, define no predictors (an `__init__` with just `super().__init__()`).

2) `def forward(self, **inputs):` — returns `dspy.Prediction(<output_fields...>)`
   matching the declared signature. The body may call the predictors
   (`self.<name>(...)`) AND/OR run arbitrary deterministic Python — parsing,
   arithmetic, regex, string and data-structure manipulation, control flow, and
   small helper functions defined *inside* `forward`. A module may be fully
   deterministic (no predictors), fully LM-driven, or a mix. Prefer plain Python
   for anything that doesn't require an LM's judgment.

The whole class is one optimizable unit — `__init__` and `forward` are rewritten
together, so a predictor renamed in one place must be updated in the other.

Skeleton (the shape of every module you write):

```
class SolveModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict("invoice: str -> qty: int, unit_price_cents: int")

    def forward(self, **inputs):
        e = self.extract(invoice=inputs["invoice"])
        return dspy.Prediction(total_cents=int(e.qty) * int(e.unit_price_cents))
```

`dspy` is already in scope — do NOT add `import` statements at module scope (import
stdlib modules like `re` *inside* `forward` if you need them).

Available DSPy primitives:

- `dspy.Predict("a, b -> c, d")`, one LM call. `result = self.p(a=x, b=y)` returns a
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

- Define every predictor in `__init__` as `self.<name> = dspy.Predict(...)` and call
  it in `forward` as `self.<name>(...)`.

- Do NOT access dunder attributes (anything matching `__*__`). Use the public API.

- Every `while` loop must be bounded, either with an explicit `break`, or rewrite
  to `for _ in range(N):`. Pick a small `N` (e.g. 5).

- Always return `dspy.Prediction(...)` from `forward`. Never raise from `forward`
  unless the input itself is malformed.

- Use predictors only where an LM is genuinely needed (extraction, classification,
  generation, open-ended reasoning); do everything else — math, parsing, formatting,
  validation, control flow — in plain Python. When no step needs an LM, write a
  pure-Python `forward` and define no predictors in `__init__`.

- Define any helper functions NESTED inside `forward` (not at module scope), and
  add no module-scope code other than the single class definition, so the persisted
  module file and the live module execute identically.

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
   class TotalModule(dspy.Module):
       def __init__(self):
           super().__init__()
           self.parse = dspy.Predict("invoice -> qty: int, unit_price: float")
           self.shipping = dspy.Predict("invoice -> shipping_dollars: float")

       # RIGHT — pull fields off each predictor result, then build the final Prediction
       def forward(self, **inputs):
           parsed = self.parse(invoice=inputs["invoice"])
           ship = self.shipping(invoice=inputs["invoice"])
           cents = round((parsed.qty * parsed.unit_price + ship.shipping_dollars) * 100)
           return dspy.Prediction(total_cents=int(cents))

       # WRONG — passing the whole inner Prediction as the field value, producing
       # Prediction(total_cents=Prediction(total_cents=...)) at runtime:
       #     return dspy.Prediction(total_cents=self.parse(invoice=inputs["invoice"]))
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

5) A fully-deterministic task needs no LM: define no predictors and implement
   `forward` in plain Python, with any helpers nested inside it.

   ```
   class SlugifyModule(dspy.Module):
       def __init__(self):
           super().__init__()

       def forward(self, **inputs):
           def slugify(text):
               cleaned = "".join(c.lower() if c.isalnum() else "-" for c in text)
               return "-".join(part for part in cleaned.split("-") if part)

           return dspy.Prediction(slug=slugify(inputs["title"]))
   ```
"""
