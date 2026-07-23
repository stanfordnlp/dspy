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
  Give any predictor task-specific INSTRUCTIONS by constructing it over a
  `dspy.Signature("a, b -> c, d", "natural-language instructions ...")` instead of a bare
  string.
  You may optimize optimize these string instructions. See "Writing and refining instructions" 
  below.

- `dspy.ChainOfThought("a -> b")`, like Predict but adds an implicit `reasoning`
   output field; useful when the answer benefits from explicit step-by-step thought.

- `dspy.ReAct(signature_str, tools=[tool1, tool2])`, multi-step reasoning + tool
   use. `tools` is a list of plain Python callables or `dspy.Tool` instances.

- `dspy.RLM("context, query -> output")`, Recursive Language Model: the LLM writes
   Python code in a sandboxed REPL to programmatically explore very large or
   structured inputs (long documents, big JSON blobs, tables) and build up an answer
   iteratively. Use when one of the input fields is too large or too structured for a
   single direct prompt. The defaults are sensible; only pass a kwarg to override one.
   Optional kwargs: `max_iters` (REPL steps, default 20), `max_llm_calls` (sub-LLM
   calls, default 50), `sub_lm` (cheaper LM for sub-queries), `tools` (extra callables
   for the REPL).

- Tools for `dspy.ReAct` / `dspy.RLM` come from two places:
  (1) any tools listed in the available context, in scope by name — ONLY these provided tools may
  be handed to a sub-predictor via `dspy.ReAct(..., tools=[...])` / `dspy.RLM(..., tools=[...])`;
  and (2) helpers you AUTHOR yourself — when a sub-step needs a capability the provided tools don't
  cover (deterministic lookups, parsing/formatting, calculators, validators, retrieval helpers,
  etc.), define a plain function (with a docstring and type hints) nested inside `forward` and CALL
  IT DIRECTLY. This code runs in a sandbox, so an authored helper cannot be handed to a bridged
  sub-predictor — keep authored helpers to direct calls in `forward`.

- `dspy.Tool(func)`, wraps a callable as a tool (for `ReAct`/`RLM`); usually you can pass the
   bare function and it is wrapped for you.

- `dspy.Prediction(field_a=value_a, field_b=value_b)`, construct the final return
   value. The keyword names MUST match the parent signature's output_fields
   exactly, and each `value_*` must be a plain value of the declared output
   type — never a `dspy.Prediction` you got back from a predictor call.

Rules you MUST follow:

- Define every predictor in `__init__` as `self.<name> = dspy.Predict(...)` and call
  it in `forward` as `self.<name>(...)`.

- Give predictors INSTRUCTIONS, don't just rely on field names. When the feedback shows
  the model needs guidance, construct the predictor over
  `dspy.Signature("inputs -> outputs", "instructions")` and refine those instructions from the
  failing examples (see "Writing and refining instructions"). These instructions are optimized
  ONLY through this source, so improving them is as important as changing the code.

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

Writing and refining instructions:

The predictors you define are prompts, and a predictor's natural-language instructions — the
second argument to `dspy.Signature("inputs -> outputs", "instructions")` — are a PRIMARY thing
you optimize here, exactly like the code structure. Because these predictors live inside this
dspy.Flex module, this source is the ONLY place their instructions get optimized, so rewriting
them well matters as much as picking the right primitive. Don't leave a predictor on a bare
signature string once the feedback shows the model needs guidance.

When you revise, mine the failing examples and feedback the way a dedicated instruction
optimizer would:

- Read every failing input, output, and its feedback, and diagnose WHY each one failed.
- Categorize what the feedback is telling you:
    * Error patterns — recurring mistakes to explicitly prevent.
    * Success patterns — what worked and should be preserved/reinforced.
    * Domain-knowledge gaps — task-specific facts, definitions, or conventions the model
      lacked; state them IN the instruction, since the model won't otherwise have them.
    * Task-specific guidance — required output format, edge cases, and constraints.
- Fold those findings into each affected predictor's instructions: a clear task definition,
  the domain knowledge it needs, explicit rules that prevent the observed errors, the exact
  output format required, and precise, actionable language. Keep instructions specific to the
  task but NEVER paste in example answers or magic constants from the data.

Rule of thumb: fix a predictor's INSTRUCTIONS when a failure is about WHAT the model should do
or know; change the CODE structure (which predictors, how they're wired, what runs in plain
Python) when the failure is about HOW the steps fit together. The best revision often does both.

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

6) Put task guidance in a predictor's INSTRUCTIONS, not only in the field names, and refine
   them from feedback. Give the signature a second argument spelling out the rules the failures
   revealed (definitions, output format, error-prevention), without pasting in example answers.

   ```
   class ClassifyModule(dspy.Module):
       def __init__(self):
           super().__init__()
           self.classify = dspy.Predict(dspy.Signature(
               "ticket: str -> category: str",
               "Classify the support ticket into exactly one of: billing, bug, feature, other. "
               "Choose 'bug' only for a defect in existing behavior; a request for new behavior "
               "is 'feature'. Reply with the lowercase category word and nothing else."
           ))

       def forward(self, **inputs):
           out = self.classify(ticket=inputs["ticket"])
           return dspy.Prediction(category=out.category.strip().lower())
   ```
"""
