PRIMITIVES_CATALOG: str = """\
You are authoring two code artifacts for a `dspy.Flex` module:

1) A `PREDICTORS` dict literal at module scope mapping attribute names to predictor
   instances. Each entry is a constructed `dspy.Predict` / `dspy.ChainOfThought` /
   `dspy.ReAct` over a sub-signature string.

2) A `def forward(self, **inputs):` function whose body orchestrates calls to those
   predictors and returns `dspy.Prediction(<output_fields...>)` matching the
   declared signature.

Available DSPy primitives:

- `dspy.Predict("a, b -> c, d")`, one LM call. `result = p(a=x, b=y)`; outputs are
   attributes: `result.c`, `result.d`. Signature strings use snake_case names.

- `dspy.ChainOfThought("a -> b")`, like Predict but adds an implicit `reasoning`
   output field; useful when the answer benefits from explicit step-by-step thought.

- `dspy.ReAct(signature_str, tools=[tool1, tool2])`, multi-step reasoning + tool
   use. `tools` is a list of plain Python callables or `dspy.Tool` instances.

- `dspy.Tool(func)`, wraps a callable as a tool (only used inside ReAct).

- `dspy.Prediction(field_a=value_a, field_b=value_b)`, construct the final return
   value. The keyword names MUST match the parent signature's output_fields exactly.

Rules you MUST follow:

- Reference every predictor as `self.<name>` (e.g. `self.classify`). The PREDICTORS
  entries will be attached to the `self` instance before `forward` is called.

- Do NOT access dunder attributes (anything matching `__*__`). Use the public API.

- Every `while` loop must be bounded, either with an explicit `break`, or rewrite
  to `for _ in range(N):`. Pick a small `N` (e.g. 5).

- Always return `dspy.Prediction(...)` from `forward`. Never raise from `forward`
  unless the input itself is malformed.

- Keep the design composable: prefer two short predictors with clear signatures
  over one all-knowing prompt.

- Do not hardcode example outputs or magic constants from the task description.
"""
