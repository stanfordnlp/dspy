## DSPy modules you can use inside `forward`

A DSPy module abstracts a prompting technique and works over any signature. Pick the
*simplest* module that fits each step; reach for heavier ones only when the step needs them.

- **`dspy.Predict("a, b -> c, d")`** — one LM call, no extra scaffolding. `r = self.p(a=x, b=y)`
  returns a `dspy.Prediction`; the declared outputs are attributes (`r.c`, `r.d`). The default
  for direct, single-call transformations: extraction, classification, short generation.

- **`dspy.ChainOfThought("a -> b")`** — like `Predict` but injects an implicit `reasoning`
  output before the declared fields, teaching the LM to think step by step. Swapping it in for
  `Predict` often improves quality when the answer benefits from explicit reasoning (judgments,
  multi-step deductions). Read your declared field (`r.b`); `r.reasoning` is auxiliary.

- **`dspy.ReAct(signature, tools=[...])`** — an agent that interleaves reasoning with tool
  calls to implement the signature. `tools` is a list of plain callables or `dspy.Tool`
  instances. Use ONLY when the step must call external tools mid-reasoning. The tools must be
  available by name in the module's context.

- **`dspy.ProgramOfThought("a -> b")`** — the LM writes code whose execution dictates the
  answer. Niche inside vibe: usually prefer to let the LM extract structured values and do the
  computation yourself in `forward` (more reliable, easier to debug).

- **`dspy.RLM("context, query -> answer")`** — Recursive Language Model: the LM explores a
  large/structured input through a sandboxed Python REPL with recursive sub-LLM calls, loading
  only the context it needs. Use ONLY when an input field is too large or too structured to fit
  a single prompt (long documents, big JSON, multi-table data). Useful kwargs: `max_iterations`
  (REPL steps, default 20), `max_llm_calls` (default 50), `sub_lm` (a cheaper LM for sub-queries).
  RLM is the vibe *baseline*; keeping it is right only when the input genuinely needs iterative
  exploration — otherwise decompose it into direct predictors + Python.

- **`dspy.Tool(func)`** — wraps a callable as a tool (only meaningful inside `ReAct` / as an
  RLM tool).

### Composing modules

`forward` is ordinary Python: call modules in any control flow you like — sequentially, in
loops (always bounded), or conditionally — passing each predictor's *unwrapped* output field
into the next call. Prefer two short, clearly-named predictors over one all-knowing prompt.
A canonical multi-step shape (adapted from the multi-hop tutorial):

```python
def forward(self, **inputs):
    notes = []
    for _ in range(4):  # bounded loop
        q = self.generate_query(claim=inputs["claim"], notes=notes).query
        ctx = self.lookup(query=q).context           # each output unwrapped before reuse
        notes.extend(self.append_notes(claim=inputs["claim"], notes=notes, context=ctx).new_notes)
    return dspy.Prediction(notes=notes)
```
