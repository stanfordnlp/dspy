# The dspy.vibe knowledge base

This folder is the curated knowledge that guides `dspy.vibe` **code synthesis**. The loader
(`__init__.py`) assembles it into one string, `build_knowledge_base()`, which
`dspy/vibe/primitives_doc.py` re-exports as `KNOWLEDGE_BASE`. That string is fed to the
codegen / reflection LM as `extra_guidance` whenever `dspy.GEPA` optimizes a `dspy.Vibe`
module's code. It is distilled from the DSPy docs & tutorials (`docs/docs/learn`,
`docs/docs/tutorials`, `docs/docs/api/modules`) and cross-checked against
[dspy.ai](https://dspy.ai).

## Layout

```
dspy/vibe/knowledge/
  __init__.py        # loader: build_knowledge_base() = concepts + rendered examples (pure text)
  MAINTAINING.md     # this file
  concepts/          # distilled prose, concatenated in filename order
    00_overview.md   #   what a good optimized vibe module looks like
    10_modules.md    #   DSPy modules to use inside forward (Predict/CoT/ReAct/RLM/...)
    20_signatures.md #   writing tight sub-signatures + instructions
    30_optimization.md #  how GEPA/vibe optimization + failure feedback work
    40_patterns.md   #   patterns to follow + anti-patterns to avoid
  examples/          # self-contained, validated reference modules (one task each)
    classify_intent.py   # fixed label set in instructions; normalize output in Python
    invoice_total.py     # LM extracts numbers, Python does the arithmetic
    iterative_refine.py  # bounded draft/critique/revise loop, controlled in Python
    long_report_qa.py    # keep RLM for a genuinely large field; route in Python
    math_word_problem.py # ChainOfThought reasons, Python coerces the numeric answer
    slugify.py           # fully deterministic — PREDICTORS = {}, no LM
```

The assembled order is: every `concepts/*.md` (sorted by filename) then a "Worked examples"
section rendering every `examples/*.py` (sorted by filename).

## How to add a concept

Drop a new `NN_topic.md` in `concepts/`. The numeric prefix sets its position (leave gaps —
`10`, `20`, … — so you can insert between later). No code change is needed. Keep it concise:
this text is injected into **every** reflection-LM prompt during optimization, so prefer
high-signal guidance over long doc dumps.

## How to add an example

Copy an existing file in `examples/` and keep the exact shape — it is parsed by `ast` + simple
text slicing, and validated by the test suite:

```python
"""<one-line title>."""

import dspy

TASK = "<one line: what the module does>"
SIGNATURE = "<the parent signature string, e.g. 'invoice: str -> total_cents: int'>"
NOTES = "<why this design is good — the lesson it teaches>"

# === PREDICTORS ===
PREDICTORS = { ... }          # exactly the module-scope predictors region


# === FORWARD ===
def forward(self, **inputs):  # exactly the forward region
    ...
    return dspy.Prediction(...)
```

Rules for an example file:

- `TASK`, `SIGNATURE`, `NOTES` must be plain string literals (adjacent-string concatenation is
  fine; no f-strings or computed values — they are read statically via `ast`).
- The two `# === PREDICTORS ===` / `# === FORWARD ===` marker comments must appear verbatim and
  in that order. The text between/after them becomes the `predictors_src` / `forward_src` that
  the test binds through the real `Vibe` code path, so each region must be self-contained: it
  may reference only `dspy`, the declared inputs, and names it defines itself (e.g. a nested
  helper or an in-`forward` `import re`). No module-scope helpers in the regions.
- Construction must be LM-free (just `dspy.Predict(...)` / `dspy.RLM(...)` instances) — the
  validation test binds the example but never calls an LM.
- Prefer examples that teach a *distinct* lesson (decomposition, Python-vs-LM split, output
  coercion, when RLM is right, bounded loops, fully-deterministic). Don't duplicate a lesson.

## Tests

`tests/vibe/test_knowledge_base.py` validates that the base assembles, that every concept and
example is included, and that **every example binds** via `Vibe._bind_code` — so an example that
drifts out of sync with the live API fails CI rather than silently rotting in the prompt.
