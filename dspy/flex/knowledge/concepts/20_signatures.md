## Writing sub-signatures for your predictors

A **signature** is a declarative spec of a predictor's input/output behavior. Field *names*
carry meaning — `question` differs from `answer`, `sql_query` from `python_code` — so name
fields by their semantic role. Each predictor you assign in `__init__` should get its own
tight sub-signature describing exactly the one job it does.

**Inline (string) signatures** are the common case. Names default to type `str`; add types
where they matter:

- `"sentence -> sentiment: bool"`
- `"context: list[str], question: str -> answer: str"`
- `"invoice: str -> qty: int, unit_price_cents: int, shipping_cents: int"`

Supported annotations: basic types (`str`, `int`, `float`, `bool`), typing generics
(`list[str]`, `dict[str, int]`, `Optional[float]`), and `Literal[...]` for a fixed choice set.
A typed output field tells the adapter to parse that type — but still coerce in Python before
returning from `forward` (LM string outputs do not always arrive as the declared type).

**Instructions** belong *inside* the signature, never as a second positional argument to
`dspy.Predict(...)`. When a step has task-specific rules (an exact allowed-label set, a domain
constraint, an output format), build a `dspy.Signature` with instructions and pass that:

```python
def __init__(self):
    super().__init__()
    self.classify = dspy.Predict(dspy.Signature(
        "text -> intent",
        "Classify the message into exactly one of: balance, transfer, dispute, other. "
        "Return only the label, lowercase, no punctuation.",
    ))
```

Carry over any important guidance from the task description (especially the exact allowed
labels or output format) into the relevant predictor's instructions so the model still sees it.
Use `Literal` in the field type when the choice set is small and fixed: `"text -> intent: Literal['balance', 'transfer', 'dispute', 'other']"`.

Don't prematurely hand-tune signature keywords — keep names semantically clear and let the
optimizer do the tuning. Just make sure the predictor has the information it needs to do its job.
