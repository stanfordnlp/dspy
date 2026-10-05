# Decision-Making with Jev Types

!!! warning "Experimental API"
    `Noul`, `Score`, `Choice`, `TypeSafe`, and `ReAnchor` are experimental and may change
    without warning. See the [API reference](../../api/experimental/DecisionTypes.md) for details.

In this tutorial, we will walk you through building **structured decision-making programs** in DSPy
using the three experimental decision types — `Noul`, `Score`, and `Choice` — and calibrating them
automatically with the `ReAnchor` optimizer.

Decision types let you move beyond free-text LLM outputs. Instead of asking an LLM "Is this urgent?"
and parsing "Yes" from a string, you get a **probability-backed boolean** with a tunable threshold.
Instead of asking "Rate severity from 1 to 5", you get a **continuous score** with level boundaries
you can adjust. This is the core idea behind DSPy's Jev integration.

## Prerequisites

```shell
pip install dspy
```

If you want to use the TypeSafe System One backend (optional):

```shell
pip install "dspy[typesafe]"
```

## Define Decision Types

Decision types describe the **possible outcomes** of a structured judgment. Each type uses bracket
syntax to declare its options:

```python
import dspy
import os
from dspy.experimental import Noul, Score, Choice

os.environ["OPENAI_API_KEY"] = "{your_openai_api_key}"

# A boolean decision: "Is this urgent?"
Urgent = Noul[
    (True,  "The customer cannot use the product at all"),
    (False, "The customer has a workaround or minor inconvenience"),
]

# A scored rubric: rate from 0 (low) to 2 (high), labels ordered lowest to highest
Severity = Score["Cosmetic issue", "Degrades workflow", "Blocks critical path"]

# A categorical choice: each tuple is (value, description)
Category = Choice[
    ("billing",   "Payment, invoice, or subscription problem"),
    ("technical", "Bug, crash, or performance issue"),
    ("account",   "Login, permissions, or profile issue"),
]
```

**Key points:**

- `Noul` is a boolean with probability. `if result.urgent:` works — it delegates to `.value`.
- `Score` produces a continuous float in `[0, N-1]` plus a discrete `.level`. For 3 labels, the value range is `[0.0, 2.0]`.
- `Choice` selects from a fixed set. The value preserves its Python type (`str`, `int`, `bool`, or `None`).

## Build a Signature

A DSPy signature defines your program's inputs and outputs. Decision types go in output fields just
like any other type:

```python
class TriageTicket(dspy.Signature):
    """Assess a customer support ticket for routing and prioritization."""

    ticket: str = dspy.InputField(desc="The customer's support message.")
    urgent: Urgent = dspy.OutputField(desc="Is the customer completely blocked?")
    severity: Severity = dspy.OutputField(desc="How severely is the customer affected?")
    category: Category = dspy.OutputField(desc="What kind of issue is this?")
```

!!! note "Descriptions are required"
    Every decision output field must have a `desc=` or an explicit `instructions` entry in
    `predict.fields`. Without one, DSPy raises an error before calling the LM.

## Run with a Generative LM

You don't need a TypeSafe API key to use decision types. Any DSPy-supported LM works:

```python
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

triage = dspy.Predict(TriageTicket)
result = triage(ticket="I can't log in to my account. The password reset page crashes.")

print(f"Urgent:   {result.urgent}")           # Noul object (bool-like)
print(f"  value:  {result.urgent.value}")      # True or False
print(f"  prob:   {result.urgent.probability}") # P(True), e.g. 0.72

print(f"Severity: {result.severity}")
print(f"  value:  {result.severity.value}")    # e.g. 1.4 (continuous)
print(f"  level:  {result.severity.level}")    # e.g. 1 (discrete)

print(f"Category: {result.category}")
print(f"  value:  {result.category.value}")    # e.g. "account"
```

Behind the scenes, DSPy asks the LM to produce probability evidence (not just a label) and derives
the final value locally using thresholds, cuts, and weights.

## Tune Decision Boundaries

The default threshold for `Noul` is 0.5 — if P(True) ≥ 0.5, the result is `True`. But what if your
use case needs to be more conservative (e.g., only flag as urgent when the LM is very confident)?

You can set parameters directly:

```python
triage.fields["urgent"] = {"threshold": 0.7}
triage.fields["severity"] = {"cuts": [0.8, 1.5]}
triage.fields["category"] = {"weights": {"billing": 2.0, "technical": 1.0, "account": 1.0}}
```

| Type | Parameter | Effect |
|------|-----------|--------|
| `Noul` | `threshold` | P(True) must be ≥ this value to return `True`. Range: [0, 1]. |
| `Score` | `cuts` | N−1 boundaries that divide the continuous value into discrete levels. |
| `Choice` | `weights` | Multipliers on each option's probability. Higher = preferred in close calls. |

## Automatic Calibration with ReAnchor

Manually tuning thresholds is tedious. `ReAnchor` does it automatically by fitting parameters to a
labeled training set:

```python
from dspy.experimental import ReAnchor

trainset = [
    dspy.Example(
        ticket="Payment failed, order stuck",
        urgent=True, severity=2, category="billing",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="Typo in the help page",
        urgent=False, severity=0, category="technical",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="Can't reset my password, locked out completely",
        urgent=True, severity=2, category="account",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="Invoice date is wrong but amount is correct",
        urgent=False, severity=0, category="billing",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="App freezes when I open settings",
        urgent=False, severity=1, category="technical",
    ).with_inputs("ticket"),
    # ... more examples for reliable calibration
]

def metric(example, prediction, trace=None):
    score = 0.0
    if prediction.urgent.value == example.urgent:
        score += 1.0
    if prediction.category.value == example.category:
        score += 1.0
    if prediction.severity.level == example.severity:
        score += 1.0
    return score / 3.0

optimizer = ReAnchor(metric)
calibrated_triage = optimizer.compile(triage, trainset=trainset)

# The original `triage` is unchanged. `calibrated_triage` has fitted parameters.
print(calibrated_triage.fields)
# e.g. {'urgent': {'threshold': 0.62}, 'severity': {'cuts': [0.45, 1.55]}, ...}
# If defaults already score well, fields may stay empty — that's normal.
```

**How ReAnchor works:**

1. Runs your program on the training set and collects probability evidence from each call
2. For each decision output, tries candidate parameter values (threshold midpoints, cut positions, weight multipliers)
3. Cross-validates with 5-fold checks to avoid overfitting to a single example
4. Returns a copy of your program with the best-scoring parameters

!!! tip
    ReAnchor works with cached LM calls. Run your training set once, then iterate on calibration
    without re-calling the LM. Set `require_cache=False` if you want to allow uncached calls.

## Save and Load

Calibrated programs save and load like any DSPy program, including the fitted decision parameters:

```python
calibrated_triage.save("triage_calibrated.json")

# Later:
loaded_triage = dspy.Predict(TriageTicket)
loaded_triage.load("triage_calibrated.json")
print(loaded_triage.fields)  # Fitted parameters restored
```

## Use the TypeSafe Backend (Optional)

If you have a TypeSafe API key, you can switch the same predictor to the System One backend for
probability-native inference:

```python
from dspy.experimental import TypeSafe

os.environ["TYPESAFE_API_KEY"] = "your-api-key"

jev_client = TypeSafe("jev-latest")
triage.set_lm(jev_client)

result = triage(ticket="My dashboard is blank after the update")
```

The same signature, demos, and field parameters work with both backends. The difference is in how
probabilities are produced:

- **Generative LMs**: DSPy asks the LM to generate probability evidence as structured JSON, then decodes it locally.
- **TypeSafe System One**: The API returns calibrated probabilities directly. No generation settings (temperature, etc.) are supported.

You can even mix backends within a single program — use Jev for critical decisions and a generative
LM for free-text outputs.

## Native Annotations

If you want the Python result to be a plain `bool` or `Literal` member (not the rich `Noul`/`Choice`
object), use `Annotated`:

```python
from typing import Annotated, Literal

class SimpleTriageTicket(dspy.Signature):
    """Quick triage returning native Python types."""

    ticket: str = dspy.InputField(desc="Customer message.")
    urgent: Annotated[bool, Urgent] = dspy.OutputField(desc="Is the customer blocked?")
    category: Annotated[
        Literal["billing", "technical", "account"], Category
    ] = dspy.OutputField(desc="Issue type.")
```

With `Annotated[bool, Urgent]`, the result is a plain `bool` — you still get probability-backed
decisions and threshold tuning, but `result.urgent` is `True` or `False` directly.

## Rich Criteria

For complex decisions, you can attach structured criteria that guide the LM's judgment:

```python
triage.set_criteria("urgent", {
    "true": {
        "definition": "Customer is completely unable to use the product",
        "examples": ["App crashes on launch", "Login page returns 500"],
        "not": "Intermittent issues with workarounds",
    },
    "false": {
        "definition": "Customer can still use the product, possibly with degradation",
        "examples": ["Slow page load", "Minor UI glitch"],
    },
})
```

Criteria must be valid JSON (strings, dicts, lists, or null). For `Noul`, the keys are `"true"` and
`"false"`. For `Score`, use a list matching the number of levels. For `Choice`, use a dict mapping
option values to their criteria.

## Putting It All Together

Here's a complete end-to-end example:

```python
import dspy
import os
from dspy.experimental import Noul, Score, Choice, ReAnchor

os.environ["OPENAI_API_KEY"] = "{your_openai_api_key}"

# 1. Define types
Urgent = Noul[(True, "Completely blocked"), (False, "Has workaround")]
Severity = Score["Minor", "Moderate", "Critical"]
Category = Choice[("billing", "Payment"), ("technical", "Product bug"), ("account", "Access")]

# 2. Define signature
class Triage(dspy.Signature):
    """Route and prioritize support tickets."""
    ticket: str = dspy.InputField(desc="Customer message.")
    urgent: Urgent = dspy.OutputField(desc="Is the customer blocked?")
    severity: Severity = dspy.OutputField(desc="Impact level.")
    category: Category = dspy.OutputField(desc="Issue type.")

# 3. Configure and run
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
triage = dspy.Predict(Triage)

# 4. Calibrate
trainset = [
    dspy.Example(ticket="Can't check out", urgent=True, severity=2, category="technical").with_inputs("ticket"),
    dspy.Example(ticket="Logo looks blurry", urgent=False, severity=0, category="technical").with_inputs("ticket"),
    dspy.Example(ticket="Locked out of account", urgent=True, severity=2, category="account").with_inputs("ticket"),
    dspy.Example(ticket="Receipt has wrong date", urgent=False, severity=0, category="billing").with_inputs("ticket"),
]

def metric(example, pred, trace=None):
    s = float(pred.urgent.value == example.urgent) + float(pred.category.value == example.category)
    s += float(pred.severity.level == example.severity)
    return s / 3

calibrated = ReAnchor(metric).compile(triage, trainset=trainset)

# 5. Use the calibrated program
result = calibrated(ticket="Payment page crashes when I click submit")
print(f"Urgent: {result.urgent.value} (p={result.urgent.probability:.2f})")
print(f"Severity: {result.severity.level} — {result.severity.value:.2f}")
print(f"Category: {result.category.value}")

# 6. Save for production
calibrated.save("triage_calibrated.json")
```

## FAQ

**Q: Do I need a TypeSafe API key?**
No. Decision types work with any generative LM that DSPy supports. The TypeSafe backend is an
optional, probability-native alternative.

**Q: Can I mix decision outputs with regular string outputs?**
Yes. You can have `summary: str` alongside `urgent: Noul[...]` in the same signature. The generative
LM handles both; TypeSafe requires all outputs to be decision types.

**Q: What's the difference between `Noul` and just returning a `bool`?**
A bare `bool` output uses direct generation — the LM picks True or False. A `Noul` output asks
the LM for a probability and applies a threshold locally. This gives you confidence scores and
tunable boundaries. You can opt a bare `bool` into evidence decoding by adding it to
`predict.fields`.

**Q: How many training examples does ReAnchor need?**
ReAnchor requires a nonempty trainset and uses up to 5-fold cross-validation (fewer folds for
smaller sets). More data gives more reliable calibration. Since it reuses cached LM calls,
adding more examples is cheap after the first pass.

**Q: Can I use this with async?**
Yes. `TypeSafe` supports `acall()` for async inference. Decision types work with DSPy's async
support.
