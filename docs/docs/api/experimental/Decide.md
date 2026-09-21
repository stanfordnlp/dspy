# dspy.experimental.Decide

!!! warning "Experimental API"
    `Decide`, `Noul`, `Score`, `Choice`, and `TypeSafe` are experimental and may
    change or be removed without warning. Import them from `dspy.experimental`.
    The planned cascade and numeric optimizer will also start as experimental
    APIs; neither is included in this release.

`Decide` answers closed-set questions through a System One model. Declare the
answer space through types; the module owns the numeric parameters used to
interpret the answers. It never falls back to a generative LM.

## Declare decisions once

```python
from typing import Annotated, Literal
import dspy
from dspy.experimental import Choice, Decide, Noul, Score, TypeSafe

Severity = Score[(0, "Minor"), (2, "Disruptive"), (10, "Blocking")]
Category = Choice[("billing", "Payment issue"), ("technical", "Product malfunction")]

class Assess(dspy.Signature):
    """Assess the ticket."""

    ticket: str = dspy.InputField()
    urgent: Noul = dspy.OutputField()
    severity: Severity = dspy.OutputField()
    category: Category = dspy.OutputField()

# pip install "dspy[typesafe]"
# Set TYPESAFE_API_KEY in the environment.
dspy.configure(system_one=TypeSafe("jev-latest"))
assess = Decide(Assess)
result = assess(ticket="Payment failed and checkout is unavailable.")
print(result.severity.value, result.severity.confidence)
```

The same signature works with `dspy.Predict(Assess)` when a generative LM is
configured. ChatAdapter and JSONAdapter describe the rubric and request an
object containing `value` and `confidence`. Confidence from an LLM is its
**self-reported confidence**, not a provider probability or a calibration guarantee.

The bracket syntax follows DSPy's configured-type convention (like `Code["python"]`);
it is a runtime API, not a standard generic accepted by every static type checker.

## Shorthand and rich forms

| Shorthand | Rich type | Runtime rich value |
| --- | --- | --- |
| `bool` | `Noul` | Boolean `.value`, `.confidence`, optional `.probability` |
| `Annotated[float, Severity]` | `Severity` | Numeric `.value`, `.confidence`, optional `.probabilities` |
| `Literal["billing", "technical"]` | `Category` | Selected `.value`, `.confidence`, optional `.probabilities` |

Shorthand returns native Python values without confidence. `Annotated[float, Severity]`
preserves the rubric while returning a float; bare `float` works with `Predict`
but is rejected by `Decide` because it has no rubric.

Score requires at least two finite, strictly increasing numeric anchors. Its
value can fall between anchors. Choice preserves string, integer, Boolean, and
None member types. Values whose string labels collide (such as `1` and `"1"`)
are rejected because the provider uses string labels.

### Inputs and outputs with either module

| Family / form | `Predict` input | `Predict` output | `Decide` input | `Decide` output |
| --- | --- | --- | --- | --- |
| `bool` | Native Boolean | Generated Boolean | Native Boolean in state | Thresholded Boolean |
| `Noul` | Value, confidence, available evidence | Generated value + confidence | Same structured value in state | Value + derived confidence + P(True) |
| Native float with rubric | Number; rubric in prompt | Generated number | Number; rubric in question context | Expected numeric value |
| `Score[...]` | Value, confidence, available evidence; rubric | Generated value + confidence | Same structured value; rubric in question context | Expected value + provider confidence + distribution |
| `Literal[...]` | Native member; allowed values in field description | Generated member | Native member in state | Selected native member |
| `Choice[...]` | Value, confidence, available evidence; option meanings | Generated value + confidence | Same structured value; option meanings in context | Member + provider confidence + distribution |

Both modules can consume previous results. Rich inputs preserve available evidence;
LLM-created values need no probability distribution. Missing evidence is omitted
from serialized inputs. Use `.value` when passing a rich result into a native field.
Input decisions are never re-thresholded or treated as extra output questions.

Generative output schemas require confidence in [0, 1], but do **not** ask for
provider probabilities. Provider evidence remains available in Python and when
passing rich results as inputs. JSON serialization preserves it.

## Local interpretation parameters

```python
assess.thresholds["urgent"] = 0.7
assess.weights["severity"] = [0, 4, 10]
assess.weights["category"] = {"billing": 0.7, "technical": 1.0}
```

Each Boolean output starts with threshold 0.5. Its value is `probability >= threshold`,
so equality returns True. Each Score output starts with its declared numeric anchors.
Weights must remain strictly increasing within the declared score range; this keeps
results compatible with the same type used by `Predict`.

Score computes `sum(p[i] * weights[i]) / sum(p)`. The denominator accounts for
provider distributions that sum approximately to one. Weights assign numeric
values to options; they do not multiply probabilities or introduce cut points.
Raw distributions are retained unchanged, with integer rubric indices as keys.

Choice weights instead multiply probabilities for local selection:
`argmax(probability[label] * weight[label])`. Each option starts at 1.0, and
omitted option weights default to 1.0. Keys are string labels, as in the provider
distribution: use `"1"`, `"True"`, and `"None"` for integer, Boolean, and None
members. Returned values still retain their declared Python types.

For example, probabilities `{"billing": 0.7, "technical": 0.3}` with weights
`{"billing": 0.25}` select `"technical"`: 0.175 is less than 0.3. The raw
probabilities remain unchanged and are not renormalized. All-unit weights retain
the provider's selection. Weighted ties prefer the provider's selection if tied,
otherwise declaration order. Multipliers must be finite and nonnegative; zero
disables an option. Unknown labels, all-zero effective weights, or a distribution
with no positive mass remaining after weighting raise `ValueError`.

Thresholds and weights belong to the module, separately for each output. Changing
them does not alter shared types, previous results, or the provider request, so
cached answers can be reused. No optimizer is included here.

## Confidence is source-dependent

- `Predict`: the LLM generates confidence alongside the rich value.
- `Decide`, Score and Choice: confidence comes directly from the provider and
  is not recomputed from the winning probability or numeric score.
  If Choice weights change the selected option, `.confidence` still describes
  the original provider decision, **not confidence in the weighted selection**.
- `Decide`, Noul: confidence is `abs(p - t) / max(t, 1 - t)`. It measures
  threshold-relative distance, not statistical calibration. At `t=0.75`, both
  `p=0.6` and `p=0.9` have confidence 0.2; the decision at `p=0.75` is True
  with confidence zero. Changing the threshold changes this confidence.

These sources are not guaranteed to be numerically interchangeable. `Decide`
does not route based on confidence; a separate cascade can define that policy.

## Client, composition, and persistence

`TypeSafe` uses the optional `typesafe-sdk` dependency, native sync/async calls,
DSPy's shared cache, bounded client history, and DSPy's usage tracking. Model and
endpoint default to `TYPESAFE_DEFAULT_MODEL` and `TYPESAFE_BASE_URL`, or `jev-latest`
and `https://api.typesafe.ai`. A client passed to `Decide(..., client=...)` takes
precedence over `dspy.settings.system_one`; no text LM is needed.

`Decide` participates in module composition, `named_predictors()`, callbacks,
traces, `batch`, and `acall`. It validates required inputs and rejects unknown
inputs and unsupported outputs. Signature task instructions and field descriptions
are included in each provider question. `Decide` does not support demonstrations:
passing `demos=` or attaching nonempty `.demos` raises before inference. Optimizers
that attach demonstrations to `Predict` are not supported for `Decide`.

A per-call `signature=` override may change instructions and field descriptions,
but must preserve output names, value types, and declared Choice options/Score
rubrics. Equivalent native/rich forms are allowed when they resolve to the same
decision definition. Existing thresholds and weights remain in effect. Construct
a new `Decide` to change the answer space; incompatible overrides fail before
any provider request.

```python
copy = assess.deepcopy()
assess.save("assess.json")
restored = Decide(Assess)
restored.load("assess.json")
```

State-only loading requires the same signature architecture, as with `Predict`.
Parameters survive this round trip; demonstrations are not saved, and loading
state with nonempty demonstrations raises rather than silently discarding them. Explicit TypeSafe
client settings are saved without API keys; credentials come from the environment
after loading. Saved endpoints require `allow_unsafe_lm_state=True` for trusted
files, following DSPy's LM-state policy. Custom callable clients should be supplied
through settings rather than serialized. Whole-program saving uses DSPy's existing
trusted-pickle workflow and must never be loaded from untrusted sources.

::: dspy.experimental.Decide
    options:
        members: [__init__, forward, aforward, dump_state, load_state]

::: dspy.experimental.Noul
    options:
        members: false

::: dspy.experimental.Score
    options:
        members: false

::: dspy.experimental.Choice
    options:
        members: false

::: dspy.experimental.TypeSafe
    options:
        members: [__init__, __call__, acall]
