# Predict ergonomics on top of PR 9313

This note captures the integration branch built on top of upstream PR #9313
(`feat(signature): add type validation for input fields`).

It is intended as a contributor note for review, testing, and follow-up PR work.

## Branch and worktree

This work lives in a separate worktree so the main checkout stays untouched.

- Worktree: `../MaximeRivest-dspy-pr9313`
- Branch: `predict-ergonomics-on-pr9313`
- Base branch: `pr-9313-type-checking-signature`

## Why build on top of PR 9313?

PR 9313 makes DSPy input type hints matter at runtime by warning when a value is
incompatible with the declared input type.

This branch complements that work by making the same input information visible at
call time:

- input names
- input order
- input defaults
- input type hints

Together, the two changes make `dspy.Predict` feel more Pythonic and more
helpful in IDEs.

## Commit stack

The branch contains four commits on top of PR 9313:

1. `7e1d23c8` — `feat(predict): add signature_override for call-time signature swaps`
2. `97ece9f5` — `feat(predict): accept positional inputs in signature order`
3. `c52ebaca` — `feat(predict): expose typed call signatures for IDEs`
4. `2652c558` — `docs(predict): document positional calls and call-time overrides`

## What changed

### 1. `signature_override=` for one-call signature changes

`Predict` already supported a call-time `signature=` override internally, but the
name was overloaded:

- `Predict(signature=...)` at construction time
- `predict.signature` as stored predictor state
- `predict(..., signature=...)` as a one-call override

This branch adds the clearer call-time name:

```python
predict(question="...", signature_override=new_sig)
```

Legacy `signature=` is still accepted for compatibility.

If both are passed, `Predict` raises a clear error.

### 2. Positional inputs in signature order

Inputs can now be passed by position or by keyword.

For a signature like:

```python
"question, context -> answer"
```

these are all valid:

```python
predict("What is the capital of France?", "Use world knowledge.")
predict("What is the capital of France?", context="Use world knowledge.")
predict(question="What is the capital of France?", context="Use world knowledge.")
```

The implementation also adds clear errors for:

- too many positional arguments
- passing the same input both positionally and by keyword

### 3. Dynamic per-instance call signatures for IDEs

Each `Predict` instance now sets `self.__signature__` so tools such as
`inspect.signature()` can show the actual call shape for that predictor.

For a typed signature like:

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    context: list[str] = dspy.InputField()
    answer: str = dspy.OutputField()
```

`inspect.signature(predict)` can now show something close to:

```python
(question: str, context: list[str], *, config=None, signature_override=None, demos=None, lm=None)
```

### 4. Untyped inputs stay untyped

A key design choice in this branch is that untyped string-signature inputs stay
unannotated in the Python call signature.

For example:

```python
plain = dspy.Predict("question -> answer")
```

`inspect.signature(plain)` leaves `question` unannotated.

That matches the runtime intent: if the user did not provide a type hint, DSPy
should not imply one in IDE help.

## Important interaction with PR 9313

PR 9313 already distinguishes between:

- fields that are explicitly typed, and
- fields whose type is only the implicit default from a string signature.

That distinction is stored via `IS_TYPE_UNDEFINED`.

This branch respects the same signal in two places:

1. runtime warnings
   - typed field + incompatible value -> warning
   - untyped field -> no type mismatch warning
2. call signatures
   - typed field -> annotation shown
   - untyped field -> annotation omitted

This keeps the Python call surface aligned with the runtime behavior.

## One subtle edge case: defaulted inputs before required inputs

Python's `inspect.Signature` does not allow a positional-or-keyword parameter
with a default to come before a required positional-or-keyword parameter.

That can happen in DSPy signatures, for example:

```python
class Sig(dspy.Signature):
    context: str = dspy.InputField(default="DEFAULT_CONTEXT")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
```

Runtime DSPy behavior still supports positional calls based on input order.

For the displayed Python signature only, the builder uses this compromise:

- keep inputs positional-or-keyword until a required field appears after a
  defaulted one
- then switch the remaining inputs to keyword-only so the displayed signature is
  still valid Python

This affects IDE display only. Runtime binding still follows DSPy signature
order.

## Tests added or updated

The focused tests cover:

- `signature_override` works
- legacy `signature=` still works
- conflict between `signature_override=` and legacy `signature=` raises cleanly
- positional inputs work
- mixed positional + keyword input works
- too many positional inputs raise a clear error
- duplicate positional + keyword input raises a clear error
- `inspect.signature()` exposes typed inputs and reserved kwargs
- untyped string-signature inputs stay unannotated
- the dynamic call signature updates when `predict.signature` changes
- untyped fields do not trigger type mismatch warnings
- existing type mismatch warning behavior remains intact
- save/load and default-value behavior still work

## Focused test commands

From the integration worktree:

```bash
cd ../MaximeRivest-dspy-pr9313
```

### Ergonomics + type-aware call surface

```bash
pytest tests/predict/test_predict.py -q -k "signature_override_for_one_call or legacy_signature_kwarg_still_works_when_not_an_input_field or signature_override_conflicts_with_legacy_signature_kwarg or positional_arguments or mixed_positional_and_keyword_arguments or too_many_positional_arguments or duplicate_positional_and_keyword_arguments or call_signature_exposes_typed_inputs_and_reserved_kwargs or call_signature_leaves_untyped_string_signature_inputs_unannotated or call_signature_updates_when_predictor_signature_changes or untyped_string_signature_does_not_warn_on_type_mismatch or type_mismatch_warning or correct_types_no_warning or list_type_validation or literal_type_validation or predicted_outputs_piped_from_predict_to_lm_call or input_field_default_value or error_message_on_invalid_lm_setup"
```

### Save/load + signature updates

```bash
pytest tests/predict/test_predict.py -q -k "initialization_with_string_signature or lm_after_dump_and_load_state or instructions_after_dump_and_load_state or demos_after_dump_and_load_state or typed_demos_after_dump_and_load_state or signature_fields_after_dump_and_load_state or input_field_default_value or call_signature_updates_when_predictor_signature_changes"
```

## Manual experiments

### Inspect the call signature

```python
import inspect
import dspy

class QA(dspy.Signature):
    question: str = dspy.InputField()
    context: list[str] = dspy.InputField()
    answer: str = dspy.OutputField()

predict = dspy.Predict(QA)
print(inspect.signature(predict))
```

### Positional calls

```python
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
predict("What is the capital of France?", ["Use world knowledge."])
```

### Signature override

```python
concise = predict.signature.with_instructions("Answer in one short sentence.")
predict(
    "Why is the sky blue?",
    ["Use world knowledge."],
    signature_override=concise,
    config={"temperature": 0.2},
)
```

### Untyped signature stays untyped

```python
plain = dspy.Predict("question -> answer")
print(inspect.signature(plain))
```

## How to push the branch

```bash
git -C ../MaximeRivest-dspy-pr9313 push -u origin predict-ergonomics-on-pr9313
```

## PR strategy

This branch is based on PR 9313, not directly on `main`.

That makes it useful as an integration branch, but not yet the cleanest branch
for an upstream PR.

### Recommended upstream path after PR 9313 merges

```bash
git checkout main
git pull upstream main
git checkout -b predict-ergonomics-clean
git cherry-pick 7e1d23c8 97ece9f5 c52ebaca 2652c558
```

## Suggested PR framing

### One-line summary

Make `dspy.Predict` calls more Pythonic and more discoverable in IDEs.

### Why

DSPy signatures already know:

- input field names
- input field order
- input defaults
- input type hints

This work surfaces that information at the Python call boundary.

### What to highlight

- `signature_override=` is the clearer name for one-call signature changes
- positional inputs now work in DSPy signature order
- per-instance `__signature__` improves IDE help and `inspect.signature()`
- typed inputs show typed annotations
- untyped inputs remain untyped
- the work is scoped to `dspy.Predict`, not the whole `dspy.Module` ecosystem

### Why it pairs well with PR 9313

PR 9313 makes input type hints meaningful at runtime.
This branch makes those same type hints visible and pleasant to use in the call
API.

## Demo ideas

### Short video

A 1-minute demo could show:

1. `inspect.signature(predict)` before/after
2. positional calling
3. `signature_override=`
4. typed signature vs untyped signature
5. typed mismatch warning vs untyped no-warning

### Tiny script or notebook

A demo script can be organized into three sections:

1. ergonomic calling
   - positional
   - keyword
   - mixed
2. IDE/introspection
   - `inspect.signature(...)`
   - typed vs untyped
3. overrides
   - `config=...`
   - `lm=...`
   - `signature_override=...`

A simple script may be easier to share than a notebook, but either works.

## Current state

At the time this note was written:

- the integration branch is clean
- the focused predict tests listed above pass
- the original checkout remains untouched
