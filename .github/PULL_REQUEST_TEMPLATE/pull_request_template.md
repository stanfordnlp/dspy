## What changed

<!-- Describe the behavior change and the code that implements it. Code samples
are welcome when they make the before/after behavior easier to see. -->

## Why

<!-- Link the issue or include a minimal reproduction. For fixes, explain the
root cause—not only the symptom—and any historical constraint that made this
boundary difficult to maintain. -->

## Why this fix

<!-- Explain why the change addresses the root cause and why it is the smallest
complete fix. Note alternatives considered when the choice is not obvious. -->

## Verification

<!-- List exact commands and results. For fixes, include failing evidence before
the change and passing evidence after it when practical. -->

## Review context

<!-- Call out context, tradeoffs, compatibility impact, or follow-up work a
reviewer needs to validate this change. -->

## Contributor checklist

- [ ] I ran the relevant tests and pre-commit checks.
- [ ] I added or updated tests, or explained why tests are not needed.
- [ ] I kept this PR focused and listed follow-up work separately.

### Dependency changes

<!-- Delete this section when the PR does not change a dependency contract. -->

- [ ] The full dependency graph resolves at the proposed minimum and at current/latest versions.
- [ ] A shipped runtime path fails below the proposed floor and passes at it.
- [ ] Published metadata and lock metadata agree, without unrelated lock updates.
