# dr-dspy

This package holds reusable helpers for DSPy experiments in this workspace.
Experiment logic should stay in single-script implementations under
`scripts/` whenever possible. A script should make the exact dataset, optimizer,
adapter, metric, run flow, and artifact choices easy to inspect in one place.

Library code under `src/dr_dspy/` is for behavior that is expected to remain
stable across experiments. Examples include event-log storage, DSPy callback
telemetry, and DSPy-aware serialization. Extracting those helpers keeps new
experiments small without hiding the decisions that define the experiment.

The split has two goals:

- Make each experiment traceable by keeping experiment-specific control flow in
  the script that runs it.
- Make shared infrastructure robust by centralizing the code that should not
  change when trying a new optimizer, adapter, model, dataset, or metric.

When adding a new experiment, default to a script first. Move code into
`src/dr_dspy/` only when it is likely to be reused unchanged by multiple
experiments and when centralizing it reduces the chance of setup bugs.
