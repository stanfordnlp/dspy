# dr-dspy

This package holds reusable helpers and experiment scripts for DSPy work in
this workspace. The repo is intentionally split between readable experiment
entrypoints in `scripts/` and stable infrastructure in `src/dr_dspy/`.

## Experiments

### 1. HumanEval Bootstrap v0

Script:
[`scripts/humaneval_dspy_harness_bootstrap_v0.py`](scripts/humaneval_dspy_harness_bootstrap_v0.py)

This experiment runs a DSPy `BootstrapFewShot` pass over HumanEval Plus. It:

- builds a shuffled HumanEval train/dev split;
- asks an LM to emit a single Python function for each prompt;
- evaluates generated code in a subprocess sandbox;
- logs run, flow, module, adapter, LM, and metric events;
- saves the compiled DSPy program artifact to
  `logs/compiled_humaneval.json` by default.

The script keeps the experiment-defining choices local: dataset, signature,
metric, optimizer, model setup, run flow, and CLI flags. Shared mechanics are
imported from `src/dr_dspy/`.

For a deterministic smoke run, use the parallel mock script:
[`scripts/mocks/humaneval_dspy_harness_bootstrap_v0_mock.py`](scripts/mocks/humaneval_dspy_harness_bootstrap_v0_mock.py).
It imports the real `run_humaneval_bootstrap_flow` but supplies a tiny mock
dataset and `LoggingCallableLM`, so it exercises the same harness without
calling a real model.

## Repository Shape

`scripts/` contains experiment entrypoints. A script should make the exact
dataset, optimizer, adapter, metric, run flow, and artifact choices easy to
inspect in one place.

`scripts/mocks/` contains deterministic mock runners for experiments. Mock
scripts are allowed to import the real flow function from the experiment script,
but they own fake datasets, fake solvers, and smoke-test-specific setup.

`src/dr_dspy/` contains behavior expected to remain stable across experiments:

- `code_eval.py`: generated-code extraction and subprocess evaluation.
- `dspy_event_log.py`: DSPy callback telemetry.
- `dspy_programs.py`: reusable DSPy execution helpers.
- `event_log.py`: SQLite/Postgres event writers and writer construction.
- `flow.py`: flow context tracking for event logs.
- `lm_logging.py`: logging LM wrappers.
- `run_metadata.py`: run metadata capture and sanitization.
- `runtime.py`: shared script runtime setup.
- `serialization.py`: DSPy-aware JSON-safe serialization.

`tests/` covers both library behavior and the mock harness path. The tests are
not part of the Ruff/Ty target by default; they remain executable with pytest.

## Design Decisions

Default to a script first. Move code into `src/dr_dspy/` only when it is likely
to be reused unchanged by multiple experiments and centralizing it reduces setup
bugs.

Keep experiment-defining decisions in the script. The library should not hide
which dataset, signature, optimizer, metric, model, or artifact path makes an
experiment what it is.

Prefer clean boundaries over compatibility shims. This package is early enough
that breaking changes are acceptable when they make the structure clearer.

Use Postgres as the default event store. `DATABASE_URL` is the standard
configuration key, and scripts load the package-local `.env` file before writer
construction. SQLite remains available via `--event-store sqlite`.

Keep mock infrastructure parallel to, not inside, experiment scripts. The main
script should stay readable as the real experiment; the mock script should prove
that the same flow can run with prepared train/dev data and a prepared LM.

## Local Setup

Create a package-local `.env` from the example:

```sh
cp .env.example .env
```

The default local database URL is:

```sh
postgresql:///dr_dspy
```

Create the local database if needed:

```sh
createdb dr_dspy
```

Run package checks from `dr-dspy/`:

```sh
uv run ruff check src scripts
uv run ty check
uv run pytest tests
```

See [`TESTING.md`](TESTING.md) for the mock harness smoke command and success
criteria.
