# Testing

Run the package checks from `dr-dspy/`:

```sh
uv run ruff check src scripts
uv run ty check
uv run pytest tests
```

Run the mock HumanEval bootstrap smoke harness:

```sh
direnv exec .. uv run python \
  scripts/mocks/humaneval_dspy_harness_bootstrap_v0_mock.py \
  --compiled-path /tmp/dr-dspy-smoke.json
```

The smoke harness defaults to Postgres and reads `DATABASE_URL`; the repo-local
`.envrc` sets it to `postgresql:///dr_dspy` when it is unset. The smoke harness
should exit with status `0`, write the compiled JSON file, write events to
Postgres, and print `baseline:  100.000` and `optimized: 100.000`.
