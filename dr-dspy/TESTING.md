# Testing

Run the package checks from `dr-dspy/`:

```sh
uv run ruff check src scripts
uv run ty check
uv run pytest tests
```

Run the mock HumanEval bootstrap smoke harness:

```sh
uv run python scripts/mocks/humaneval_dspy_harness_bootstrap_v0_mock.py \
  --db-path /tmp/dr-dspy-smoke.db \
  --compiled-path /tmp/dr-dspy-smoke.json
```

The smoke harness should exit with status `0`, write both files, and print
`baseline:  100.000` and `optimized: 100.000`.
