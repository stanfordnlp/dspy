# DSPy + HumanEval+ Harness — Final Plan

A single-file Python harness that runs `Evaluate` → `BootstrapFewShot` → `Evaluate` on the HumanEval+ dataset using DSPy, logs every observable boundary to SQLite, and ships with a mock-LM test mode that verifies the harness behaves correctly without touching a real LLM.

---

## Goals

1. **End-to-end eval & optimize flow** on HumanEval+ with `dspy.Predict("prompt -> code: dspy.Code")`.
2. **Comprehensive logging** of every DSPy boundary (modules, LM calls, adapter format/parse, tools, evaluate, metric) plus the *true* post-preprocess LM payloads, to a single SQLite DB.
3. **Sandboxed scoring** via `multiprocessing` subprocess running the HumanEval+ tests, with timeouts and resource limits.
4. **Async throughout**, using `dspy.asyncify` on the student so `Evaluate` drives the program via its thread pool.
5. **Self-tests** runnable with `--test` that use a `CallableLM` mock to deterministically prove:
   - The metric scores correctly for pass / partial / parse-fail / timeout / crash.
   - `BootstrapFewShot` selects exactly the demos whose mocked outputs pass the metric.
   - A full mocked harness run produces the expected SQLite rows, score deltas, and saved artifacts.

---

## Locked decisions

| Decision | Value |
|---|---|
| Model (real run) | `openai/gpt-4o-mini` |
| Dataset split | shuffle with `seed=0`; first 32 → train, next 32 → dev |
| `num_threads` for Evaluate | 8 |
| Subprocess timeout | 15 s |
| `RLIMIT_AS` (child) | 1 GiB |
| `RLIMIT_CPU` (child) | 17 s |
| `BootstrapFewShot` knobs | `max_bootstrapped_demos=4, max_labeled_demos=0, teacher=None` |
| Bootstrap threshold | full pass (`score >= 1.0`); document the soft-fallback path but don't auto-trigger |
| Zero-demo behavior | log loudly, exit nonzero |
| Async strategy | `dspy.asyncify(student)` so Evaluate's thread pool drives it |
| Output paths | `./runs.db` (shared across runs), `./compiled_humaneval.json` |
| `dspy.Code` extraction | helper with three fallbacks + logged warning if it falls through |
| Cost cap | none in v1 |
| Progress UI | tqdm + display_table=10 |
| DSPy stderr logger level | `WARNING` |
| SQLite scope | shared file, runs distinguished by `run_id` |
| Subprocess crash policy | score = 0.0, log error, continue |
| Code quality | typed; `ruff check`, `ruff format`, `ty check` must pass |

---

## File layout (single file: `humaneval_dspy_harness.py`)

```
humaneval_dspy_harness.py
├── imports & constants
├── to_jsonable / _sanitize           (serialization helpers)
├── SQLiteWriter                       (queue + writer-thread)
├── SQLiteCallback(BaseCallback)       (all DSPy hooks → writer.put_event)
├── LoggingLM(dspy.LM)                 (real LM wrapper)
├── CallableLM(dspy.BaseLM)            (mock LM)
├── LoggingCallableLM                  (CallableLM + same logging hooks)
├── _run_in_subprocess / humaneval_metric
├── Solve signature                    (prompt -> code: dspy.Code)
├── build_dataset()
├── run_flow_baseline / run_flow_optimize / run_flow_eval_optimized
├── main_real(args) / main_mock(args)
├── --test self-test cases             (pytest-free, pure asserts)
└── __main__ guard with argparse
```

CLI:
- `python humaneval_dspy_harness.py` — real run.
- `python humaneval_dspy_harness.py --mock` — full flow with `CallableLM` instead of OpenAI.
- `python humaneval_dspy_harness.py --test` — runs the deterministic self-tests, prints PASS/FAIL per test, exits nonzero on any failure.
- `python humaneval_dspy_harness.py --dev-size N --train-size N --num-threads N --timeout S` — overrides.

---

## SQLite schema

One table, minimal real columns + JSON payload. PRAGMAs at connect: `journal_mode=WAL`, `synchronous=NORMAL`, `temp_store=MEMORY`.

```sql
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL    NOT NULL,
    run_id          TEXT    NOT NULL,
    flow            TEXT    NOT NULL,
    event_type      TEXT    NOT NULL,
    call_id         TEXT,
    parent_call_id  TEXT,
    example_id      TEXT,
    score           REAL,
    error           TEXT,
    payload         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run   ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_call  ON events(call_id);
CREATE INDEX IF NOT EXISTS idx_events_type  ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_ex    ON events(example_id);
```

Event types written:

| event_type | source |
|---|---|
| `run.start` / `run.end` | `main_*` entry/exit, includes versions, args, seed |
| `flow.start` / `flow.end` | each of the three phases |
| `module.start` / `module.end` | `BaseCallback.on_module_*` |
| `lm.start` / `lm.end` | `BaseCallback.on_lm_*` (parsed-output view) |
| `lm.request` / `lm.response` / `lm.error` | `LoggingLM.forward` / `aforward` (true payload view) |
| `adapter.format.start` / `adapter.format.end` | `BaseCallback.on_adapter_format_*` |
| `adapter.parse.start` / `adapter.parse.end` | `BaseCallback.on_adapter_parse_*` |
| `tool.start` / `tool.end` | `BaseCallback.on_tool_*` |
| `evaluate.start` / `evaluate.end` | `BaseCallback.on_evaluate_*` |
| `metric.score` | `humaneval_metric` directly |
| `bootstrap.no_demos` | post-compile check |

`flow` is held in a `contextvars.ContextVar[str]` and switched between phases. `call_id` chaining uses a second `ContextVar[str | None]` for `parent_call_id`.

---

## Component contracts

### `to_jsonable(x: Any, *, max_len: int = 256 * 1024) -> Any`

Pure, never raises, never blocks. Handling order:

1. `None | bool | int | float | str` → as-is.
2. `list | tuple` → recurse.
3. `dict` → recurse on values, `str(k)` on non-str keys.
4. `dspy.Example | dspy.Prediction` → `.toDict()` then recurse.
5. `Signature` class (`isinstance(x, type) and issubclass(x, dspy.Signature)`) → `{"signature": x.signature, "instructions": x.instructions, "fields": [...]}`.
6. `dspy.BaseLM` → `{"model": x.model, "kwargs": _sanitize(getattr(x, "kwargs", {}))}`.
7. `pydantic.BaseModel` → `x.model_dump(mode="json")`.
8. coroutine / async-iterator / generator → `f"<{type(x).__name__}>"`. Never await.
9. Fallback → `repr(x)[:4096]`.

After serialization, if `json.dumps(result)` exceeds `max_len`, wrap as `{"_truncated": True, "preview": result_str[:max_len]}`.

`_sanitize(kwargs)` strips: `api_key`, `api_base`, `base_url`, `model_list`, `authorization`. Mirrors DSPy's own `_sanitize_lm_state`.

### `SQLiteWriter`

```python
class SQLiteWriter:
    def __init__(self, path: str, run_id: str) -> None: ...
    def put_event(
        self,
        event_type: str,
        payload: Any,
        *,
        flow: str | None = None,            # defaults to current flow ContextVar
        call_id: str | None = None,
        parent_call_id: str | None = None,
        example_id: str | None = None,
        score: float | None = None,
        error: str | None = None,
    ) -> None: ...
    def close(self) -> None: ...
```

Internals:
- Daemon thread with a `queue.Queue`, sentinel for shutdown.
- Batches up to 256 records per transaction.
- Writer thread owns the `sqlite3.Connection`.
- `put_event` is non-blocking, swallows exceptions, prints to stderr on failure.
- `close()` enqueues sentinel and `thread.join(timeout=10)`.

### `SQLiteCallback(BaseCallback)`

Implements every hook in `dspy.utils.callback.BaseCallback`. Each hook body:

```python
def on_module_start(self, call_id, instance, inputs):
    try:
        parent = _current_call_id.get()
        token = _current_call_id.set(call_id)
        self._tokens[call_id] = token
        self._writer.put_event(
            "module.start",
            payload={"instance": to_jsonable(instance), "inputs": to_jsonable(inputs)},
            call_id=call_id, parent_call_id=parent,
        )
    except Exception as e:
        print(f"[SQLiteCallback] {e!r}", file=sys.stderr)
```

`*_end` hooks pop the ContextVar token. Never raises. Captures `exception` field when present.

### `LoggingLM(dspy.LM)`

```python
class LoggingLM(dspy.LM):
    def __init__(self, model: str, *, log: PutEventFn, **kwargs: Any) -> None: ...
    def forward(self, prompt=None, messages=None, **kwargs): ...
    async def aforward(self, prompt=None, messages=None, **kwargs): ...
```

Both `forward` and `aforward`:
1. Generate `req_id = uuid.uuid4().hex`.
2. `log("lm.request", {"req_id": req_id, "messages": messages, "kwargs": _sanitize(kwargs)})`.
3. Try `super().{forward,aforward}(...)`.
4. On success: `log("lm.response", {"req_id": req_id, "dt": elapsed, "response": to_jsonable(resp)})`.
5. On exception: `log("lm.error", {"req_id": req_id, "dt": elapsed, "error": repr(e)})`, re-raise.

This is the only place that sees the post-`_call_preprocess` payload and the raw provider response.

### `CallableLM(dspy.BaseLM)`

```python
class CallableLM(dspy.BaseLM):
    forward_contract = "legacy"

    def __init__(
        self,
        fn: Callable[[list[dict], dict], dict[str, str]],
        *,
        model: str = "callable/test",
    ) -> None: ...

    def forward(self, prompt=None, messages=None, **kwargs): ...
    async def aforward(self, prompt=None, messages=None, **kwargs): ...
```

- `fn(messages, kwargs) -> {"field_name": "value", ...}`.
- Wraps the dict into the OpenAI-style provider response shape that `ChatAdapter` expects (mirroring `DummyLM` from `dspy.utils.dummies`).
- Records every call to `self.calls: list[dict]` for test assertions.
- `aforward` just calls `forward` (no real async work needed).

### `LoggingCallableLM(LoggingLM, CallableLM)`

Same logging surface as `LoggingLM` but mock backend. Used by `--mock` and the self-tests.

### `_run_in_subprocess`

```python
def _run_in_subprocess(
    *,
    code: str,
    test: str,
    entry_point: str,
    timeout: float,
    mem_limit_bytes: int = 1 << 30,
    cpu_limit_seconds: int = 17,
) -> tuple[float, str | None]: ...
```

Returns `(score, error_message_or_None)`. Score is `1.0` if `check(candidate)` returns normally, `0.0` otherwise.

Child function (`_worker`, must be top-level for `spawn` compatibility):

```python
def _worker(code, test, entry_point, conn):
    import resource, signal
    try:
        resource.setrlimit(resource.RLIMIT_AS,  (MEM, MEM))
        resource.setrlimit(resource.RLIMIT_CPU, (CPU, CPU))
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(int(CPU))
        ns: dict = {}
        exec(code, ns)
        exec(test, ns)
        ns["check"](ns[entry_point])
        conn.send(("ok", None))
    except BaseException as e:
        conn.send(("err", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()
```

Parent:
- `mp.get_context("fork")` on POSIX, `"spawn"` on Windows (auto-detected).
- `Process.start()`, `parent.poll(timeout)`, `recv()`.
- On timeout: `terminate()`, then `kill()` if needed; return `(0.0, "timeout")`.

### `humaneval_metric(example, pred, trace=None) -> float | bool`

```python
def humaneval_metric(example, pred, trace=None):
    code = _extract_source(pred)         # tries pred.code (str), pred.code.code, str(pred.code)
    score, err = _run_in_subprocess(
        code=code,
        test=example.test,
        entry_point=example.entry_point,
        timeout=SUBPROCESS_TIMEOUT,
    )
    _writer.put_event(
        "metric.score",
        payload={"task_id": example.task_id, "code": code, "error": err},
        example_id=example.task_id,
        score=score,
        error=err,
    )
    if trace is None:
        return score        # eval mode
    return score >= 1.0     # bootstrap mode (full-pass threshold)
```

`_writer` is module-level, set by `main_*` at boot. The `trace is not None` switch is the only place to relax the threshold if `BootstrapFewShot` finds zero demos.

### Signature

```python
class Solve(dspy.Signature):
    """Write a self-contained Python function that satisfies the prompt. Include any imports inside the function or at the top. Do not include tests or example calls."""
    prompt: str = dspy.InputField()
    code: dspy.Code = dspy.OutputField()
```

### `build_dataset(seed: int = 0, train_size: int = 32, dev_size: int = 32) -> tuple[list[Example], list[Example]]`

- `datasets.load_dataset("evalplus/humanevalplus", split="test")`.
- `random.Random(seed).shuffle(indices)`.
- Map each row to `dspy.Example(prompt=row["prompt"], test=row["test"], entry_point=row["entry_point"], task_id=row["task_id"]).with_inputs("prompt")`.
- Slice → `(trainset, devset)`.

---

## Run orchestration

```python
async def main_real(args):
    run_id = uuid.uuid4().hex
    writer = SQLiteWriter(args.db_path, run_id=run_id)
    _bind_writer(writer)  # module-level _writer used by humaneval_metric
    try:
        cb = SQLiteCallback(writer)
        lm = LoggingLM(args.model, log=writer.put_event, cache=False)
        dspy.configure(lm=lm, callbacks=[cb], track_usage=True, max_trace_size=10_000)

        writer.put_event("run.start", payload=_run_metadata(args))

        trainset, devset = build_dataset(args.seed, args.train_size, args.dev_size)
        student = dspy.Predict(Solve)
        async_student = dspy.asyncify(student)
        evaluator = dspy.Evaluate(
            devset=devset, metric=humaneval_metric,
            num_threads=args.num_threads, display_progress=True, display_table=10,
        )

        with _flow_context("eval_baseline"):
            baseline = evaluator(async_student)

        with _flow_context("optimize"):
            opt = BootstrapFewShot(metric=humaneval_metric,
                                   max_bootstrapped_demos=4, max_labeled_demos=0)
            compiled = opt.compile(student=dspy.Predict(Solve), trainset=trainset)
            _check_demos(compiled, writer)   # exits nonzero on zero demos

        with _flow_context("eval_optimized"):
            optimized = evaluator(dspy.asyncify(compiled))

        compiled.save(args.compiled_path)
        writer.put_event("run.end", payload={"baseline": baseline.score,
                                             "optimized": optimized.score})
        print(f"baseline:  {baseline.score:.3f}")
        print(f"optimized: {optimized.score:.3f}")
    finally:
        writer.close()
```

`main_mock(args)` is identical but uses `LoggingCallableLM(_mock_solver, log=...)` and a tiny inline dataset of ~6 hand-built `dspy.Example`s with simple known-correct test code, so the full DB-and-flow shape is exercised in seconds without HF download.

---

## Self-tests (`--test`)

No pytest dependency. Each test is a function returning `(name, ok, detail)`. The runner prints a table and exits nonzero on any failure.

### Test 1 — `test_metric_scoring`

Pure metric verification. No DSPy, no LM, no callbacks.

| Case | code | expected score |
|---|---|---|
| pass | `def add(a,b):\n  return a+b` with `assert candidate(1,2)==3` | 1.0 |
| fail | `def add(a,b):\n  return a-b` | 0.0 |
| syntax error | `def add(a,b: return a+b` | 0.0 |
| infinite loop | `def add(a,b):\n  while True: pass` (timeout=2s for this test) | 0.0 |
| crash | `def add(a,b):\n  raise ValueError("x")` | 0.0 |

All five must match exactly, AND the infinite-loop case must complete in < timeout + 2s wall.

### Test 2 — `test_bootstrap_selects_passing_demos`

Builds a 6-example trainset of toy arithmetic problems. `CallableLM` answers correctly for examples 0, 2, 4 and incorrectly for 1, 3, 5 (keyed on prompt content). Runs `BootstrapFewShot(metric=metric, max_bootstrapped_demos=4, max_labeled_demos=0).compile(student, trainset)`.

Asserts:
- `len(compiled.predictors()[0].demos) >= 3` (the three passing examples).
- Demo prompts ⊆ the prompts of examples 0, 2, 4.
- `lm.calls` contains entries for all six trainset prompts.

### Test 3 — `test_full_harness_smoke`

Runs `main_mock` over a 4-example dataset, then inspects the resulting SQLite DB.

Asserts:
- File exists, has > 0 rows.
- Every expected `event_type` appears at least once:
  `run.start, flow.start, module.start, module.end, lm.request, lm.response, adapter.format.end, adapter.parse.end, metric.score, evaluate.end, run.end`.
- `event_type='metric.score'` rows include the expected `task_id`s and float scores in [0, 1].
- There exist `event_type='flow.start'` rows for `eval_baseline`, `optimize`, `eval_optimized`.
- `compiled_humaneval.json` is loadable via `dspy.Predict(Solve).load(...)`.
- All `payload` columns parse as valid JSON.
- No row has a `NULL` `payload`.

### Test 4 — `test_to_jsonable_robustness`

Pure function. Throws every weird object at `to_jsonable` and asserts no exception + valid JSON:
- `dspy.Example(...).with_inputs("a")`
- `dspy.Prediction(code="x")`
- the `Solve` signature class
- a fresh `dspy.LM("openai/gpt-4o-mini")` instance (no network)
- a coroutine object (without awaiting it)
- a `pydantic.BaseModel` subclass instance
- a deeply nested dict with bytes / set / frozenset values
- a 1 MB string (must be truncated)

### Test 5 — `test_logging_lm_captures_payload`

With a `LoggingCallableLM` wired to a `SQLiteWriter`, run one `Predict` call. Then query the DB:
- Exactly one `lm.request` and one `lm.response` row for that call.
- `req_id` matches across the pair.
- `messages` in `lm.request` payload is non-empty and includes the input prompt.
- `lm.response` payload has a `response` field and a numeric `dt`.

---

## Test runner output (target)

```
$ python humaneval_dspy_harness.py --test
[PASS] test_metric_scoring            (5/5 cases, 2.41s)
[PASS] test_bootstrap_selects_passing_demos (3 demos selected as expected, 0.18s)
[PASS] test_full_harness_smoke        (event_types: 11/11 present, 412 rows, 0.95s)
[PASS] test_to_jsonable_robustness    (8/8 cases, 0.02s)
[PASS] test_logging_lm_captures_payload (req/resp pair matched, 0.08s)

5 passed, 0 failed in 3.64s
```

On failure, the runner prints the offending assertion's expected/actual values and exits with code 1.

---

## Implementation notes / gotchas baked in

- **`if __name__ == "__main__":` guard** wraps everything orchestration-level. `multiprocessing` workers need it on `spawn`, and `_worker` is defined at module top level so it's importable in spawned children.
- **`mp.set_start_method("fork", force=True)` on POSIX**, fallback to `spawn` on Windows. Documented with a comment about fork-after-threads being technically unsafe but fine for short-lived `exec` workers.
- **`asyncio.run(main_real(args))`** at the bottom; `Evaluate` itself is sync and drives the asyncified student through its thread pool — clean.
- **`SQLiteWriter.close()` in a `finally:`** so the daemon thread drains. Daemon=True is belt-and-suspenders.
- **Callback hooks never raise.** Every body wrapped in `try/except`, logs to stderr on failure.
- **`call_id` parent tracking** via `contextvars.ContextVar[str | None]`, with the token stored in `self._tokens[call_id]` on start and popped on end. Survives async correctly because `ContextVar` propagates through `await`.
- **`flow` ContextVar** for the same reason.
- **LM kwarg sanitization** strips api keys / base URLs before logging.
- **Payload truncation** at 256 KiB.
- **DSPy stderr logger** set to `WARNING` via `dspy.configure_dspy_loggers("dspy")` + `logging.getLogger("dspy").setLevel(WARNING)`.
- **`dspy.Code` extraction helper** tries three forms with a logged warning if all fail.
- **Bootstrap zero-demo guard**: `_check_demos(compiled, writer)` logs `bootstrap.no_demos` and `sys.exit(2)` with a message suggesting either lowering the threshold (`score >= 0.8` inside the `trace is not None` branch) or passing a stronger `teacher=` to `compile`.
- **`run.start` row** contains: `dspy.__version__`, `datasets.__version__`, `sys.version`, `platform.platform()`, `args.__dict__`, `seed`, `model`, `train_size`, `dev_size`, `num_threads`, `timeout`, `mem_limit`, `cpu_limit`.
- **`--mock` does not download HumanEval+.** Uses a tiny inline dataset.
- **All paths absolute** when passed to `multiprocessing.Process` or `sqlite3.connect`.

---

## Code-quality requirements

- Full type annotations on every function and method (PEP 604 unions, `from __future__ import annotations`).
- Passes `ruff check humaneval_dspy_harness.py` with default ruleset plus `E`, `F`, `I`, `B`, `UP`, `SIM`, `RUF`.
- Passes `ruff format --check humaneval_dspy_harness.py`.
- Passes `ty check humaneval_dspy_harness.py`.
- No `# type: ignore` except where genuinely needed for DSPy untyped surfaces; each occurrence commented with the reason.
- Docstrings on all public classes and functions.
- No global mutable state except: `_writer` (module-level binding for the metric), `_current_call_id` and `_current_flow` (`ContextVar`s). All others scoped.

---

## Build & iterate workflow (what happens after you approve this plan)

1. Generate `humaneval_dspy_harness.py` matching this spec.
2. Run `--test` first; iterate on bugs until all 5 self-tests PASS.
3. Run `--mock` end-to-end; iterate until clean exit + valid DB.
4. Run `ruff check`, `ruff format --check`, `ty check`; fix all findings.
5. Run real mode against `openai/gpt-4o-mini` with the locked defaults; report baseline vs optimized scores.
6. If `BootstrapFewShot` produces zero demos in the real run, flip to the soft-threshold path (`score >= 0.8` under `trace is not None`) and re-run.

---

## Out of scope for v1

- Streaming / `streamify`.
- Multiple models or model routing.
- LM-judge / semantic metrics.
- `MIPROv2`, `GEPA`, or any other optimizer beyond `BootstrapFewShot`.
- A web/notebook UI over the SQLite DB.
- Cost / token budget enforcement.

All of these are mechanically additive: same metric contract, same dataset shape, same logging machinery.
