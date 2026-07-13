# Harness optimization × `dspy.Flex` + `dspy.GEPA` — findings & resume notes

Working notes for using `dspy.Flex` + `dspy.GEPA` as **harness optimization** (evolve the code around a
frozen model) on Harvey's Legal Agent Benchmark. Companion to `test_flex_harness_optimization.py`.
Status: **built and verified offline; not yet run end-to-end** (needs a local LAB checkout + API key).

---

## 1. The article

**"Don't Train the Model, Evolve the Harness"** — Joel Niklaus.
<https://huggingface.co/spaces/joelniklaus/harness-optimization>. An automated loop that rewrites the
*harness* (the runtime wrapper around a frozen model — context selection, tools, tool-call execution,
termination, and **deliverable landing**) against Harvey's Legal Agent Benchmark (LAB), no weight
updates. Headline: lifted DeepSeek-V4-Pro from **0% → 5.0% all-pass** (and pooled criterion rate
63.4% → 80.1% on held-out) by evolving the harness alone; the proposer is **Claude Opus 4.8**.

Two findings that make this a natural Flex+GEPA case:
- **"General agent harnesses were worse than vanilla."** Off-the-shelf frameworks scored *below* the
  plain baseline (Pi 45.4%, Goose 23.2%, mini-swe-agent 3.5% vs vanilla ~63%), and the failures were
  **delivery/format/timeout**, not reasoning — e.g. the model's answer was right but written under the
  wrong filename, so the grader read nothing and scored 0. *"The 0 was measuring the harness, not the model."*
- **Deterministic CODE mechanisms dominated** — deliverable-landing, tool-call repair, loop robustness;
  **5 of the top 6 mechanisms were code, not prompts.** The article explicitly frames GEPA as
  *prompt*-evolution and its own method as *code* evolution. Code mechanisms transferred across model
  families (+14.4 same family, +0.4 different); model-specific prompt playbooks could backfire.

---

## 2. Why `dspy.Flex` + `dspy.GEPA` fits

A `dspy.Flex` module's code (`module_src`) **is** a harness, and `dspy.GEPA` rewrites that *code*, not
just prompts. So Flex+GEPA is automated harness optimization in DSPy — it extends GEPA from the
prompt-evolution the article cites to the code/harness-evolution the article shows matters most. The
frozen model stays put; GEPA evolves the wrapper.

---

## 3. The dataset — Harvey LAB (`github.com/harveyai/harvey-labs`)

MIT-licensed, Python/`uv`, **~503 MB** (documents bundled), **1,660 tasks** across **25 practice areas**.
NOT on PyPI — clone it and point the demo at it.

- **Layout:** `tasks/<practice-area>/<task>/{task.json, documents/}`.
- **`task.json` schema:** `title`, `work_type` (analyze/draft/review/research), `tags`, `instructions`
  (short), `deliverables` (dict: filename → id), `criteria` (array of `{id, title, match_criteria,
  deliverables}`). `match_criteria` is the PASS/FAIL standard text — **there is no gold answer**.
- **`documents/`:** supporting files, typically `.docx` / `.xlsx` / `.pdf`.
- **Judge:** per-criterion LLM judge (LAB default `claude-sonnet-4-6`) gets task title + agent output +
  criterion `match_criteria` → PASS/FAIL. **all-pass** (1.0 iff *every* criterion passes) is the
  headline; **pooled criterion pass rate** is the dense diagnostic.
- **LAB's own harness** (what the article optimizes) is a file-writing agent: six tools
  (`bash/read/write/edit/glob/grep`) in a sandbox, deliverables landed as files under `output/`.

---

## 4. The demo (`test_flex_harness_optimization.py`) — NATIVE Flex, real data

```python
harness = dspy.Flex(LegalDeliverable)          # starts as the RLM baseline (an LLM-as-agent harness)
optimized = dspy.GEPA(metric=rubric_score, reflection_lm=Opus, ...).compile(harness, trainset, valset)
```

- **No seeding / no `_bind_code` / no synthetic data.** The harness *is* `dspy.Flex(signature)` (RLM
  baseline); GEPA evolves its `module_src` with the executor model (Haiku) frozen; proposer is Opus.
- **Real LAB tasks** loaded from `HARVEY_LABS_DIR` — real `instructions`, `documents`, and expert
  `criteria`. Signature: `task_instructions, documents -> deliverable`.
- **Metric = a faithful rubric judge** (`RubricJudge` signature, default `claude-sonnet-4-6` like LAB):
  per-criterion PASS/FAIL vs `match_criteria`, reported as pooled (dense, what GEPA optimizes) +
  all-pass (sparse headline).
- **Cost-bounded small defaults** (env-overridable): `N_TRAIN=3 N_VAL=2 N_TEST=3`, `MAX_CRITERIA=12`,
  `MAX_DOC_CHARS=12000`, `GEPA_BUDGET=30`. `HARVEY_TASKS="area/task,..."` hand-picks tasks.

---

## 5. Honest adaptations (single-module analog, kept cheap)

- LAB's harness writes **multiple files** in a tool sandbox; here the harness is **one `dspy.Flex`
  producing the deliverable as text**, and every criterion is judged against that text. So this
  exercises the *content/structure* harness — it does **not** model the filesystem "land the file with
  the exact name" mechanism (which, ironically, was the article's #1 gain; it needs a real sandbox).
- The judge **batches** a task's criteria into one call (LAB judges each independently) — a large cost
  saving; not LAB's exact judge code.
- Documents are extracted **best-effort** (needs `python-docx` / `pypdf` / `openpyxl` for Office/PDF;
  text formats always work) and truncated. Tasks needing full documents won't pass document-grounded
  criteria without the extractors installed.

---

## 6. Setup, run, cost

```bash
git clone https://github.com/harveyai/harvey-labs      # ~503 MB
export HARVEY_LABS_DIR=/path/to/harvey-labs
pip install python-docx pypdf openpyxl                  # doc extraction
.venv/bin/python -m pytest tests/flex/demo/test_flex_harness_optimization.py -s
```

Skips cleanly if `HARVEY_LABS_DIR` is unset. Cost is dominated by the RLM harness (multi-call) over the
documents, so start with the tiny defaults; scale `N_*` / `GEPA_BUDGET` / `MAX_CRITERIA` via env once
you've seen the number. Judge model / exec model / proposer via `HARVEY_JUDGE_LM` / `HARVEY_EXEC_LM` /
`HARVEY_REFLECTION_LM`.

---

## 7. Verified offline (no API, no run)

Against a fabricated one-task fixture + a `DummyLM` judge: the loader reads the real schema (criteria
capped, documents extracted), `dspy.Flex(LegalDeliverable)` builds the **native RLM baseline**
(`dspy.RLM(` in `module_src`, no seed), and the rubric judge scores + pads + emits failing-criteria
feedback correctly. Ruff-clean; collection skips cleanly without the dataset.

- **Bug caught this way:** `instructions` is a reserved attribute on `dspy.Signature` (can't be an
  input-field name) → the input field is `task_instructions`, mapped from LAB's `instructions` JSON key.

---

## 8. Pointers

- Demo: `tests/flex/demo/test_flex_harness_optimization.py`. Artifact: `harvey_lab_harness.json`
  (baseline + evolved `module_src` + scores).
- Related: `pajama_findings.md` + `test_flex_pajama_ensemble.py` — the same "evolve the code, not the
  model" idea applied to program-as-a-judge (and the Flex+GEPA engineering lessons: executor
  `max_tokens`, GEPA budget must exceed one eval, codification penalty).
- Article: <https://huggingface.co/spaces/joelniklaus/harness-optimization>. Benchmark:
  <https://github.com/harveyai/harvey-labs>.
