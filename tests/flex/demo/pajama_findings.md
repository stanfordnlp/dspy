# PAJAMA × `dspy.Flex` + `dspy.GEPA` — findings & resume notes

Working notes for reproducing the PAJAMA paper with `dspy.Flex` + `dspy.GEPA`. Read this first when
resuming — it captures what was built, what was verified on live models, the key (non-obvious)
finding, and the recommended next step. Companion to the demo `test_flex_pajama.py` in this folder.

---

## 1. The paper

**"Time To Impeach LLM-as-a-Judge: Programs are the Future of Evaluation"** — Tzu-Heng Huang, Harit
Vishwakarma, Frederic Sala (UW-Madison / SprocketLab). arXiv **2506.10403**. Project page:
<https://sprocketlab.github.io/PAJAMA/>.

**PAJAMA = Program-As-a-Judge for Automated Model Assessment.** Instead of calling an LLM to judge
which of two responses is better (LLM-as-a-judge), have an LLM **synthesize executable Python judging
programs** that score each response; the programs run locally — orders of magnitude cheaper,
interpretable, auditable, and less biased.

**Task.** Pairwise preference: given a query `x` and two responses `y1, y2`, decide which is better.
Each judge program `λ_i: (query, response) → [0,1]`; discretized vote `+1` if `λ_i(y1) > λ_i(y2)` else
`-1` (binary — no tie in the vote).

**Pipeline.** (1) *Synthesis*: GPT-4o writes ~**52** judging programs across **6 criteria** —
Structure, Relevance, Readability, Bias, Factuality, Safety. (2) *Aggregation*: a **Snorkel** label
model learns per-program reliability weights and combines the votes. (3) *Routing* (optional): distill
into a reward model for local use.

**Datasets.** JudgeLM, PandaLM, Prometheus (also MultiPref, Preference-700K).

**Headline numbers.**
- Programmatic judge in-domain ≈ **63–73%** vs LLM-as-judge ≈ **74–83%** (a few points lower) — but
  at **~3 orders of magnitude lower cost** ($0.053 vs $130–300 to label a set).
- Scales with #programs: **3 programs ≈ 59%**, **52 ≈ 82.2%** on Prometheus (Fig. 2).
- Beats LLM-as-judge out-of-domain on RewardBench **Chat-Hard**: +2.19 (Prometheus), +8.67 (JudgeLM).
- Less biased: +15.83% judgment consistency, −23.7% biased-response win rate vs Qwen2.5-14B.

---

## 2. Why Flex + GEPA is the natural analog

`dspy.Flex` is a module whose **code** (`module_src`) `dspy.GEPA` may rewrite — not just its prompts.
So GEPA literally does *program synthesis for the judge*: start from an LLM-as-a-judge baseline
(`dspy.RLM`) and evolve the judge's Python toward scoring the two responses in code, reserving an LLM
call only for ambiguous pairs. A single evolved program is the **single-program** analog of PAJAMA
(which uses a 52-program ensemble).

---

## 3. The demo (`test_flex_pajama.py`)

- **Data:** `BAAI/JudgeLM-100K` (HuggingFace), streamed once and cached to `judgelm_pairs.jsonl`.
  Gold winner from the GPT-4 reference scores `score=[s1, s2]`: A if `s1>s2`, B if `s2>s1`; **~5% ties
  dropped**; **balanced** A/B so chance = 50%. Responses truncated to 2000 chars.
- **Signature:** `question, response_a, response_b -> winner` ("A"/"B").
- **Baseline:** `dspy.Flex(PairwiseJudge)` → `dspy.RLM` = LLM-as-a-judge.
- **Metric:** `accuracy − LLM_CALL_PENALTY · (traced LLM calls)`, with feedback carrying PAJAMA's 6
  criteria + an explicit anti-verbosity warning.
- **Output:** prints baseline vs optimized (accuracy + avg LLM calls), `module_src`, saves
  `pajama_flex.json` + a 2-panel `pajama_improvement.png`.
- **Run:** `.venv/bin/python -m pytest tests/flex/demo/test_flex_pajama.py -s`
  (needs `ANTHROPIC_API_KEY`; ~25–35 min at shipped defaults). Models override via
  `PAJAMA_EXEC_LM` / `PAJAMA_REFLECTION_LM`. Defaults: exec `anthropic/claude-haiku-4-5`,
  reflection `anthropic/claude-opus-4-8`.
- **Knobs:** `LLM_CALL_PENALTY`, `N_TRAIN/N_VAL/N_TEST`, `REFLECTION_MINIBATCH`, `MAX_METRIC_CALLS`.
- Runs **in-process** (no sandbox), like the other demos. Adding
  `interpreter=lambda: dspy.PythonInterpreter()` would sandbox the generated judge code (ties to the
  `Add CodeAct to sandboxing` bridge work).

---

## 4. Verified results (5 live runs — exec Haiku-4.5, reflection Opus-4.8)

| # | penalty | val | minibatch | budget | test | baseline acc / calls | optimized acc / calls | GEPA codified? |
|---|---------|-----|-----------|--------|------|----------------------|-----------------------|----------------|
| 1 | 0.05    | 2   | 2         | 8      | 2    | 100% / 3.5           | **50%** / 0.5         | yes → collapse |
| 2 | 0.02    | 8   | 4         | 25     | 10   | 80% / 3.7            | 80% / 3.7             | **no change**  |
| 3 | 0.03    | 6   | 4         | 40     | 16   | 68.8% / 2.8          | **37.5%** / 0.4       | yes → collapse (below chance) |
| 4 | 0.015   | 8   | 6         | 45     | 16   | 87.5% / 3.8          | 87.5% / 3.8           | **no change**  |
| 5 | 0.04    | 16  | 8         | 70     | 40   | 85% / 3.8            | 85% / 3.8             | **no change**  |

`calls` = avg traced LLM calls per example (the cost proxy).

---

## 5. THE KEY FINDING (this is the point)

**A single GEPA-synthesized code judge does NOT reproduce PAJAMA's ~63–73% cheap-judge band on
JudgeLM — and this is fundamental, not a tuning miss.**

- The **baseline is faithful**: Flex-`RLM` LLM-as-a-judge scores **~68–88%**, matching the paper's
  LLM-judge band (74–83%).
- GEPA maximizes the metric. So the `LLM_CALL_PENALTY` knob has **no stable sweet spot** here:
  - **High penalty / small val** → GEPA codifies, but one program **overfits** and latches onto
    **verbosity/length** (the exact bias the paper flags), scoring **at or below chance** on held-out
    test (runs #1, #3).
  - **Low penalty, or robust val + wide minibatch** → GEPA **correctly refuses** the cheaper-but-worse
    code judge and keeps the accurate RLM (runs #2, #4, #5). Accuracy holds; no codification.
- Why fundamental: a single code judge genuinely can't beat the LLM on JudgeLM. The paper's own Fig. 2
  puts **3 programs at ≈59%**, so **one program is below that** and below the LLM judge. The paper's
  headline is a property of the **52-program Snorkel ensemble**, not any single program. With honest
  validation, GEPA will never adopt something worse than the RLM — which is *correct* behavior.
- Corollary: there is **no free lunch on JudgeLM**. Nearly every pair genuinely needs the LLM, so an
  accuracy-preserving hybrid can't save much; PAJAMA's cost win **comes with** an accuracy cost.

---

## 6. Recommended next step — the real reproduction

**Multi-program aggregation** (the paper's actual method). Reuses all current scaffolding (data,
signature, gold labels, `_evaluate`, plot):

1. Synthesize **K diverse code judges** — e.g. K independent `dspy.Flex` variants (different seeds /
   different criterion emphasis in the instructions), each GEPA-nudged toward a *pure-Python* scorer
   `judge(question, response) -> float` (drop the LLM fallback so each is a cheap program).
2. **Aggregate** their binary votes: start with **majority vote**; then a light **label-model** /
   reliability-weighted vote (Snorkel-style — learn per-judge accuracy from agreement) to match the
   paper. `snorkel-metal` / a tiny hand-rolled Dawid–Skene both work; majority is a fine first cut.
3. **Plot accuracy vs #programs** (the paper's Fig. 2) — expect the curve to climb from ~55–59% (1–3
   programs) toward the ~65–75% band as K grows, at near-zero LLM cost.

Success criterion ("kinda similar"): the **ensemble** (not any single program) lands in the paper's
~63–73% band at a fraction of the LLM-judge's cost, and the accuracy-vs-#programs curve trends up.

Alternative lighter deliverables if the ensemble is too much:
- **Tradeoff curve:** run the single-program demo at several penalties and plot accuracy-vs-cost (shows
  the knob and where it collapses).
- **Ship single-program honestly:** baseline matches the paper; document the finding (already in the
  demo docstring).

---

## 7. Pointers

- Demo: `tests/flex/demo/test_flex_pajama.py`. Data cache: `judgelm_pairs.jsonl`.
- Sister demos (same pattern): `test_flex_conflation.py`, `test_flex_banking77.py`.
- Sandboxing work that landed alongside: commit `a8bdcb1ec Add CodeAct to sandboxing`
  (`_sandbox_shim.py`, `bridge.py`, `flex.py` — ReAct/ReActV2 whitelisted; CodeAct/ProgramOfThought/RLM
  inherit the Flex interpreter).
- JudgeLM row schema: `question_body`, `answer1_body`, `answer2_body`, `score=[s1,s2]` (GPT-4 scores).
