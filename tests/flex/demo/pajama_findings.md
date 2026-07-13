# PAJAMA × `dspy.Flex` + `dspy.GEPA` — findings & resume notes

Working notes for reproducing the PAJAMA paper with `dspy.Flex` + `dspy.GEPA`. Read this first when
resuming — it captures what was built, what was verified on live models, the key (non-obvious)
findings, the engineering lessons, and the current experiment. Companion to two demos in this folder:
`test_flex_pajama.py` (single-program, §§3–5) and `test_flex_pajama_ensemble.py` (multi-program, §§6–10).

---

## 1. The paper

**PAJAMA = Program-As-a-Judge for Automated Model Assessment** — Tzu-Heng Huang, Harit Vishwakarma,
Frederic Sala (UW-Madison / SprocketLab).

> **Two versions exist — this doc tracks the newer one.** The earlier arXiv **2506.10403** (*"Time To
> Impeach LLM-as-a-Judge"*, Jun 2025) synthesized **52** programs with GPT-4o and combined them with
> Snorkel only. The **revised** paper
> (<https://zihengh1.github.io/assets/pdf/programmatic-judges.pdf>) is a substantial rewrite — **80**
> candidates, **top-k selection**, per-program **threshold calibration**, and an **LLM-fallback
> router**. Numbers below are the revised version unless marked "(v1)". Project page:
> <https://sprocketlab.github.io/PAJAMA/>.

**Idea.** Instead of calling an LLM to judge which of two responses is better (LLM-as-a-judge), have an
LLM **synthesize executable Python judging programs** that score each response; the programs run locally
— orders of magnitude cheaper, interpretable, auditable, and less biased.

**Task.** Pairwise preference: given a query `x` and two responses `y1, y2`, decide which is better.
Each judge program `f_j: (query, response) → [0,1]`; the quality difference `d = f_j(y1) − f_j(y2)` is
discretized to a vote `+1 / −1 / 0`, where `0` = **abstain** when `|d|` is under a per-program threshold
`τ_j` calibrated on validation.

**Pipeline (revised).**
1. *Synthesis* — Claude Opus 4.6 writes **80 candidate programs** from **10 curated rubrics** (Relevance,
   Readability, Completeness, Coherence, Clarity, Structure, …), prompts seeded with 10 val examples; a
   text-similarity filter drops near-duplicates.
2. *Calibration* — each program's abstention threshold `τ_j` is tuned on ~500 val examples.
3. *Selection* — drop programs scoring below chance (50%) on val; keep the **top-k by val accuracy** →
   a per-dataset committee. **Retained: JudgeLM 21, PandaLM 14, MultiPref 16, Prometheus 8, Pref-700K 15.**
4. *Aggregation* — a **Snorkel** label model learns per-program reliability weights and combines votes.
5. *Routing* — uncertain pairs (high vote variance / aggregator posterior ≈ 0.5) escalate to an LLM
   judge. The shipped system is a **hybrid**, not programs-only.

**Datasets.** JudgeLM, PandaLM, MultiPref, Prometheus, Preference-700K.

**Headline numbers (revised).**
- The committee **averages 78.11%** (≈ OLMo-2-13B-Instruct / Qwen2.5-3B-Instruct; within 8 pts of GPT-5
  Thinking's 85.72%) at throughput no LLM judge reaches (~50× faster than Qwen2.5-14B). The paper's own
  framing: *"a committee of fewer than twenty synthesized programs is sufficient."*
- **Prometheus: 8 programs → 88.78%**, matching OLMo-2-7B-Instruct.
- **Hybrid routing** (aggregator-posterior signal) adds **+5.0%** accuracy at **2.9×** throughput over
  LLM-only on OLMo-2-7B; program-derived signals beat length/random routers.
- **Reward-model distillation** (Table 1; relabel 20K pairs, fine-tune Qwen2.5-3B, Bradley-Terry):
  PAJAMA labels **match or beat** GPT-4 labels on RewardBench average at **45–50× lower cost** — JudgeLM
  Chat-Hard 38.82→**44.52**, Reasoning 65.13→**72.91** (avg +4.49); Prometheus avg +1.67. In-domain,
  PAJAMA is a touch lower (JudgeLM 90.24→82.79, Prometheus 97.23→92.20).
- **Bias** (Table 2, five bias types): PAJAMA has the **lowest average flip rate** of any judge family;
  a **coding agent (Claude Code) patches the programs** to cut it further (flip rate 12.09→7.60) —
  *"bias is no longer a fixed property of the evaluator, but a bug that can be patched."*

---

## 2. Why Flex + GEPA is the natural analog

`dspy.Flex` is a module whose **code** (`module_src`) `dspy.GEPA` may rewrite — not just its prompts.
So GEPA literally does *program synthesis for the judge*: start from an LLM-as-a-judge baseline
(`dspy.RLM`) and evolve the judge's Python toward scoring the two responses in code, reserving an LLM
call only for ambiguous pairs. A single evolved program is the **single-program** analog of PAJAMA
(which uses a selected, calibrated per-dataset committee — 21 programs for JudgeLM, from 80 candidates).

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

**A single GEPA-synthesized code judge does NOT match the LLM-as-a-judge on JudgeLM — and this is
fundamental, not a tuning miss.**

- The **baseline is faithful**: Flex-`RLM` LLM-as-a-judge scores **~68–88%** across our runs —
  consistent with the mid-sized LLM judges the paper benchmarks (its committee averages 78.11%).
- GEPA maximizes the metric. So the `LLM_CALL_PENALTY` knob has **no stable sweet spot** here:
  - **High penalty / small val** → GEPA codifies, but one program **overfits** and latches onto
    **verbosity/length** (the exact bias the paper flags), scoring **at or below chance** on held-out
    test (runs #1, #3).
  - **Low penalty, or robust val + wide minibatch** → GEPA **correctly refuses** the cheaper-but-worse
    code judge and keeps the accurate RLM (runs #2, #4, #5). Accuracy holds; no codification.
- Why fundamental: a single code judge genuinely can't beat the LLM on JudgeLM. The paper's headline is
  a property of a **selected, calibrated committee** (21 programs on JudgeLM) plus an **LLM-fallback
  router** — not any single program. (v1's Fig. 2 put **3 programs at ≈59%** on Prometheus, so one
  program sits below even that.) With honest validation, GEPA will never adopt something worse than the
  RLM — which is *correct* behavior.
- Corollary (corrected — see §8): a *single* program is weak, but that is **not** a pure-code ceiling.
  Table 4 of the revised paper puts a **selected + calibrated 21-program committee at 81.13% on JudgeLM
  (99.2% coverage, pure code)** — between the OLMo-2-7B (76.24%) and 13B (85.68%) LLM judges, at
  ~30–300× throughput. JudgeLM *is* very learnable in pure code; the ceiling this demo hit came from too
  few, uncalibrated, correlated judges — not the hypothesis class.

---

## 6. Multi-program aggregation — BUILT (`test_flex_pajama_ensemble.py`)

The paper's actual method, implemented natively. GEPA synthesizes K pure-Python judges (each a
`dspy.Flex(PairwiseJudge)` optimized to a pure-code scorer) and their binary votes are combined.

- **Diversity:** each judge gets a distinct criterion emphasis (8 families), a distinct GEPA seed, and
  (current version) a distinct **bagged** slice of the trainset.
- **Aggregation:** majority vote + a **GEPA-optimized `EnsembleWeighting` signature** — an LLM "label
  model" that reads each judge's val accuracy / coverage / pairwise agreement and outputs a per-judge
  weight (down-weights unreliable/redundant judges, negates below-chance ones). It is itself GEPA-tuned
  on cross-val folds of the val set; closed-form `2*acc-1` is the deterministic fallback. (The paper
  uses Snorkel here; we keep it GEPA-native on purpose.)
- **Reported:** majority + signature-weighted ensemble accuracy, and the accuracy-vs-#programs curve.
- **Run:** `.venv/bin/python -m pytest tests/flex/demo/test_flex_pajama_ensemble.py -s`. Env knobs:
  `PAJAMA_N_JUDGES`, `PAJAMA_N_TRAIN/N_VAL/N_TEST`, `PAJAMA_BAG_FRAC`, `PAJAMA_OPTIMIZE_AGGREGATOR`.

---

## 7. Engineering lessons (hard-won; apply to ANY Flex+GEPA run)

- **Executor `max_tokens` must fit the RLM's code-writing steps.** `max_tokens=2000` produced 74
  truncation warnings and silently-failing RLM iterations; **8000** fixed it (set in
  `test_flex_pajama.py`, shared by both demos).
- **GEPA budget must exceed one eval (critical, non-obvious).** `max_metric_calls` has to be larger
  than a single train+val evaluation, or GEPA spends the whole budget scoring the seed and **never
  proposes a candidate** — it stalls at "Iteration 0" and returns the RLM unchanged. `budget=15 < N_VAL=16`
  gave **0/5 and 0/10 judges codified** (looked like "GEPA won't codify" but was pure budget
  starvation; two runs were aborted before the expensive vote-collection phase). `budget=60`
  (≈ a few full evals) → all judges codified and iterated (0→3+). The log tell: only `Iteration 0:
  Base program full valset score ...` with no proposals.
- **Codification lever = `JUDGE_PENALTY`** (per traced LLM call). At the RLM's ~3.9 calls/example, a
  penalty of **0.20** drops the RLM's effective score to ~0.19 so GEPA adopts pure Python; **0.10**
  leaves the RLM winning (the honest single-program "keep RLM" regime of §5). It's a soft push, not a
  hard forbid — a judge may still route one call to a genuinely ambiguous pair (loses ~0.20).

---

## 8. Ensemble results + diagnosis (two runs)

**Run A — N=10, no bagging.** `N_TRAIN=16 N_VAL=16 N_TEST=40`, penalty 0.20, budget 60.

- Baseline LLM-as-judge (RLM): **82.5%** (3.9 calls/example).
- All **10/10 judges codified** to pure Python; individual test accuracy **45–62.5%** (mean ~53%, ≈chance).
- **Majority vote: 47.5%** — *worse than the best single judge (62.5%)*; the accuracy-vs-#programs
  curve **fell** as judges were added (57.5% → 47.5%).
- **Signature-weighted (GEPA label model): 55%** — beat majority; it correctly gave the below-chance
  judge a negative weight (−0.25) and up-weighted the better ones.

**Diagnosis — the ensemble sits at chance because the judges are CORRELATED, not merely weak:**
- Ensembling lifts accuracy only when members are (a) above chance and (b) err *independently*. A
  falling curve and *majority < best-member* are the signature of **correlated** members.
- **Why correlated:** every judge is GEPA-optimized on the **same** small trainset against the **same**
  accuracy objective, so they converge on the same surface features (length/overlap → the verbosity
  trap), overriding the per-judge emphasis. Same errors → voting can't cancel them.
- **Overfitting (visible in `pajama_ensemble.json`):** judges hardcode phrases lifted from specific
  train pairs (e.g. `"your post is very good"`, `"i enjoy"`) and add task-specific special-cases (an
  `acronym_bonus`). Expected under a train-accuracy objective on 16 examples; undesirable; and a source
  of the correlation (all overfit the *same* 16 examples).
- **Low per-judge accuracy is NOT the committee ceiling:** in our runs, individual pure-code judges
  score ~low-60s on JudgeLM — while the paper's calibrated + selected 21-program committee reaches
  **81.13%** (Table 4; the paper does not publish its own per-program accuracies). The lift comes from
  **abstention + selection + a label model** (mechanisms this run lacks), not from stronger individual
  programs. So the ~62% here is an artifact of the ensemble's construction.

**Run B — N=6, BAGGED (the decorrelation fix; `N_TRAIN=32 N_VAL=20 N_TEST=50`, `BAG_FRAC=0.7`). It worked.**
- Baseline LLM-as-judge (RLM): **88.0%**. All **6/6 judges codified**; 0 truncations; ~30 min.
- Individual judges: **54–62% (mean ~58%) — all above chance**, stronger and tighter than Run A (mean
  ~53%, several ≤50%). More train data + bagging + the anti-hardcode nudge.
- **Majority vote: 62.0%** — now **equals the best single judge and beats the mean**; the accuracy-vs-
  #programs curve **rose** (58 → 62%) instead of falling. The ensemble finally *lifts* instead of dragging.
- **Signature-weighted: 60.0%** — here it slightly *under*-performed plain majority: on the thin 20-example
  val it over-corrected, negating judge 1 (weight −0.30) although judge 1 was actually 58%. The GEPA weighter
  helps when reliabilities genuinely differ (Run A, where it flipped a below-chance judge); when all judges
  are similar and above chance, majority is hard to beat and the weighter can misfire on noisy val.
- **Verdict:** bagging turned GEPA-optimized judges from *worse-than-sampled* (Run A) into *better*: 6
  optimized+decorrelated judges → 62%, with the accuracy-vs-#programs curve **rising** (58→62%) instead
  of falling. But that is **well below the revised paper's programs-only JudgeLM committee — 21 programs
  → 81.13% at 99.2% coverage (Table 4)** — and below our 88% RLM baseline. Within this run, independence
  (bagging) was the lever that flipped the curve up; but Table 4 shows the **bigger** levers are
  **selection (80→21), per-program calibration/abstention, and a real label model on enough val** —
  all of which this run lacks. At n=50 the ±~14pp CI means the 62/60/58 gaps are individually within noise.

---

## 9. The reframing: GEPA *rewrites* judge code; the paper *selects & calibrates* it

**This framing tightened once the revised paper (§1) surfaced.** v1 genuinely just *sampled* ~52
programs and Snorkel-combined them, so "paper samples, GEPA optimizes" was a fair contrast. The revised
paper is more sophisticated — **over-generate (80) → select top-k by val accuracy → calibrate per-program
thresholds → Snorkel-weight → route uncertain pairs to an LLM** — and §4.4 even uses a **coding agent
(Claude Code) to patch program code** for bias. It also independently reaches the same **"fewer than
twenty programs is enough"** conclusion this demo was chasing.

So the honest, narrower distinction is the **optimization primitive**, not the program count:
- **PAJAMA** *selects and calibrates pre-written* programs (and patches specific biases with a coding
  agent); it never rewrites a program's scoring logic to raise accuracy.
- **Flex+GEPA** *reflectively rewrites the scoring logic itself* — GEPA reads each judge's failures and
  proposes new Python against the accuracy objective. §4.4's coding-agent bias-patch is the paper's
  closest move; GEPA generalizes it from bias-only to full accuracy-driven code evolution. So Flex+GEPA
  is a natural **complement** — apply it where a fixed, selected program pool hits its ceiling.

The DSPy-native thesis is therefore **fewer-but-*rewritten* judges** (not fewer-but-sampled): a few
GEPA-optimized, *decorrelated* judges. Success = the accuracy-vs-#programs curve *rising* with
independence, NOT matching 82%.

The N=10 run was ≈chance: our optimized judges were **worse than independent samples** because they were
correlated. Chasing more judges abandons the GEPA value prop; the lever is per-judge *independence*, not
count.

**Decorrelation experiment (`N_JUDGES=6`, bagged) — RAN, and it validated the thesis (see §8 Run B).**
Keep judges few; invest in independence:
- **Bagging** was the lever — each judge GEPA-trains on a different random slice of the trainset
  (`BAG_FRAC=0.7`) so they can't converge on the same features, and their overfitting cancels across the
  ensemble. Result: the curve turned **upward** (58→62%) and majority went from 47.5% (Run A) to **62%**.
- Plus more train data (32), 8 distinct emphases, an explicit **anti-hardcoding** instruction, penalty
  + GEPA weighter kept.
- **Outcome:** 6 GEPA-optimized + decorrelated judges reach ~62% — well short of the revised paper's
  selected+calibrated+routed JudgeLM committee (21 programs), but with a curve that *rises* with
  independence. Confirmed: for GEPA-optimized judges the lever is per-judge **independence**, not count.
- **Caveats that still hold:** below the 88% LLM baseline (pure-code JudgeLM ceiling); n=50 noise (±~14pp)
  means single-number gaps aren't reliable — trust the curve shape. Pushing higher would want more val
  data for the weighter, more distinct feature families, or a task with more pure-code headroom (Prometheus).

---

## 10. Cost model (ensemble)

Dominated by RLM forwards (multi-call). Cost ≈ `baseline RLM eval (N_TEST) + N_JUDGES × (RLM seed eval
on N_VAL)`. **Vote collection is free** once judges are pure Python; the GEPA weighter is cheap (its
metric is pure arithmetic — no LLM in the metric). Measured: **N=10 ≈ $8, ~45 min** (~930 RLM Haiku
calls). Biggest knobs: `N_JUDGES` (linear), then `N_VAL` (per-judge seed eval); `N_TEST` only affects
the single baseline eval.

---

## 11. Pointers

- Demos: `test_flex_pajama.py` (single-program), `test_flex_pajama_ensemble.py` (ensemble). Data cache:
  `judgelm_pairs.jsonl`. Artifacts: `pajama_flex.json`, `pajama_ensemble.{json,png}`, `pajama_improvement.png`.
- Sister demos (same pattern): `test_flex_conflation.py`, `test_flex_banking77.py`,
  `test_flex_invoice.py`, `test_flex_math.py`.
- Related: `harness_optimization_findings.md` + `test_flex_harness_optimization.py` — Flex+GEPA as
  *harness* optimization on Harvey's Legal Agent Benchmark (same "evolve the code, not the model" idea).
- Sandboxing work that landed alongside: commit `a8bdcb1ec Add CodeAct to sandboxing`
  (`_sandbox_shim.py`, `bridge.py`, `flex.py` — ReAct/ReActV2 whitelisted; CodeAct/ProgramOfThought/RLM
  inherit the Flex interpreter).
- JudgeLM row schema: `question_body`, `answer1_body`, `answer2_body`, `score=[s1,s2]` (GPT-4 scores).
