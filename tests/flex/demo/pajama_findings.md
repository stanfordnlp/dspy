# PAJAMA × `dspy.Flex` + `dspy.GEPA` — findings & resume notes

Working notes for reproducing the PAJAMA paper with `dspy.Flex` + `dspy.GEPA`. Read this first when
resuming — it captures what was built, what was verified on live models, the key (non-obvious)
findings, the engineering lessons, and the current experiment. Companion to two demos in this folder:
`test_flex_pajama.py` (single-program, §§3–5) and `test_flex_pajama_ensemble.py` (multi-program, §§6–10).

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
- **Pure-code ceiling on JudgeLM:** surface features can't model GPT-4 preference well → individual
  ceiling ~low-60s. GEPA optimizes *to* the ceiling but cannot exceed the hypothesis class.

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
  optimized+decorrelated judges → 62%, at/above the paper's ~6-sampled-program point (~low-60s) and touching
  the bottom of its JudgeLM programmatic band (63–73%) — with far fewer programs. **The lever was
  independence, not count.** Still below the 88% LLM baseline (pure-code ceiling); at n=50 the ±~14pp CI
  means the 62/60/58 gaps are individually within noise — the reliable signal is the *curve shape* (up).

---

## 9. The reframing: GEPA *optimizes*, the paper *samples*

The paper **samples** ~52 programs (no per-program optimization) and leans on Snorkel to combine many
weak ones. GEPA instead **optimizes** each program — so the DSPy-native thesis is **fewer-but-better
judges**: a handful of GEPA-optimized, *decorrelated* judges beating many sampled ones **at matched
#programs**. The right success metric is our accuracy-vs-#programs curve sitting **above** the paper's
(their 3→59%, 52→82.2% on Prometheus; JudgeLM programmatic 63–73%) — NOT hitting 82%.

The N=10 run was *below* the paper's low end (≈chance at 3 judges vs their 59%): our optimized judges
were **worse than independent samples** because they were correlated. Chasing more judges (toward 52)
abandons the GEPA value prop; the lever is per-judge *independence*, not count.

**Decorrelation experiment (`N_JUDGES=6`, bagged) — RAN, and it validated the thesis (see §8 Run B).**
Keep judges few; invest in independence:
- **Bagging** was the lever — each judge GEPA-trains on a different random slice of the trainset
  (`BAG_FRAC=0.7`) so they can't converge on the same features, and their overfitting cancels across the
  ensemble. Result: the curve turned **upward** (58→62%) and majority went from 47.5% (Run A) to **62%**.
- Plus more train data (32), 8 distinct emphases, an explicit **anti-hardcoding** instruction, penalty
  + GEPA weighter kept.
- **Outcome:** 6 GEPA-optimized + decorrelated judges reach the paper's *matched-N* neighborhood (~low-60s)
  and the bottom of its JudgeLM band — with far fewer programs than the paper's 52. Confirmed: the lever
  is per-judge **independence**, not count.
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
