# BootstrapFewShot family

## Intent

The BootstrapFewShot family — `LabeledFewShot`, `BootstrapFewShot`, `BootstrapFewShotWithRandomSearch` (aliased `BootstrapRS`), `KNNFewShot`, and `InferRules` — are DSPy’s demo-tuning optimizers. They all share one mechanic: collect (or sample) example traces and stuff them into each predictor’s `demos` list. They differ in *how* the demos are chosen — passive sampling, metric-filtered bootstrap, random search across demo sets, KNN retrieval at inference, or rule extraction from bootstrapped traces.

Read this when you’ve picked a family member off the selection guide and want to understand how the variants relate, what each one adds over the base, and what failure modes to watch for.

## Design decisions

### 1. Demos live on the predictor, not on the optimizer

Every variant in this family writes to `predictor.demos` — a plain Python list attached to each `Predict`. The optimizer is the loader; the predictor is the storage. Once compiled, the program’s behavior is fully encoded in its predictors’ demos and signatures; the optimizer object can be discarded. `program.save(path)` pickles the demos along with the program.

### 2. `LabeledFewShot` is the no-LM baseline

Random sample (or deterministic slice) of the trainset, no teacher, no metric, no LM calls. Reach for it to confirm whether demos alone help — if `LabeledFewShot(k=16)` already gets you to your target, anything heavier is wasted effort.

### 3. `BootstrapFewShot` filters traces through a metric

The teacher runs each training example; the metric judges each completion; passing traces become demos. Every variant above the baseline uses some version of this loop. The metric is the lever — make it strict and you get few high-quality demos; make it loose and you get many lower-quality ones.

### 4. The teacher defaults to a deepcopy of the student

When you don’t pass `teacher=`, `BootstrapFewShot` calls `student.deepcopy()`. The copy is what runs against the trainset, not the original. The reason: the teacher gets `LabeledFewShot` demos applied to it (so it has examples to bootstrap from), and you don’t want those leaking into the student’s compiled state. Passing a stronger model as `teacher=` is the standard upgrade path when single-LM bootstrapping plateaus.

### 5. `max_rounds` re-runs the teacher with a fresh rollout per round

Each round bumps `rollout_id` and forces `temperature=1.0`, which busts the LM’s cache and triggers a new sample. Multiple rounds give a single example multiple chances to produce a passing trace. Don’t set `max_rounds` higher than 2 or 3 unless your metric is genuinely stringent — additional rounds compound LM cost linearly.

### 6. `max_bootstrapped_demos` caps the bootstrapped slots; `max_labeled_demos` caps the combined total

The first `max_bootstrapped_demos` slots go to traced and validated demos; the remaining slots up to `max_labeled_demos` fill with raw labeled examples from the trainset. Default `max_bootstrapped_demos=4, max_labeled_demos=16` gives 4 bootstrapped + up to 12 raw, per predictor. The raw demos act as ballast against bootstrap overfitting.

### 7. The example being bootstrapped is removed from the teacher’s demos first

Right before running the teacher on example `e`, the optimizer strips `e` from every predictor’s demo list. Without this, the teacher could “succeed” by retrieving `e`‘s answer from its own demos. The cleanup happens once per example and is silently undone for the next round.

### 8. `BootstrapRS` is `BootstrapFewShot` run N times, then evaluated

The randomness comes from shuffling the trainset and varying the bootstrap demo count per candidate. Three of the N seeds are pinned baselines (zero-shot, labeled-only, unshuffled bootstrap); the rest are randomized. The valset (defaults to trainset) is the tiebreaker — the highest-scoring candidate wins. Use it when one bootstrap pass produces inconsistent demo quality.

### 9. `KNNFewShot` picks demos at inference time, not compile time

The trainset is embedded once at construction. Each forward call embeds the new input, retrieves the `k` nearest training examples, and runs `BootstrapFewShot` on that micro-trainset before invoking the program. Per-call demo selection means different inputs see different demos — the right call when one demo set can’t generalize, expensive when called often because every forward pays for a mini-compilation.

### 10. `InferRules` extracts interpretable rules from the bootstrapped demos

On top of `BootstrapFewShot`, it asks a teacher LM to read the bootstrapped demos and produce `num_rules` natural-language rules. Those rules get appended to each predictor’s signature instructions. The win is interpretability: you can read the rules, decide whether they generalize, and edit them before saving.

### 11. A flaky metric corrupts the family

Every variant above the baseline depends on the metric to filter traces. If the metric is non-deterministic (calls an LM, depends on retrieval randomness), bootstrap discovers whichever traces happened to pass on that particular run. The demos you end up with are a function of metric noise as much as program quality. Make metrics deterministic where you can; when you can’t, accept that you should also use `BootstrapRS` and a held-out valset to average over the noise.

### 12. Compiled programs are pickled, demos and all

`program.save(path)` serializes the predictors with their `demos` lists in place. No special demo-extraction step is needed. The portability story is the same as for the spine modules: compile once, save, reload.

## API walkthrough

Grouped by family role.

### `LabeledFewShot` — the no-LM baseline

**`dspy.LabeledFewShot(k=16)`**
**`.compile(student, *, trainset, sample=True)`**

No teacher, no metric, no bootstrap loop. `.compile` samples up to `min(k, len(trainset))` examples from `trainset` — random (seeded with `Random(0)`) when `sample=True`, deterministic (first-k) when `sample=False` — and attaches the same set to each predictor’s `demos`. Single pass over the trainset; no LM calls.

### `BootstrapFewShot` — the core bootstrapper

**`dspy.BootstrapFewShot(metric=None, metric_threshold=None, teacher_settings=None, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, max_errors=None)`**
**`.compile(student, *, teacher=None, trainset)`**

On `.compile`, the optimizer:

1. **Initializes the teacher.** If no teacher passed, `student.deepcopy()`. If the teacher is uncompiled and `max_labeled_demos > 0`, applies `LabeledFewShot(k=max_labeled_demos)` so the teacher has demos to bootstrap from.
2. **Walks the trainset.** For each example, runs up to `max_rounds` attempts. Each round strips the current example from every predictor’s demos, copies the LM with `rollout_id=round, temperature=1.0`, and calls the teacher.
3. **Scores via the metric.** Calls `metric(example, prediction, trace)`. A truthy return (or numeric `>= metric_threshold` when set) marks the trace as passing. Exceptions in the teacher or metric increment an error counter; exceeding `max_errors` raises.
4. **Extracts demos from passing traces.** For each predictor invocation in the trace, builds a `dspy.Example(augmented=True, **inputs, **outputs)` and stores it under that predictor’s name.
5. **Assigns demos.** For each predictor: the first `max_bootstrapped_demos` slots get augmented demos; the rest, up to `max_labeled_demos`, get raw labeled examples sampled from the unbootstrapped portion of the trainset.

Returns a new compiled module; the student is unmodified.

**`metric_threshold`** is a numeric floor for float-returning metrics. Without it, the metric’s return is coerced to bool. Pass `metric_threshold=0.5` to mean “pass if score >= 0.5.”

**`teacher_settings`** is a dict merged onto `dspy.settings` for the duration of teacher calls — useful when the teacher needs a different LM or adapter than the student.

### `BootstrapFewShotWithRandomSearch` (`BootstrapRS`) — candidate search

**`dspy.BootstrapFewShotWithRandomSearch(metric, teacher_settings=None, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, num_candidate_programs=16, num_threads=None, max_errors=None, stop_at_score=None, metric_threshold=None)`**
**`.compile(student, *, teacher=None, trainset, valset=None, restrict=None, labeled_sample=True)`**

Generates `num_candidate_programs` candidates and picks the winner. Three of those seeds are pinned baselines:

- `seed=-3`: zero-shot (no demos).
- `seed=-2`: `LabeledFewShot(k=max_labeled_demos)` alone.
- `seed=-1`: `BootstrapFewShot` with the unshuffled trainset.

The remaining seeds shuffle the trainset and vary the bootstrap demo count. Each candidate gets evaluated on `valset` (defaults to trainset). Returns the highest-scoring candidate, with `best_program.candidate_programs` carrying the full ranked list of `(seed, program, score, subscores)` tuples for inspection.

`stop_at_score=N` short-circuits once a candidate’s valset score meets `N`. Use this when “good enough” is well-defined and you’d rather save the rest of the budget.

`num_threads` parallelizes candidate evaluation, not the bootstrap step.

### `KNNFewShot` — inference-time demo retrieval

**`dspy.KNNFewShot(k, trainset, vectorizer, **bootstrap_kwargs)`**
**`.compile(student, *, teacher=None)`**

The constructor embeds the entire trainset via `vectorizer` (a `dspy.Embedder`) and stores a `dspy.KNN(k, trainset, vectorizer)` instance. `.compile` overrides the student’s `forward` so each call:

1. Embeds the new input.
2. Retrieves the `k` nearest training examples.
3. Constructs a fresh `BootstrapFewShot(**bootstrap_kwargs)` and compiles a one-call program against those `k` examples.
4. Runs the compiled program on the input.

The trainset embedding is fixed at construction; changing the trainset means re-constructing the optimizer. Inference cost is amplified: every forward call runs a mini-compilation.

### `InferRules` — rule extraction on top of bootstrap

**`dspy.InferRules(num_candidates=10, num_rules=10, num_threads=None, teacher_settings=None, **bootstrap_kwargs)`**
**`.compile(student, *, teacher=None, trainset, valset=None)`**

Extends `BootstrapFewShot`. After running the base bootstrap once, it:

1. **Copies the bootstrapped student** `num_candidates` times.
2. **For each copy**, asks the teacher LM to read the bootstrapped demos and produce `num_rules` natural-language rules per predictor. The rules come from a `ChainOfThought` over a `RulesInductionProgram` signature (`examples_text -> natural_language_rules`).
3. **Appends the rules** to each predictor’s `signature.instructions`, after a “Please adhere to the following rules...” preamble.
4. **Evaluates each candidate on `valset`** and returns the best.

The rules are visible and editable — you can read them, decide which to keep, and treat the optimizer’s output as a starting point rather than a finished program.

## Cross-links

- [Optimizers: choosing one](choosing-an-optimizer.md) — the selection guide; describes when each family member wins against optimizers outside the family.
- [Metrics and evaluation](metrics-and-evaluation.md) — the metric shape every bootstrap variant uses, and the failure-score behavior that keeps one bad example from breaking a long bootstrap.
- [Modules: composing your own](modules.md) — `predictor.demos`, `_compiled` flags, and `deepcopy()` behavior, all of which the bootstrap family relies on.
- [GEPA in depth](gepa-in-depth.md) — the instruction-optimization counterpart; combine via `dspy.BetterTogether` when both knobs need to turn.
