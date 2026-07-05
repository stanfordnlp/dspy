# SBO Hyperparameters and New Dataset Guide

This document explains the practical knobs for the two optimizers in this repo:

- Full SBO: `SemanticBundleOptimization`, usually launched through `optimizer_ref: sbo_light` or `optimizer_ref: sbo`.
- SBO-Lite: `SemanticBundleOptimizationLite`, launched through `optimizer_ref: sbo_lite`.

The benchmark configs live under `benchmarks/configs`. The implementation lives mainly in `dspy/teleprompt/sbo.py`, with benchmark adapter plumbing in `benchmarks/optimizers/sbo.py`.

## Which Optimizer To Use

Use full SBO when you want the paper-faithful calibrated method:

- numeric semantic judge scores,
- centered critique-alignment scores,
- positive loss-scale parameters `lambda`,
- semantic max-envelope model `M_k`,
- loss-first serious-step set over the generated candidate set.

Use SBO-Lite when you want a simpler and often more robust practical loop:

- qualitative verifier,
- candidate acceptance driven mostly by validation loss,
- no numeric semantic model,
- fewer sensitive hyperparameters.

On small local models, SBO-Lite can be easier to stabilize. Full SBO gives richer traces and is closer to the current theory.

## Full SBO Parameters

These parameters are accepted by `SemanticBundleOptimization`.

### LM Role Parameters

`judge_lm`

The model used for numeric semantic scores. It reads reference prompt, critique, and candidate prompt, then returns a float in `[-1, 1]`.

Common values:

- `null`: use the main configured model.
- `ollama_chat/qwen3:4b-instruct`: local Qwen model.
- Any DSPy/LiteLLM model string supported by the environment.

`proposer_lm`

The model used to generate candidate prompt variants.

`critic_lm`

The model used to generate critiques from validation failures.

`judge_max_tokens`

Maximum tokens for judge output. Full SBO judge only needs a number, so `10` or `32` is usually enough.

`proposer_max_tokens`

Maximum tokens for candidate generation. Increase this if candidates are truncated. Typical values:

- HotPotQA: `2048`
- AIME: `4096`

`critic_max_tokens`

Maximum tokens for critique generation. Increase this when failure evidence is long. Typical values:

- HotPotQA: `2048`
- AIME: `4096`

### Candidate Generation

`num_candidates`

Number of candidates `N` generated per iteration.

Effect:

- Higher values improve search breadth.
- Runtime increases because current full SBO evaluates every candidate.

Starting values:

- Smoke: `1`
- Laptop realistic: `2` or `3`
- Larger runs: `3` to `5`

`temperature`

Default temperature for optimizer LLM calls when a role-specific temperature is not used.

`proposer_temperature`

Temperature for candidate generation.

Suggested range:

- `0.2` for deterministic smoke tests.
- `0.4` to `0.7` for real exploration.
- Higher values can diversify candidates but produce malformed or overly broad rewrites.

`max_bundle_critique_chars`

Maximum characters from each stored critique included in proposer/verifier bundle text. This controls prompt size.

Suggested range:

- HotPotQA: `700` to `1200`
- AIME: `700` to `1500`

### Judge Sampling

`num_judge_samples`

Monte Carlo sample count `J` for the numeric judge score.

Effect:

- Higher values reduce judge noise.
- Runtime scales linearly with `num_candidates * active_bundle_size * num_judge_samples`.

Starting values:

- Smoke: `1`
- Laptop realistic: `2` or `3`
- Heavier runs: `3` to `5`

`judge_temperature`

Temperature for the judge.

Important:

- If `judge_temperature > 0`, judge calls use fresh rollout IDs and `cache=False`.
- If deterministic judging is desired, use `num_judge_samples: 1` and `judge_temperature: 0.0`.

Suggested values:

- Stochastic judge: `0.5` to `0.7`
- Deterministic judge: `0.0`

`judge_cache`

Whether role LM construction enables cache. Actual judge scoring calls force `cache=False` to make stochastic sampling meaningful.

### Robust Loss Evaluation

`num_eval_samples`

Number of stochastic task-LM samples per validation example when estimating robust loss.

Effect:

- Higher values estimate the smoothed objective more faithfully.
- Runtime scales linearly with `num_eval_samples`.

Starting values:

- Smoke: `1`
- HotPotQA realistic: `1` to `3`
- AIME realistic: `1` or `2`

`eval_temperature`

Temperature used for task-program evaluation samples.

Suggested values:

- `0.0` for deterministic task evaluation.
- `0.5` to `0.7` for stochastic robust-loss estimates.

`eval_cache`

Whether task-program evaluation calls may use the LM cache. Use `false` when measuring stochastic robustness.

### Serious-Step Controller

`descent_param`

The serious-step fraction `m` in the paper. Full SBO forms the serious set using:

```text
actual_improvement > 0
actual_improvement >= descent_param * max(predicted_improvement, 0)
```

Effect:

- Smaller values accept more candidates.
- Larger values require actual improvement to better match semantic-model-predicted improvement.

Suggested range:

- `0.05` to `0.2`
- Default: `0.1`

`max_iterations`

Maximum optimization iterations.

Suggested values:

- Smoke: `1`
- Laptop realistic: `3` to `10`
- Larger runs: `10+`

`max_null_steps`

Stop after this many consecutive null steps.

Suggested values:

- Smoke: `1`
- Real runs: `3` to `10`

`stop_on_no_improving_candidate`

If `true`, full SBO stops when no generated candidate improves validation loss. This matches the paper stop condition.

Suggested value: `true`.

### Lambda / Sensitivity Parameters

`lambda_init`

Initial loss-scale parameter. Converts unitless semantic alignment into predicted loss reduction.

Suggested value: `1.0`.

`lambda_min`

Lower clipping bound for lambda. Must be positive for full SBO theory.

Suggested value: `0.1`.

`lambda_max`

Upper clipping bound for lambda.

Suggested values:

- Conservative: `5.0`
- Default: `10.0`
- Larger values make semantic scores dominate predicted improvement more strongly.

`lambda_gamma`

EMA update weight for lambda after serious steps.

Suggested range:

- `0.1` to `0.5`
- Default: `0.3`

`lambda_stability_epsilon`

Small stabilizer in the observed lambda update denominator.

Suggested value: `0.000001`.

### Bundle And Memory

`bundle_size`

Maximum stored bundle entries after pruning.

Suggested values:

- Smoke: `4` to `6`
- Real runs: `8` to `12`

`active_bundle_size`

Number of bundle entries used in the active semantic max-envelope model.

Effect:

- Higher values use more historical critiques.
- Runtime scales linearly with active size.

Suggested values:

- HotPotQA: `2` to `3`
- AIME: `2` to `3`

`watchlist_size`

Number of non-active critiques included as lightweight proposer context.

Suggested values:

- `1` to `3`

`enable_exact_null_cuts`

If `true`, full SBO stores exact null self-cuts so that for the exact rejected prompt, the model value is at least that prompt's evaluated loss. This implements the finite-pool null-step stabilization idea.

Suggested value: `true`.

### Legacy / Diagnostic Violation Parameters

These remain in configs for trace compatibility, but full SBO now treats active/watchlist violations as diagnostics rather than hard acceptance gates.

`tau_margin`

Historical margin parameter for violation calculations.

`active_tau_margin`

Margin used when computing active-bundle diagnostic violations.

`watchlist_tau_margin`

Margin used when computing watchlist diagnostic violations.

`active_violation_tolerance`

Legacy hard-gate tolerance. In the current full SBO controller, this is diagnostic.

`watchlist_violation_tolerance`

Legacy hard-gate tolerance. In the current full SBO controller, this is diagnostic.

`tau_stop`

Legacy predicted-improvement stopping threshold. Current full SBO stop logic uses no-improving-candidate and null-step patience instead.

### Critique Prompt Controls

`max_critique_examples`

Maximum failure examples sent to the critic.

Suggested values:

- Smoke: `1` or `2`
- Real runs: `2` to `4`

`max_critique_field_chars`

Maximum characters per field in critic evidence snapshots.

Suggested values:

- HotPotQA: `1000` to `2000`
- AIME: `2000` to `4000`

### Parse-Retry Controls

`parse_failure_retries`

Number of retries when a DSPy adapter parse error occurs during task-program calls.

Useful for local small models and JSON-like output signatures.

Suggested values:

- HotPotQA answer-only: `0` or `1`
- AIME/math: `2` or `3`

`parse_retry_temperature`

Temperature used for parse retries. If unset, uses the base LM temperature or `eval_temperature`.

Suggested value: `0.7`.

### Trace Controls

`track_stats`

If `true`, writes detailed optimizer traces into result JSON:

- role LM configs,
- actual task LM calls,
- candidate prompts,
- semantic scores,
- candidate losses,
- serious/null decisions,
- bundle entries,
- exact null self-cut metadata.

Suggested value: `true` for research/debugging.

## SBO-Lite Parameters

SBO-Lite shares many practical parameters with full SBO, but it ignores the full numeric bundle model machinery.

### Used By SBO-Lite

`judge_lm`

Used as the qualitative verifier LM.

`proposer_lm`

Generates candidate prompts.

`critic_lm`

Generates critiques.

`num_candidates`

Number of candidates per iteration.

Suggested values:

- Smoke: `1`
- Real runs: `3` to `5`

`max_iterations`

Maximum iterations.

`max_null_steps`

Consecutive null-step patience.

`bundle_size`

Maximum stored critique entries.

`active_bundle_size`

Number of active critiques shown to the proposer/verifier.

`watchlist_size`

Number of older critiques shown as lightweight context.

`temperature`, `proposer_temperature`, `critic_temperature`

Sampling controls for candidate and critique generation.

`judge_temperature`

Sampling temperature for the qualitative verifier.

`num_eval_samples`, `eval_temperature`, `eval_cache`

Same robust-loss evaluation controls as full SBO.

`parse_failure_retries`, `parse_retry_temperature`

Same parse-retry controls as full SBO.

`max_critique_examples`, `max_critique_field_chars`, `max_bundle_critique_chars`

Same critique prompt size controls.

`judge_max_tokens`

For SBO-Lite, the verifier returns JSON, so it needs more than numeric full-SBO judge output.

Suggested values:

- `256` to `512`

`proposer_max_tokens`, `critic_max_tokens`

Same as full SBO.

`track_stats`

Writes the qualitative verifier trace and candidate records.

### Mostly Ignored Or Not Central In SBO-Lite

These belong to full SBO's calibrated numeric model and are not central to Lite:

- `num_judge_samples`
- `descent_param`
- `lambda_init`
- `lambda_min`
- `lambda_max`
- `lambda_gamma`
- `lambda_stability_epsilon`
- `tau_margin`
- `active_tau_margin`
- `watchlist_tau_margin`
- `active_violation_tolerance`
- `watchlist_violation_tolerance`
- `tau_stop`
- `enable_exact_null_cuts`

If present in a shared config, they are usually harmless, but they do not define the Lite algorithm's main behavior.

## Practical Starting Configs

### Full SBO HotPotQA Laptop Run

```yaml
optimizer_ref: sbo_light
program_ref: simple_context
dataset_overrides:
  train_size: 20
  dev_size: 5
  test_size: 0
  params:
    use_context: true
    input_fields: [question, context]
optimizer_overrides:
  params:
    max_iterations: 3
    max_null_steps: 3
    num_candidates: 3
    num_judge_samples: 3
    num_eval_samples: 1
    enable_exact_null_cuts: true
```

### Full SBO AIME Laptop Run

```yaml
optimizer_ref: sbo_light
program_ref: math_answer_only
model_overrides:
  params:
    max_tokens: 2048
dataset_overrides:
  train_size: 10
  dev_size: 5
  test_size: 10
  params:
    input_fields: [problem]
    test_repeat: 1
optimizer_overrides:
  params:
    max_iterations: 10
    max_null_steps: 10
    num_candidates: 3
    num_judge_samples: 3
    num_eval_samples: 2
    parse_failure_retries: 3
    proposer_max_tokens: 4096
    critic_max_tokens: 4096
    enable_exact_null_cuts: true
```

### SBO-Lite HotPotQA Run

```yaml
optimizer_ref: sbo_lite
program_ref: simple_context
optimizer_overrides:
  params:
    max_iterations: 10
    max_null_steps: 3
    num_candidates: 3
    judge_max_tokens: 512
```

## Interpreting Result JSON

Result files are saved under `results/logs`.

Important top-level fields:

- `baseline_results`: baseline evaluation before optimization.
- `optimization_results`: final optimized evaluation.
- `final_results`: compact baseline/optimized summary.
- `optimizer_trace.trace`: detailed SBO or SBO-Lite internals.

For full SBO, check:

- `optimizer_trace.trace.final.num_serious_steps`
- `optimizer_trace.trace.final.num_null_steps`
- `optimizer_trace.trace.iterations[*].candidate_records`
- `optimizer_trace.trace.iterations[*].serious_set_candidate_indices`
- `optimizer_trace.trace.iterations[*].selection_reason`
- `optimizer_trace.trace.iterations[*].null_reason`
- `optimizer_trace.trace.final.bundle[*].exact_cut_loss`

Good signs:

- Candidate records include losses for all generated candidates.
- Serious steps happen when `actual_improvement > 0`.
- Null reason is usually `no_candidate_improved_validation_loss` or `insufficient_model_confirmed_improvement`.

Concerning signs:

- Many candidates are identical to the center.
- Proposer parse warnings appear often.
- Candidate loss improves on dev but test drops sharply.
- Critiques become dataset-specific or example-specific rather than general prompt guidance.

## Adding A New Dataset

Adding a dataset has five pieces:

1. Dataset adapter.
2. Metric.
3. Program signature.
4. YAML configs.
5. Registry entries.

### Step 1: Define The Data Shape

Decide:

- Which fields are model inputs?
- Which fields are labels?
- Which field(s) should the metric compare?

Examples:

- HotPotQA input fields: `question`, `context`; label: `answer`.
- AIME input fields: `problem`; label: `answer`.

Only model input fields should be included in `input_fields`. Do not include IDs, metadata, support facts, or labels as inputs unless the task truly requires them.

### Step 2: Add A Dataset Adapter

Create a file under `benchmarks/data_adapters`, for example:

```text
benchmarks/data_adapters/my_dataset.py
```

Subclass `DatasetAdapter` and implement:

- `load_dataset`
- `get_metric`
- `get_gepa_metric`
- `name`
- `uses_context`

Inside `load_dataset`, convert raw rows to `dspy.Example` objects and call:

```python
input_fields = self.get_input_fields(default=("question", "context"))
examples = self.apply_input_fields(examples, input_fields)
```

This ensures the answering LM receives only the configured fields.

### Step 3: Add Or Reuse A Metric

Metrics live in `benchmarks/core/metrics.py`.

For a standard metric, use this signature:

```python
def my_metric(example, pred, trace=None) -> float:
    ...
```

It should return a float where:

- `1.0` is best,
- `0.0` is worst,
- intermediate values are allowed.

SBO converts score to loss as:

```text
loss = 1.0 - score
```

If you also want GEPA support, add a GEPA-compatible metric that returns `dspy.Prediction(score=..., feedback=...)`.

### Step 4: Choose Or Add A Program

Programs live in `benchmarks/programs`.

The program signature must match your dataset `input_fields`.

Examples:

```python
self.answer = dspy.Predict("context, question -> answer")
```

or:

```python
self.answer = dspy.Predict("problem -> answer")
```

For small local models, prefer simpler output signatures. For AIME, `problem -> answer` is easier than `problem -> solution, answer` because the metric only needs the final answer and parse failures are less frequent.

Register the program in `benchmarks/programs/registry.py`.

### Step 5: Register The Dataset

In `benchmarks/data_adapters/registry.py`, import and register your adapter:

```python
from data_adapters.my_dataset import MyDatasetAdapter
DatasetRegistry.register("my_dataset", MyDatasetAdapter)
```

### Step 6: Add Dataset YAML

Create:

```text
benchmarks/configs/datasets/my_dataset.yaml
```

Example:

```yaml
name: my_dataset
train_size: 50
dev_size: 50
test_size: 50
train_seed: 1
eval_seed: 2023
keep_details: true
params:
  input_fields:
    - question
    - context
```

### Step 7: Add Program YAML

Create or reuse a file under:

```text
benchmarks/configs/programs/
```

For many tasks, the program config is just a name and optional params.

### Step 8: Add Experiment YAML

Create:

```text
benchmarks/configs/experiments/my_dataset_sbo_smoke.yaml
```

Example:

```yaml
name: my_dataset_sbo_smoke
description: Quick full SBO smoke test for my dataset

dataset_ref: my_dataset
model_ref: qwen3_4b
optimizer_ref: sbo_light
program_ref: my_program
logging_ref: logging

dataset_overrides:
  train_size: 2
  dev_size: 1
  test_size: 1

optimizer_overrides:
  params:
    max_iterations: 1
    max_null_steps: 1
    num_candidates: 1
    num_judge_samples: 1
    num_eval_samples: 1
    max_critique_examples: 1

logging_overrides:
  save_models: false
```

Then run:

```bash
cd /Users/zl458/Documents/Codex/2026-06-13/sbo-study-using-dspy-i-am/dspy
SBO/bin/python benchmarks/scripts/run_experiment.py benchmarks/configs/experiments/my_dataset_sbo_smoke.yaml --no-analysis
```

### Step 9: Inspect The JSON

Before a serious run, check:

- Are `input_fields` correct?
- Do `actual_task_lm_calls` contain only intended inputs?
- Does the metric compare the right output field?
- Are parse failures rare?
- Are critiques general enough to improve the instruction?
- Does the proposer produce valid candidate instructions?

## Dataset Design Checklist

Use this checklist before launching a larger run:

- The train/dev/test split is clear.
- Optimization uses dev/validation, not final test.
- Final reporting uses test when available.
- `input_fields` exclude labels and metadata.
- Program signature exactly matches the intended inputs and outputs.
- Metric returns a score in `[0, 1]`.
- The model has enough `max_tokens` for the task.
- For local models, output format is simple.
- Smoke test runs with `train_size: 2`, `dev_size: 1`, `test_size: 1`.
- Result JSON contains full optimizer trace.

## Tuning Advice

If full SBO has too many null steps:

- Increase `num_candidates`.
- Lower `descent_param`.
- Increase proposer temperature slightly.
- Increase `max_critique_examples`.
- Check whether candidates are actually different from the center.
- Check whether validation set is too small or too hard.

If prompts overfit dev:

- Increase dev size.
- Keep a separate test set.
- Reduce `max_iterations`.
- Reduce candidate/proposer temperature.
- Use more diverse validation examples.

If runs are too slow:

- Lower `num_candidates`.
- Lower `num_judge_samples`.
- Lower `num_eval_samples`.
- Lower `active_bundle_size`.
- Reduce `max_critique_field_chars`.

If parse failures are common:

- Simplify the program output signature.
- Increase `parse_failure_retries`.
- Increase model `max_tokens`.
- Use answer-only outputs when the metric only needs the answer.
- Make the initial instruction stricter about output format.

If critiques are too noisy:

- Increase `max_critique_examples`.
- Cap very long fields with `max_critique_field_chars`.
- Use a stronger `critic_lm`.
- Inspect `critique_prompt` in JSON.

If proposer candidates are too conservative:

- Increase `proposer_temperature`.
- Increase `num_candidates`.
- Reduce active bundle size.
- Make critiques more actionable.

If proposer candidates are too wild:

- Lower `proposer_temperature`.
- Reduce `max_bundle_critique_chars`.
- Use stronger local-edit language in the initial prompt or proposer template.

