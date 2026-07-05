# SBO Changes and Runbook

This document summarizes the changes made on top of the originally cloned
`wooginawunan/dspy` `nan/main` branch during the SBO implementation/debugging pass.
It is intended as a handoff for collaborators who need to understand what changed,
why it changed, and how to launch the current HotPotQA and AIME optimization jobs.

## High-Level Summary

The main changes are:

1. Fixed several SBO algorithm issues that affected serious/null step behavior.
2. Added detailed structured optimizer traces to experiment result JSON files.
3. Made robust loss evaluation support stochastic task-LM sampling.
4. Made judge sampling genuinely stochastic and traceable.
5. Separated optimization validation data from final test reporting when a test split exists.
6. Added configurable dataset input fields so task inputs are explicit in YAML.
7. Added parse-error retry handling and parse-error trace behavior for local small-model runs.
8. Added an AIME `math_answer_only` program to reduce JSON parse failures on local Qwen.
9. Added smoke and laptop-realistic experiment YAMLs for HotPotQA and AIME.

## Key Code Changes

### `dspy/teleprompt/sbo.py`

This is the largest change.

Algorithm fixes:

- Serious steps now require positive predicted improvement and positive actual improvement.
- Non-positive predicted improvement is always a null step.
- Lambda updates after serious steps compare the candidate against the old center, not a mutated center reference.
- Null-step traces include explicit `null_reason`.

Stochastic robust loss:

- Added `num_eval_samples`, `eval_temperature`, and `eval_cache`.
- Each validation example can be evaluated multiple times.
- Each sample receives a fresh rollout id when temperature is nonzero.
- Robust loss is averaged over all example/sample observations.

Judge stochasticity:

- Added `judge_temperature`.
- Judge calls use fresh rollout ids and `cache=False`.
- Judge sample details are stored in trace.

Trace export:

- Added `SBOResult.trace`.
- Added `trace_version`, config snapshot, dataset input fields, LM config snapshots, initial state, evaluations, iterations, and final bundle.
- Evaluation traces include per-example, per-sample predictions, losses, errors, and actual task LM calls.
- Candidate/proposer/verifier/model-value traces include raw prompts, raw responses, semantic scores, judge samples, model-value terms, and selected candidates.

Task LM call capture:

- Added `actual_task_lm_calls`.
- Entries are serialized from DSPy's LM history and include fields such as:
  - `messages`
  - `outputs`
  - `kwargs`
  - `usage`
  - `model`
  - `attempt_idx`
  - `is_retry`

Parse retry behavior:

- Added `parse_failure_retries`.
- Added `parse_retry_temperature`.
- Retries are only applied when a program call fails with `AdapterParseError`.
- Retries use a fresh rollout id and `cache=False`.
- Failed attempts and retry attempts are both preserved in `actual_task_lm_calls`.

Critique prompt cleanup:

- Answering-LM task prompt snapshots no longer include expected answer, prediction, score, or example labels.
- Failure feedback is stored separately in `feedback_texts`.
- Parse failures are recorded in `skipped_parse_failures` and are not sent as semantic critique evidence.
- If all sampled failures are parse failures, SBO uses a deterministic schema critique instead of spending another critic LM call.

Critique prompt size control:

- Added `max_critique_examples`.
- Added `max_critique_field_chars`.
- These limits only apply to critic evidence prompts.
- Full task-call details remain available in `actual_task_lm_calls`.

Clarified trace fields:

- `prompt_text_kind` clarifies what a `prompt_text` field represents.
- `critic_prompt_text` stores the full prompt sent to the critic LM.
- `critic_evidence_prompt_text` stores the batched task-prompt evidence used inside the critic prompt.

Important distinction:

- `actual_task_lm_calls[*].messages` is the actual answering-LM payload.
- `task_prompt_texts` are readable prompt snapshots.
- `critic_evidence_prompt_text` may contain multiple examples because it is evidence for the critic LM, not one answering-LM call.

### `benchmarks/optimizers/sbo.py`

The benchmark adapter now passes more config through to SBO:

- `judge_temperature`
- `judge_cache`
- `judge_max_tokens`
- `proposer_max_tokens`
- `critic_max_tokens`
- `num_eval_samples`
- `eval_temperature`
- `eval_cache`
- `parse_failure_retries`
- `parse_retry_temperature`
- `max_critique_examples`
- `max_critique_field_chars`

It also:

- Stores `self.result`.
- Stores `self.trace`.
- Filters model params before constructing role-specific judge/proposer/critic LMs so role LMs do not receive duplicate `max_tokens`, `temperature`, or `cache` arguments.

### `benchmarks/core/experiment.py`

Experiment execution now:

- Uses test split for final baseline/optimized reporting when `test_set` is nonempty.
- Uses validation split for optimization.
- Falls back to validation reporting when no test split exists.
- Logs final split selection to timeline.
- Passes `model.params` into `dspy.LM`.
- Captures optimizer trace from the optimizer adapter and writes it to results JSON.

### `benchmarks/core/logging.py`

Added:

- `log_optimizer_trace(...)`

The result JSON now includes:

```json
"optimizer_trace": {
  "optimizer": "sbo",
  "trace": { ... }
}
```

### Dataset Adapters

Files:

- `benchmarks/data_adapters/base.py`
- `benchmarks/data_adapters/hotpotqa.py`
- `benchmarks/data_adapters/aime.py`

Added configurable input field handling:

- `get_input_fields(default)`
- `apply_input_fields(examples, input_fields)`

This makes dataset inputs explicit and validated.

HotPotQA default input fields:

```yaml
params:
  use_context: true
  input_fields:
    - question
    - context
```

AIME default input fields:

```yaml
params:
  input_fields:
    - problem
```

### AIME Answer-Only Program

Files:

- `benchmarks/programs/qa.py`
- `benchmarks/programs/registry.py`
- `benchmarks/configs/programs/math_answer_only.yaml`

Added `MathAnswerOnly`.

Signature:

```python
"problem -> answer"
```

Instruction:

```text
Solve the given math problem internally. Return only the final numerical answer as an integer.
Do not include reasoning, units, markdown, or any additional text.
```

Reason:

- AIME metric only compares `answer`.
- Requiring both `solution` and `answer` caused frequent local-model JSON parse failures.
- `math_answer_only` is better suited for `ollama_chat/qwen3:4b-instruct`.

### Config Changes

Dataset configs updated:

- `benchmarks/configs/datasets/hotpotqa_standard.yaml`
- `benchmarks/configs/datasets/aime_standard.yaml`
- `benchmarks/configs/datasets/aime_fast.yaml`

Optimizer configs updated:

- `benchmarks/configs/optimizers/sbo.yaml`
- `benchmarks/configs/optimizers/sbo_light.yaml`

New experiment configs:

- `benchmarks/configs/experiments/hotpotqa_sbo_smoke.yaml`
- `benchmarks/configs/experiments/hotpotqa_sbo_realistic_20.yaml`
- `benchmarks/configs/experiments/aime_sbo_smoke.yaml`
- `benchmarks/configs/experiments/aime_sbo_realistic_10.yaml`

New program config:

- `benchmarks/configs/programs/math_answer_only.yaml`

### Tests

Added:

- `tests/teleprompt/test_sbo_trace_and_guard.py`

Coverage includes:

- Detailed trace emission.
- Non-positive predicted improvement produces null step.
- Task LM history is preserved when program calls fail.
- Parse failures are retried and retry attempts are traceable.

Run:

```bash
SBO/bin/python -m pytest tests/teleprompt/test_sbo_trace_and_guard.py
```

Expected result:

```text
4 passed
```

### Gitignore

Updated `.gitignore` for local artifacts:

- `/results/`
- `/SBO/`

## Running Experiments

Run commands assume:

- Current directory is the repo root.
- The virtual environment is `SBO`.
- Ollama is running.
- Model is pulled locally:

```bash
ollama pull qwen3:4b-instruct
```

All benchmark commands use:

```bash
SBO/bin/python benchmarks/scripts/run_experiment.py <CONFIG> --no-analysis
```

Results are written to:

```text
results/logs/
```

## HotPotQA Jobs

### HotPotQA Smoke

Config:

```text
benchmarks/configs/experiments/hotpotqa_sbo_smoke.yaml
```

Command:

```bash
SBO/bin/python benchmarks/scripts/run_experiment.py benchmarks/configs/experiments/hotpotqa_sbo_smoke.yaml --no-analysis
```

Purpose:

- Fast regression check.
- Uses 2 train examples and 1 validation example.
- Uses context-aware `simple_context`.
- Uses 1 candidate, 1 judge sample, 1 iteration.

Important parameters:

```yaml
dataset_overrides:
  train_size: 2
  dev_size: 1
  test_size: 0

optimizer_overrides:
  params:
    max_iterations: 1
    num_candidates: 1
    num_judge_samples: 1
    max_null_steps: 1
```

Recent check:

- Completed successfully.
- Parse errors: 0.
- Retry calls: 0.
- Task LM calls captured in trace.

### HotPotQA Laptop-Realistic

Config:

```text
benchmarks/configs/experiments/hotpotqa_sbo_realistic_20.yaml
```

Command:

```bash
SBO/bin/python benchmarks/scripts/run_experiment.py benchmarks/configs/experiments/hotpotqa_sbo_realistic_20.yaml --no-analysis
```

Purpose:

- More realistic local run without being too slow.
- Uses 20 train examples and 5 validation examples.
- Uses 3 candidates, 3 judge samples, 3 eval samples, 3 max iterations.

Important parameters:

```yaml
dataset_overrides:
  train_size: 20
  dev_size: 5
  test_size: 0

model_overrides:
  params:
    max_tokens: 256

optimizer_overrides:
  params:
    max_iterations: 3
    max_null_steps: 3
    num_candidates: 3
    num_judge_samples: 3
    num_eval_samples: 3
    max_critique_examples: 2
    max_critique_field_chars: 1200
```

Note:

- DSPy's HotPotQA test split does not include `context`.
- Because `simple_context` needs `context`, this config sets `test_size: 0`.
- Final reporting therefore falls back to validation.

Recent check:

- Completed successfully.
- Baseline validation: 36.0%.
- Optimized validation: 40.9%.
- Relative improvement: about 13.5%.
- Best robust validation loss: 0.4702.

## AIME Jobs

### AIME Smoke

Config:

```text
benchmarks/configs/experiments/aime_sbo_smoke.yaml
```

Command:

```bash
SBO/bin/python benchmarks/scripts/run_experiment.py benchmarks/configs/experiments/aime_sbo_smoke.yaml --no-analysis
```

Purpose:

- Fast AIME regression check.
- Uses `math_answer_only`.
- Uses 2 train examples, 1 validation example, 1 test example.
- Uses 1 candidate, 1 judge sample, 1 iteration.

Important parameters:

```yaml
program_ref: math_answer_only

model_overrides:
  params:
    max_tokens: 512

optimizer_overrides:
  params:
    max_iterations: 1
    num_candidates: 1
    num_judge_samples: 1
    critic_max_tokens: 2048
    parse_failure_retries: 2
    parse_retry_temperature: 0.7
```

Recent check:

- Completed successfully.
- Parse errors: 0.
- Retry calls: 0.
- Task LM calls captured in trace.
- Score remained 0/1, which appears to be model/math capability rather than parse instability.

### AIME Laptop-Realistic

Config:

```text
benchmarks/configs/experiments/aime_sbo_realistic_10.yaml
```

Command:

```bash
SBO/bin/python benchmarks/scripts/run_experiment.py benchmarks/configs/experiments/aime_sbo_realistic_10.yaml --no-analysis
```

Purpose:

- Final local AIME check at a moderate scale.
- Uses held-out test reporting.
- Uses `math_answer_only` to avoid solution/answer JSON schema failures.

Important parameters:

```yaml
dataset_overrides:
  train_size: 10
  dev_size: 3
  test_size: 3
  params:
    input_fields:
      - problem
    test_repeat: 1

model_overrides:
  params:
    max_tokens: 512

optimizer_overrides:
  params:
    max_iterations: 2
    max_null_steps: 2
    num_candidates: 3
    num_judge_samples: 3
    num_eval_samples: 2
    parse_failure_retries: 2
    parse_retry_temperature: 0.7
    max_critique_examples: 2
    max_critique_field_chars: 1600
    judge_max_tokens: 10
    proposer_max_tokens: 2048
    critic_max_tokens: 2048
```

Recent check:

- Completed successfully.
- Parse errors: 0.
- Retry calls: 0.
- Task LM calls captured in trace: 30.
- Baseline test: 0/3.
- Optimized test: 0/3.
- Best robust validation loss: 1.0.
- Remaining issue is model/math capability, not parsing or experiment stability.

## Parameter Reference

### Dataset Parameters

`train_size`

- Number of examples used by optimizer for critique generation and candidate feedback.

`dev_size`

- Validation examples used for SBO robust loss estimation.
- This is the optimization set.

`test_size`

- Final reporting set.
- If nonempty, baseline and optimized reports use test set.
- If empty, final reporting falls back to validation set.

`input_fields`

- Explicit fields passed to the task LM.
- HotPotQA: `question`, `context`.
- AIME: `problem`.

`test_repeat`

- AIME-specific repeat count for test examples.
- Set to 1 for local runs.

### Model Parameters

`max_tokens`

- Task LM output cap.
- AIME uses 512 to reduce truncation and parse failures.
- HotPotQA realistic uses 256.

### SBO Core Parameters

`max_iterations`

- Maximum SBO iterations.

`max_null_steps`

- Stop after this many consecutive null steps.

`num_candidates`

- Number of proposer candidates per iteration.

`num_judge_samples`

- Number of stochastic judge calls averaged per semantic score.
- This is the `J` sampling in the SBO semantic score.

`descent_param`

- Serious-step acceptance parameter.
- Candidate is accepted only when actual improvement is positive and large enough relative to predicted improvement.

`lambda_init`, `lambda_min`, `lambda_max`, `lambda_gamma`

- Sensitivity parameter controls the semantic bundle model penalty.
- `lambda_gamma` controls EMA smoothing after serious steps.

`tau_margin`

- Hinge margin used in verifier candidate filtering.

### LM Sampling Parameters

`temperature`

- General optimizer LM temperature where a role-specific temperature is not used.

`judge_temperature`

- Judge sampling temperature.
- Nonzero values use fresh rollout ids and `cache=False`.

`proposer_temperature`

- Proposer LM temperature.

`critic_temperature`

- Critic LM temperature.

`eval_temperature`

- Task LM temperature during robust loss evaluation.

`eval_cache`

- Whether robust loss task calls may use cache.
- Usually false for stochastic evaluation.

### Robust Loss Parameters

`num_eval_samples`

- Number of task-LM samples per validation example.
- Total validation observations per evaluation are approximately:

```text
dev_size * num_eval_samples
```

This estimates robustness over model stochasticity.

### Parse Retry Parameters

`parse_failure_retries`

- Number of retries after adapter parse failures.
- Retries are only for parse failures, not wrong answers.

`parse_retry_temperature`

- Temperature for retry attempts.

Retry trace fields:

- `attempt_idx`
- `is_retry`

### Critique Prompt Size Parameters

`max_critique_examples`

- Maximum number of evidence examples passed to critic prompts.

`max_critique_field_chars`

- Per-field character cap for critic evidence.
- This prevents local models with small context windows from failing on long HotPotQA/AIME examples.
- This does not truncate `actual_task_lm_calls`.

### Role LM Token Caps

`judge_max_tokens`

- Output cap for judge LM.
- Usually 10.

`proposer_max_tokens`

- Output cap for proposer LM.

`critic_max_tokens`

- Output cap for critic LM.
- AIME uses 2048 to avoid critic truncation.

## Trace Reading Guide

Result JSON path pattern:

```text
results/logs/<experiment_name>_results_<timestamp>.json
```

Important fields:

```text
optimizer_trace.trace.config
optimizer_trace.trace.dataset
optimizer_trace.trace.initial
optimizer_trace.trace.evaluations
optimizer_trace.trace.iterations
optimizer_trace.trace.final
```

Actual task LM prompts:

```text
optimizer_trace.trace.evaluations[*].examples[*].samples[*].actual_task_lm_calls[*].messages
```

Critic prompt:

```text
optimizer_trace.trace.initial.critique_generation.critic_prompt_text
optimizer_trace.trace.iterations[*].critique_generation.failure_critique_generation.critic_prompt_text
```

Readable task prompt snapshots:

```text
task_prompt_texts
```

Parse failures skipped from critic evidence:

```text
skipped_parse_failures
```

Clarifier:

- If `prompt_text_kind == "critic_evidence_task_prompt_snapshots"`, `prompt_text` may contain multiple examples.
- That is expected because it is a critic evidence block.
- It is not one task LM call.

## Known Caveats

1. HotPotQA test split from DSPy's loader lacks `context`, so context-based HotPotQA configs currently set `test_size: 0` and report on validation.
2. AIME with `qwen3:4b-instruct` remains weak mathematically. The current changes make it stable and parseable; they do not make the model solve AIME reliably.
3. `math_answer_only` is better for local smoke/realistic runs. `math_naive` and `math_cot` are still available if a stronger model is used.
4. Critique prompt evidence can be bounded for local context-window limits. Full task LM calls remain in trace.
5. Benchmark result JSON files are ignored by git under `/results/`.

## Verification Commands

Unit tests:

```bash
SBO/bin/python -m pytest tests/teleprompt/test_sbo_trace_and_guard.py
```

Compile touched modules:

```bash
SBO/bin/python -m compileall \
  dspy/teleprompt/sbo.py \
  benchmarks/optimizers/sbo.py \
  benchmarks/programs/qa.py \
  benchmarks/programs/registry.py \
  tests/teleprompt/test_sbo_trace_and_guard.py
```

Smoke checks:

```bash
SBO/bin/python benchmarks/scripts/run_experiment.py benchmarks/configs/experiments/hotpotqa_sbo_smoke.yaml --no-analysis
SBO/bin/python benchmarks/scripts/run_experiment.py benchmarks/configs/experiments/aime_sbo_smoke.yaml --no-analysis
```

Laptop-realistic checks:

```bash
SBO/bin/python benchmarks/scripts/run_experiment.py benchmarks/configs/experiments/hotpotqa_sbo_realistic_20.yaml --no-analysis
SBO/bin/python benchmarks/scripts/run_experiment.py benchmarks/configs/experiments/aime_sbo_realistic_10.yaml --no-analysis
```
