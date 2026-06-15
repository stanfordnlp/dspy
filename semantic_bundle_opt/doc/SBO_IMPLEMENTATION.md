# Semantic Bundle Optimization (SBO) Implementation

Implementation of "Semantic Bundle Methods: Rigorous Prompt Optimization via Discrete-Continuous Relaxation" for DSPy.

## Overview

SBO is a rigorous prompt optimization framework that addresses critical weaknesses in greedy methods (GEPA, OPRO):

**Problems with Greedy Methods:**
- **Limit Cycles**: Optimizer fixes error A at step t, but re-introduces error B (fixed at t-k), cycling indefinitely
- **Catastrophic Forgetting**: Only considers latest critique, forgets historical constraints
- **No Convergence Guarantees**: Can oscillate without mathematical guarantees

**SBO Solution:**
- **Bundle of Constraints**: Maintains ALL historical critiques, not just the latest
- **Cutting Plane Model**: M_k(p) = max{F̃_i - λ·Ŝ_J(p, p_i, c_i)} enforces global consistency
- **Rigorous Descent Test**: Serious vs Null steps with proven convergence to noise ball

## Algorithm Components

### 1. Judge (Semantic Inner Product Ŝ_J)
- Scores how well candidate p addresses critique c relative to reference p_ref
- Returns value in [-1, 1]: +1 = fully resolved, 0 = orthogonal, -1 = violated
- **Monte Carlo Averaging**: Ŝ_J = (1/J) Σ Judge(p, p_ref, c, ξ_j) over J samples
- Reduces variance from tokenization artifacts

### 2. Proposer (Candidate Generation)
- Generates N candidate variations addressing current critique
- **Trust Region**: Local edits only, maintains semantic proximity to center
- Diversity through temperature sampling

### 3. Verifier (Bundle Discrimination)
- Filters candidates using **Cumulative Semantic Violation**:
  ```
  p_k+1 = argmin Σ max(0, τ - Ŝ_J(p, p_i, c_i))
  ```
- Hinge loss formulation robust to noisy judge scores
- Ensures candidates satisfy ENTIRE bundle history, not just latest critique

### 4. Serious/Null Steps (Descent Control)
- **Serious Step**: F̃(p̂) - F̃(p_k+1) ≥ m·δ_k
  - Accept candidate as new center
  - Update adaptive sensitivity λ
  - Generate critique for new center

- **Null Step**: Improvement insufficient
  - Reject candidate, keep center unchanged
  - Add critique explaining failure to bundle
  - Refine cutting-plane model

### 5. Adaptive Sensitivity λ
- Bridges semantic scores ([-1,1]) to loss scale
- **Dynamic Calibration**:
  ```
  λ_obs = (F̃_old - F̃_new) / |Ŝ_J(p_new, p_old, c)|
  λ_k+1 = clip((1-γ)λ_k + γλ_obs, λ_min, λ_max)
  ```

## Mathematical Guarantees

**Theorem (Convergence)**: Under mild assumptions (local weak convexity, bounded bias/variance):
1. **Finite Termination**: No infinite null step sequences
2. **Asymptotic Optimality**: F̃(p̂) - F̃(p*) ≤ ε_total / (1-m)

Where ε_total = σ_bias·ρ + λ_max·√(2ln(1/δ)/J)
- σ_bias·ρ: Irreducible critic alignment error
- √(1/J): Reducible sampling variance

**Key Insight**: Increasing J (judge samples) directly tightens convergence radius!

## Implementation Details

### File Structure
```
dspy/teleprompt/sbo.py               # Core SBO teleprompter (~700 lines)
benchmarks/optimizers/sbo.py         # Adapter for benchmark framework
benchmarks/configs/optimizers/
  ├── sbo.yaml                       # Standard configuration
  └── sbo_light.yaml                 # Lightweight for testing
benchmarks/configs/experiments/
  ├── hotpotqa_sbo_v1.yaml           # Standard experiment
  └── hotpotqa_sbo_light.yaml        # Quick testing
benchmarks/tests/test_sbo_basic.py   # Basic functionality test
```

### Key Parameters

| Parameter | Default | Light | Description |
|-----------|---------|-------|-------------|
| `num_candidates` | 5 | 3 | N candidates per iteration (Proposer) |
| `num_judge_samples` | 3 | 2 | J Monte Carlo samples for Ŝ_J |
| `descent_param` | 0.1 | 0.1 | m ∈ (0,1) for serious step threshold |
| `lambda_init` | 1.0 | 1.0 | Initial sensitivity |
| `lambda_min/max` | 0.1/10.0 | 0.1/10.0 | Adaptive bounds |
| `lambda_gamma` | 0.3 | 0.3 | EMA smoothing |
| `tau_margin` | 0.5 | 0.5 | Slack margin in verifier |
| `max_iterations` | 50 | 20 | Budget limit |
| `max_null_steps` | 5 | 3 | Early stopping |

### LM Configuration

SBO uses **three separate LMs** for different roles:
- **Judge LM** (`judge_lm`): Semantic scoring (temperature=0.0, deterministic)
- **Proposer LM** (`proposer_lm`): Candidate generation (temperature=0.7)
- **Critic LM** (`critic_lm`): Critique generation (temperature=0.7)

All default to main model if not specified.

## Usage

### Direct API
```python
from dspy import SemanticBundleOptimization

optimizer = SemanticBundleOptimization(
    metric=my_metric,
    num_candidates=5,
    num_judge_samples=3,
    descent_param=0.1,
    max_iterations=50,
)

optimized_program = optimizer.compile(
    student=program,
    trainset=train_examples,
    valset=val_examples  # Required for robust loss estimation
)

# Access results
print(f"Iterations: {optimizer.result.total_iterations}")
print(f"Serious steps: {optimizer.result.num_serious_steps}")
print(f"Bundle size: {len(optimizer.result.bundle)}")
```

### Benchmark Framework
```bash
# Standard experiment
python scripts/run_experiment.py configs/experiments/hotpotqa_sbo_v1.yaml

# Quick testing with lighter settings
python scripts/run_experiment.py configs/experiments/hotpotqa_sbo_light.yaml
```

## Test Results

**Basic Test** (`benchmarks/tests/test_sbo_basic.py`):
- ✅ Successfully ran 1 iteration with 3 train, 2 val examples
- ✅ Generated 2 candidates via Proposer
- ✅ Filtered via Verifier with bundle scoring
- ✅ Made 1 serious step
- ✅ Terminated correctly on convergence
- ✅ Optimized program answers correctly: "What is the capital of Spain?" → "Madrid"

```
Results:
  Total iterations: 1
  Serious steps: 1
  Null steps: 0
  Bundle size: 2
  Best loss: 0.5000
```

## Advantages Over Existing Methods

### vs GEPA
- **No Limit Cycles**: Bundle prevents forgetting fixed errors
- **Global Constraints**: All historical critiques enforced, not just subset
- **Theoretical Guarantees**: Proven convergence vs heuristic

### vs OPRO
- **Structured Memory**: Bundle vs simple context window
- **Rigorous Acceptance**: Descent test vs greedy selection
- **Variance Reduction**: Monte Carlo judge scoring vs single evaluation

### vs MIPROv2
- **Simpler**: No Optuna hyperparameter search
- **Direct Optimization**: Gradient-inspired semantic descent vs combinatorial search
- **Provable**: Convergence guarantees vs empirical

## Known Limitations

1. **LM-Dependent**: Quality limited by critic/judge alignment (σ_bias term)
2. **Computational Cost**: Multiple judge samples (J) per candidate evaluation
3. **Local Convergence**: Finds local optimum based on initial prompt
4. **Validation Set Required**: Need robust loss estimation F̃(p)

## Future Work

- [ ] Benchmark against GEPA/MIPRO on HotPotQA, GSM8K
- [ ] Adaptive J (judge samples) based on confidence intervals
- [ ] Bundle compression for long optimization runs
- [ ] Multi-predictor support with per-predictor bundles
- [ ] Integration with DSPy evaluation framework

## References

Paper: "Semantic Bundle Methods: Rigorous Prompt Optimization via Discrete-Continuous Relaxation" (January 2026)

Key concepts from optimization theory:
- Proximal Bundle Methods (Kiwiel, 1990)
- Clarke Subdifferential for nonsmooth functions
- Weak convexity and proximally regular functions
- Monte Carlo variance reduction

## Citation

If using SBO in research, please cite:
```bibtex
@article{sbo2026,
  title={Semantic Bundle Methods: Rigorous Prompt Optimization via Discrete-Continuous Relaxation},
  year={2026},
  month={January}
}
```
