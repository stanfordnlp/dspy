# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DSPy is a framework for **programming—rather than prompting—language models**. It allows developers to build modular AI systems using compositional Python code and provides algorithms for optimizing prompts and weights. The framework implements a "declarative self-improving" approach where you write Python modules and DSPy teaches the LM to deliver high-quality outputs.

## Development Commands

### Installation & Setup

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install with specific optional dependencies
pip install -e ".[anthropic]"  # Anthropic support
pip install -e ".[mcp]"        # MCP support
pip install -e ".[weaviate]"   # Weaviate retrieval

# Install test extras
pip install -e ".[dev,test_extras]"

# Set up pre-commit hooks (required before first commit)
pre-commit install
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/path/to/test_file.py

# Run specific test function
pytest tests/path/to/test_file.py::test_function_name

# Run tests with specific marker
pytest -m "not slow"

# Run tests with coverage
pytest --cov=dspy --cov-report=html
```

### Code Quality

```bash
# Lint and format code (pre-commit handles this automatically)
ruff check --fix dspy/ tests/
ruff format dspy/ tests/

# Run pre-commit hooks manually on all files
pre-commit run --all-files

# Run pre-commit on staged files
git commit  # Hooks run automatically
```

### Building & Publishing

```bash
# Build the package
python -m build

# Install from local build
pip install dist/dspy-*.whl
```

## Architecture Overview

### Core Component Hierarchy

DSPy's architecture follows a **composition-over-inheritance** pattern with several key layers:

**1. Primitives Layer** (`dspy/primitives/`)
- **BaseModule**: Abstract base providing parameter discovery, serialization, and state management
- **Module**: Main base class with execution flow, LM binding, callbacks, history tracking, and batch processing
  - Uses **ProgramMeta** metaclass to ensure proper initialization order
  - Implements `__call__()` with callback system and usage tracking
  - `named_parameters()` recursively collects learnable components
  - `named_predictors()` finds all Predict instances
- **Example**: Flexible data container for training data with input/output separation
- **Prediction**: Extends Example to represent module outputs with LM usage and completion tracking
- **Parameter**: Marker class for learnable components

**2. Signatures** (`dspy/signatures/`)
- **Signature**: Pydantic-based input/output specification system
  - Supports both string parsing (`"input1, input2 -> output1, output2"`) and class-based definitions
  - **SignatureMeta** metaclass enables dynamic field manipulation
  - Methods: `prepend()`, `append()`, `delete()` for field manipulation
  - Manages type inference and state persistence
- **InputField/OutputField**: Pydantic FieldInfo wrappers with descriptions

**3. Predictors** (`dspy/predict/`)
All predictors extend Module and compose with each other:
- **Predict**: Base predictor mapping inputs to outputs via LM
  - Maintains `demos` (few-shot examples), `traces`, `train` data
  - Accepts `config` dict for LM parameters
- **ChainOfThought**: Adds reasoning step to signature
- **ReAct**: Reasoning + Acting with tool use
- **CodeAct**: Code execution capabilities
- **ProgramOfThought**: Numerical reasoning via code
- **BestOfN**: Multi-rollout selection with reward function
- **Refine**: Iterative refinement
- **Parallel**: Parallel execution of modules
- **KNN**: K-nearest neighbors with examples

**4. Adapters** (`dspy/adapters/`)
Translation layer between signatures and LM formats:
- **Adapter**: Base class defining `format()` and `parse()` methods
- **ChatAdapter**: Default for most LMs, uses `[[ ## field_name ## ]]` delimiters
- **JSONAdapter**: Structured output with JSON parsing
- **XMLAdapter**: XML-based structure
- **TwoStepAdapter**: Planning + Execution
- **Custom Types**: Image, Audio, File, Tool, ToolCalls, Reasoning, Code, History

**5. Language Models** (`dspy/clients/`)
- **BaseLM**: Abstract protocol defining `forward()`, `__call__()`, history tracking, callbacks
- **LM**: Concrete implementation supporting all LiteLLM-compatible providers
  - Model format: `"provider/model_name"`
  - Includes caching (via DSPY_CACHE), retry logic, finetuning support
- **Settings System**: Thread-safe global configuration with context-local overrides
  - `main_thread_config`: Global state
  - `thread_local_overrides`: Context overrides via `dspy.context()`

**6. Optimizers/Teleprompters** (`dspy/teleprompt/`)
Algorithms for optimizing prompts and few-shot examples:
- **Teleprompter**: Base class with `compile(student, trainset, teacher, valset)` method
- **BootstrapFewShot**: Generates few-shot demos via teacher model
- **MIPROv2**: Multi-objective optimization with Optuna, generates diverse demo sets + instruction variants
- **COPRO**: Compositional prompt optimization with breadth-first variation generation
- **GEPA**: Grounded exemplar prompt adaptation with trace-based optimization
- **SBO** (Semantic Bundle Optimization): Rigorous bundle method preventing limit cycles via historical critique constraints
- **SIMBA**: Simple in-context metric-based adaptation
- **Ensemble**: Combines multiple compiled programs with voting
- **LabeledFewShot**: Simple few-shot with labeled examples only

**7. Propose System** (`dspy/propose/`)
- **Proposer**: Abstract interface for generating signatures/instructions
- **GroundedProposer**: Analyzes data samples to propose field names and infer instructions
- **DatasetSummaryGenerator**: Creates summaries for proposal context

### Execution Flow

```
User calls module()
  → __call__() triggered
  → Callbacks fire (with_callbacks)
  → Usage tracking starts
  → forward() executes:
      → Calls predictors with signature-based inputs
      → Each predictor: Adapter.format() → LM.__call__() → Adapter.parse() → Prediction
  → Callbacks fire post-execution
  → Usage metrics attached to Prediction
  → History recorded in module.history
```

### Key Patterns

**Lazy LM Binding**: Modules don't require LM at init; obtained at call time via `self.lm or settings.lm`

**Parameter Discovery**: `named_parameters()` recursively collects all learnable components (Parameter or Predict instances)

**State Serialization**: All learnable components implement `dump_state()`/`load_state()` for JSON (portable state-only) or pickle (full program with cloudpickle)

**Execution Tracing**: Global `settings.trace` list tracks all LM calls; used for debugging and GEPA optimization

**Few-Shot as State**: Demonstrations stored in `predictor.demos` (list of Examples); updated during optimization and persisted via state methods

**Branch/Rollout IDs**: `branch_idx` and `rollout_id` enable cache bypass and multi-sample generation

**Thread Safety**: Settings use locks for main thread config, context vars for thread-local overrides, AsyncIO support via `asyncify()`/`syncify()`

## Important Conventions

### Code Style
- Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Line length: 120 characters
- Use double quotes for strings
- Ruff handles import sorting (isort) with `dspy` as first-party

### Module Development
- Extend `dspy.Module` for new modules
- Implement `forward()` method (not `__call__`)
- Use composition: modules contain predictors, don't extend them
- Predictors should be assigned as instance attributes for parameter discovery

### Signature Definitions
- Prefer class-based signatures with `InputField`/`OutputField` for clarity
- Use string parsing for simple signatures: `dspy.Signature("q -> a")`
- Add descriptions to fields for better LM understanding

### Testing
- Tests in `/tests/` focus on code correctness and Adapter reliability
- For end-to-end quality testing of modules/optimizers, see [LangProBe](https://github.com/Shangyint/langProBe)
- Use pytest fixtures for common setup
- Mark slow tests appropriately

### Git Workflow
- **Commit frequently**: Commit changes as you complete logical units of work (after implementing a feature, fixing a bug, or completing a task)
- Pre-commit hooks automatically run ruff on staged files
- Hooks check: YAML/TOML validity, large files, merge conflicts, debug statements
- Code must pass ruff checks before commit
- Use descriptive commit messages explaining what changed and why
- Include "Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>" in commits when appropriate

## Benchmarks Directory

The `/benchmarks/` directory contains an optimizer benchmark framework with a modular, config-driven system:

```bash
# Run baseline evaluation
python scripts/run_experiment.py configs/experiments/hotpotqa_baseline.yaml

# Run GEPA optimization with analysis
python scripts/run_experiment_with_analysis.py configs/experiments/hotpotqa_gepa_v1.yaml
```

**Configuration System**: Composable configs split into `datasets/`, `models/`, `optimizers/`, `programs/`, `experiments/`
- Experiments reference shared configs via `*_ref` fields
- Override specific params with `*_overrides` fields

**Available Optimizers**:
- `baseline`: No optimization (evaluation only)
- `gepa`: GEPA (Grounded Exemplar Prompt Adaptation)
- `mipro`: MIPROv2 (Multi-objective Instruction PRoposal Optimizer)
- `sbo`: SBO (Semantic Bundle Optimization) - prevents limit cycles via bundle method
- `bootstrap`: Bootstrap few-shot learning
- `copro`: COPRO (Compositional Prompt Optimization)

**Key Parameters for GEPA**:
- `reflection_minibatch_size`: 1-2 for small models (4B-7B), 2-3 for 70B+
- HotPotQA examples are long; prompts can exceed 5,000-7,000 tokens with minibatch_size=3
- Reduce minibatch size if seeing empty/truncated proposals

**Key Parameters for SBO**:
- `num_candidates`: Number of candidate variations per iteration (default: 5)
- `num_judge_samples`: Monte Carlo samples for semantic scoring (J in paper, default: 3)
- `descent_param`: Threshold m ∈ (0,1) for serious step acceptance (default: 0.1)
- `lambda_init/min/max`: Adaptive sensitivity parameter bounds (default: 1.0, 0.1, 10.0)
- `max_iterations`: Maximum optimization iterations (default: 50)
- `max_null_steps`: Consecutive null steps before termination (default: 5)
- Use `sbo_light.yaml` for faster iteration with fewer candidates and shorter runs

## Caching

- Two-tier cache: Memory + Disk
- Disk cache location: `~/.dspy_cache/` (default 30GB limit)
- Disable during debugging with `cache=false` in model config
- Different `rollout_id` values bypass cache for multi-sample generation

## Python Version Support

- Requires Python >=3.10, <3.15
- MCP support requires Python >=3.10
- LiteLLM proxy dependencies differ on Windows and Python 3.14

## Common Gotchas

- **Relative imports**: Banned except in `tests/` and `__init__.py` (ruff enforces)
- **Wildcard imports**: Allowed in `__init__.py` but discouraged elsewhere
- **Adapter auto-fallback**: ChatAdapter falls back to JSONAdapter on parse failure
- **Context window**: GEPA reflection prompts can be very long with HotPotQA; adjust `reflection_minibatch_size`
- **Thread safety**: Use `dspy.context()` for thread-local LM/adapter overrides
