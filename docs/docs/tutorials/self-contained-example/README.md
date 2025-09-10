# DSPy Simple Optimizer Tutorial

This tutorial demonstrates DSPy optimization techniques using a French-English vocabulary translation task. The script
includes 155 translation examples and supports three different optimizers: BootstrapFewShot, COPRO, and MIPROv2.

## Requirements

## Setup

The script is configured to use LM Studio with a local model. Ensure LM Studio is running on `http://localhost:1234`
with a loaded model. If you want to use your a different model, change it in the dspy configuration.

```bash
pip install dspy
```

This script uses the unsloth version of Google Gemma 1 (gemma-3-1b-it-GGUF/gemma-3-1b-it-Q8_0.gguf). Which you will
likely need to download with the lm studio tool.

## Usage

### Basic Evaluation

```bash
python vocab_agent.py
```

### Optimization

```bash
# BootstrapFewShot (recommended)
python vocab_agent.py --optimize bootstrap

# COPRO
python vocab_agent.py --optimize copro

# MIPROv2
python vocab_agent.py --optimize miprov2
```

## Dataset

The `vocab_examples.csv` contains 155 French-English translation pairs with correct/incorrect labels for training and
evaluation.