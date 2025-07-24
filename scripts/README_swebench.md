# DSPy ReAct SweBench Benchmarking

This directory contains scripts for benchmarking DSPy ReAct's performance on the SweBench benchmark - a comprehensive evaluation framework for testing language models on real-world GitHub issues.

## Overview

SweBench evaluates models on their ability to resolve GitHub issues by generating patches that fix bugs or implement features. The benchmark uses Docker containers to create reproducible evaluation environments.

## Files

- `benchmark_swebench.py` - Main benchmarking script
- `swebench_tools.py` - SweBench-specific tools for the ReAct agent
- `swebench_adapter.py` - Adapter to convert DSPy outputs to SweBench format

## Requirements

### System Requirements
- **Storage**: ~120GB free space
- **Memory**: 16GB RAM recommended
- **CPU**: 8+ cores recommended
- **Docker**: Required for evaluation environments

### Software Requirements
```bash
# Install required packages
pip install swebench datasets docker

# Ensure Docker is running
docker --version
```

## Quick Start

### Basic Usage
```bash
# Run benchmark with OpenAI GPT-4o-mini
python scripts/benchmark_swebench.py --model openai/gpt-4o-mini

# Set OPENAI_API_KEY environment variable first
export OPENAI_API_KEY=your_api_key_here
python scripts/benchmark_swebench.py --model openai/gpt-4o-mini
```

### Configuration Options

```bash
# Use different dataset variants
python scripts/benchmark_swebench.py --model openai/gpt-4o-mini --dataset swe-bench-lite
python scripts/benchmark_swebench.py --model openai/gpt-4o-mini --dataset swe-bench-verified

# Limit instances for testing
python scripts/benchmark_swebench.py --model openai/gpt-4o-mini --max-instances 5

# Custom output directory
python scripts/benchmark_swebench.py --model openai/gpt-4o-mini --output-dir my_results

# Adjust parallel processing
python scripts/benchmark_swebench.py --model openai/gpt-4o-mini --max-workers 8
```

## Supported Models

### OpenAI Models
```bash
python scripts/benchmark_swebench.py --model openai/gpt-4o-mini
python scripts/benchmark_swebench.py --model openai/gpt-4o
python scripts/benchmark_swebench.py --model openai/gpt-3.5-turbo
```

### Anthropic Models
```bash
# Set ANTHROPIC_API_KEY first
export ANTHROPIC_API_KEY=your_api_key_here
python scripts/benchmark_swebench.py --model anthropic/claude-3-sonnet-20240229
```

### Other Models
The script supports any model compatible with DSPy's LM interface.

## Dataset Variants

- **`swe-bench`** - Full dataset (2,294 instances)
- **`swe-bench-lite`** - Curated subset (534 instances) - **Recommended for initial testing**
- **`swe-bench-verified`** - Human-verified subset (500 instances)

## How It Works

### 1. Repository Setup
For each SweBench instance, the script:
- Clones the target repository
- Checks out the specific commit before the issue was fixed
- Sets up the evaluation environment

### 2. ReAct Agent Execution
The DSPy ReAct agent is equipped with tools to:
- Read and write files
- Execute shell commands
- Run tests
- Search code patterns
- Generate git diffs

### 3. Solution Generation
The agent:
- Analyzes the problem statement
- Explores the repository structure
- Identifies relevant files and code patterns
- Generates a solution patch
- Tests the solution when possible

### 4. Evaluation
The generated patches are:
- Applied to Docker containers with the original repository state
- Tested against the repository's test suite
- Scored based on whether they resolve the issue

## Output Structure

```
swebench_results/
├── predictions.jsonl          # Generated patches in SweBench format
├── benchmark_summary.json     # Overall results summary
├── repos/                     # Cloned repositories
│   ├── instance_1/
│   └── instance_2/
├── logs/
│   ├── build_images/          # Docker build logs
│   └── run_evaluation/        # Evaluation logs
└── evaluation_results/        # Detailed evaluation results
```

## Example Output

```json
{
  "model": "openai/gpt-4o-mini",
  "dataset": "swe-bench-lite", 
  "score": 0.1273,
  "timestamp": 1703123456.789
}
```

## Performance Tips

### For Development/Testing
- Start with `--dataset swe-bench-lite --max-instances 5`
- Use `--max-workers 2` to reduce resource usage
- Monitor disk space during execution

### For Production Runs
- Ensure 120GB+ free space
- Use maximum available CPU cores with `--max-workers`
- Consider running on cloud instances with high CPU/memory

## Troubleshooting

### Common Issues

**Docker Permission Errors**
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
# Log out and back in
```

**Out of Disk Space**
```bash
# Clean up Docker containers/images
docker system prune -a
```

**API Key Errors**
```bash
# Set environment variables
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
```

**Memory Issues**
- Reduce `--max-workers`
- Use `--max-instances` to limit scope
- Consider cloud instances with more RAM

### Debug Mode
```bash
# Add debug logging
python scripts/benchmark_swebench.py --model openai/gpt-4o-mini --max-instances 1 --dataset swe-bench-lite
```

## Extending the Scripts

### Custom Tools
Add new tools to `swebench_tools.py`:

```python
def my_custom_tool(self, param: str) -> str:
    """Custom tool for specific tasks."""
    # Implementation
    return result
```

### Custom Models
Extend model initialization in `benchmark_swebench.py`:

```python
def _initialize_lm(self, model_name: str) -> dspy.LM:
    if model_name.startswith("custom/"):
        return MyCustomLM(model_name)
    # ... existing code
```

## Contributing

When modifying the scripts:
1. Maintain compatibility with SweBench evaluation format
2. Add appropriate error handling and logging
3. Test with small instance counts first
4. Update this documentation

## References

- [SweBench Paper](https://arxiv.org/abs/2310.06770)
- [SweBench Repository](https://github.com/SWE-bench/SWE-bench)
- [DSPy Documentation](https://dspy-docs.vercel.app/)