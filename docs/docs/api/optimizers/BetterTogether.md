# dspy.BetterTogether

**BetterTogether** is a meta-optimizer proposed in the paper [Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together](https://arxiv.org/abs/2407.10930) by Dilara Soylu, Christopher Potts, and Omar Khattab. It combines prompt optimization and weight optimization (fine-tuning) by applying them in a configurable sequence, allowing a student program to iteratively improve both its prompts and model parameters. The core insight is that prompt and weight optimization can complement each other: prompt optimization can potentially discover effective task decompositions and reasoning strategies, while weight optimization can specialize the model to execute these patterns more efficiently. Using these approaches together in sequences (e.g., prompt optimization then weight optimization) may allow each to build on the improvements made by the other.

<!-- START_API_REF -->
::: dspy.BetterTogether
    handler: python
    options:
        members:
            - compile
            - get_params
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
:::
<!-- END_API_REF -->

## Usage Examples

See BetterTogether usage tutorials in [BetterTogether Tutorials](../../../tutorials/bettertogether/index.md).

## How BetterTogether Works

BetterTogether executes optimizers in a configurable sequence, evaluating each intermediate result and returning the best performing program. Here's how it works:

### 1. **Initialization with Custom Optimizers**

When initialized, BetterTogether accepts any DSPy optimizers (Teleprompters) as keyword arguments. The keys become the optimizer names used in the strategy string:

```python
optimizer = BetterTogether(
    metric=metric,
    p=GEPA(...),           # 'p' can be used in strategy
    w=BootstrapFinetune(...) # 'w' can be used in strategy
)
```

If no optimizers are provided, BetterTogether defaults to `BootstrapFewShotWithRandomSearch` (key: 'p') and `BootstrapFinetune` (key: 'w').

### 2. **Strategy Execution**

The strategy string defines the sequence of optimizers to apply. For example:

```python
compiled = optimizer.compile(
    student,
    trainset=trainset,
    valset=valset,
    strategy="p -> w -> p"
)
```

This strategy `"p -> w -> p"` means:

1. Run prompt optimizer ('p')
2. Run weight optimizer ('w') on the result
3. Run prompt optimizer ('p') again on the result

At each step:
- The trainset is shuffled (if `shuffle_trainset_between_steps=True`)
- The optimizer is run on the current student program
- The result is evaluated on the validation set
- The candidate program and score are recorded

Since BetterTogether is a meta-optimizer, any sequence of optimizers can be combined. The optimizer names in the strategy string correspond to the keyword arguments from initialization. For example, you can sequence different prompt optimizers (note: this illustrates BetterTogether's flexibility, not necessarily a recommended configuration):

```python
optimizer = BetterTogether(
    metric=metric,
    mipro=MIPROv2(metric=metric, auto="light"),
    gepa=GEPA(metric=metric, auto="light")
)

compiled = optimizer.compile(
    student,
    trainset=trainset,
    valset=valset,
    strategy="mipro -> gepa -> mipro"
)
```

### 3. **Validation and Program Selection**

BetterTogether can use a validation set in three ways:

- **Explicit valset**: If `valset` is provided, it's used for evaluation
- **Auto-split**: If `valset_ratio > 0`, a portion of trainset is held out for validation
- **No validation**: If both `valset` and `valset_ratio` are None/0, no validation occurs

After all optimization steps complete, the best program is selected based on validation set availability:

- **With validation**: The program with the **best score** is returned (with earlier programs winning ties)
- **Without validation**: The **latest program** is returned

If an optimization step fails:
- The error is logged with full traceback
- Optimization stops early
- The best program found so far is returned
- `flag_compilation_error_occurred` is set to `True`

The returned program includes two additional attributes:

- `candidate_programs`: List of all evaluated programs with their scores and strategies, sorted by score (best first). When an error occurs, this contains all successfully evaluated programs up to the point of failure.
- `flag_compilation_error_occurred`: Boolean indicating if any optimization step failed during compilation.

### 4. **Further Details**

**Model Lifecycle Management**: For local models (like LocalLM), BetterTogether automatically launches models before first use, kills them after optimization completes, and relaunches them after fine-tuning when model names change. These operations are no-ops for API-based LMs but needed for local model serving.

**Custom Compile Arguments**: You can pass custom compile arguments to specific optimizers using the `optimizer_compile_args` parameter:

- **Override default arguments**: Pass custom trainset/valset/teacher to specific optimizers
- **Customize per optimizer**: Each optimizer can have different compile arguments (e.g., `num_trials`, `max_bootstrapped_demos`)

Note: The `student` argument cannot be included in `optimizer_compile_args` - BetterTogether manages the student program for all optimizers. See the `compile()` method docstring for detailed argument documentation.

## Best Practices

### When to Use BetterTogether

BetterTogether is the right optimizer when:

- **You want to squeeze every bit of performance**: Prompt optimization is often the best bang for buck, quickly discovering high-level strategies. When opportunity allows, adding weight optimization on top compounds these gains, yielding benefits that exceed either approach alone.
- **You have fine-tuning capabilities**: Weight optimizers like `BootstrapFinetune` require LMs with a fine-tuning interface. Currently supported: `LocalProvider`, `DatabricksProvider`, and `OpenAIProvider`. You can extend the `Provider` class for custom use cases, or use BetterTogether to combine prompt optimizers only.

The [Databricks case study](https://www.databricks.com/blog/building-state-art-enterprise-agents-90x-cheaper-automated-prompt-optimization) demonstrates this effectiveness. They evaluated on IE Bench—a comprehensive suite spanning enterprise domains (finance, legal, commerce, healthcare) with complex challenges: 100+ page documents, 70+ extraction fields, and hierarchical schemas. Using GPT-4.1:

- **SFT alone**: +1.9 points over baseline
- **GEPA alone**: +2.1 points over baseline (slightly exceeding SFT)
- **GEPA + SFT (BetterTogether)**: +4.8 points over baseline

This demonstrates that prompt optimization can match or surpass supervised fine-tuning, and combining these techniques yields strong compounding benefits.

### Common Strategies and Optimizers

Common strategies:

- `"p -> w"`: Optimize prompts first, then fine-tune (simple and often effective)
- `"p -> w -> p"`: Optimize prompts, fine-tune, then optimize prompts again (can build on fine-tuning improvements)
- `"w -> p"`: Fine-tune first, then optimize prompts

Example optimizer combinations:

- **GEPA + BootstrapFinetune**: Prompt optimization with fine-tuning
- **MIPROv2 + BootstrapFinetune**: Prompt optimization with fine-tuning
- **Multiple prompt optimizers**: Alternate between different prompt optimization approaches (experimental)

## Further Reading

- [BetterTogether Paper: arxiv:2407.10930](https://arxiv.org/abs/2407.10930)
- [Databricks Case Study](https://www.databricks.com/blog/building-state-art-enterprise-agents-90x-cheaper-automated-prompt-optimization) - Real-world application combining BetterTogether with GEPA
- [DSPy Optimizers Overview](../../../learn/programming/optimizers.md)
