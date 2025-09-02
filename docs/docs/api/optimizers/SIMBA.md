# dspy.SIMBA

<!-- START_API_REF -->
::: dspy.SIMBA
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
<!-- END_API_REF -->

## Example Usage

```python
optimizer = dspy.SIMBA(metric=your_metric)
optimized_program = optimizer.compile(your_program, trainset=your_trainset)

# Save optimize program for future use
optimized_program.save(f"optimized.json")
```

## How `SIMBA` works
SIMBA (Stochastic Introspective Mini-Batch Ascent) is a DSPy optimizer that uses the LLM to analyze its own performance and generate improvement rules. It samples mini-batches, identifies challenging examples with high output variability, then either creates self-reflective rules or adds successful examples as demonstrations. See [this great blog post](https://blog.mariusvach.com/posts/dspy-simba) from [Marius](https://x.com/rasmus1610) for more details.