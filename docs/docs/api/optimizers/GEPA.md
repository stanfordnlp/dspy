# dspy.GEPA

<!-- START_API_REF -->
::: dspy.GEPA
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
optimizer = dspy.GEPA(
    metric=your_metric,
    reflection_lm=dspy.LM(model='gpt-5', temperature=1.0, max_tokens=32000),
    max_metric_calls=5
)
optimized_program = optimizer.compile(your_program, trainset=your_trainset)

# Save optimize program for future use
optimized_program.save(f"optimized.json")
```

## How `GEPA` works
GEPA (Genetic-Pareto) is a prompt optimizer that uses natural language reflection to learn high-level rules from trial and error, sampling system-level trajectories and reflecting on them to diagnose problems, propose and test prompt updates, and combine complementary lessons from the Pareto frontier of its own attempts. More information can be found in [this paper](https://arxiv.org/abs/2507.19457).