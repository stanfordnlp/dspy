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

The program below shows optimizing a math program with SIMBA:

```python
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

# Initialize the LM
lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_OPENAI_API_KEY')
dspy.configure(lm=lm)

# Initialize optimizer
teleprompter = dspy.SIMBA(
    metric=gsm8k_metric,
)

# Optimize program
print(f"Optimizing program with SIMBA...")
optimized_program = teleprompter.compile(
    dspy.ChainOfThought("question -> answer"),
    trainset=gsm8k.train,
    requires_permission_to_run=False,
)

# Save optimize program for future use
optimized_program.save(f"optimized.json")
```