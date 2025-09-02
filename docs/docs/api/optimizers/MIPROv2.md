# dspy.MIPROv2

`MIPROv2` (<u>M</u>ultiprompt <u>I</u>nstruction <u>PR</u>oposal <u>O</u>ptimizer Version 2) is an prompt optimizer capable of optimizing both instructions and few-shot examples jointly. It does this by bootstrapping few-shot example candidates, proposing instructions grounded in different dynamics of the task, and finding an optimized combination of these options using Bayesian Optimization. It can be used for optimizing few-shot examples & instructions jointly, or just instructions for 0-shot optimization.

<!-- START_API_REF -->
::: dspy.MIPROv2
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

## Example Usage

The program below shows optimizing a math program with MIPROv2

```python
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

# Import the optimizer
from dspy.teleprompt import MIPROv2

# Initialize the LM
lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_OPENAI_API_KEY')
dspy.configure(lm=lm)

# Initialize optimizer
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    auto="medium", # Can choose between light, medium, and heavy optimization runs
)

# Optimize program
print(f"Optimizing program with MIPROv2...")
gsm8k = GSM8K()
optimized_program = teleprompter.compile(
    dspy.ChainOfThought("question -> answer"),
    trainset=gsm8k.train,
)

# Save optimize program for future use
optimized_program.save(f"optimized.json")
```

## How `MIPROv2` works

At a high level, `MIPROv2` works by creating both few-shot examples and new instructions for each predictor in your LM program, and then searching over these using Bayesian Optimization to find the best combination of these variables for your program.  If you want a visual explanation check out this [twitter thread](https://x.com/michaelryan207/status/1804189184988713065).

These steps are broken down in more detail below:

1) **Bootstrap Few-Shot Examples**: Randomly samples examples from your training set, and run them through your LM program. If the output from the program is correct for this example, it is kept as a valid few-shot example candidate. Otherwise, we try another example until we've curated the specified amount of few-shot example candidates. This step creates `num_candidates` sets of `max_bootstrapped_demos` bootstrapped examples and `max_labeled_demos` basic examples sampled from the training set.

2) **Propose Instruction Candidates**. The instruction proposer includes (1) a generated summary of properties of the training dataset, (2) a generated summary of your LM program's code and the specific predictor that an instruction is being generated for, (3) the previously bootstrapped few-shot examples to show reference inputs / outputs for a given predictor and (4) a randomly sampled tip for generation (i.e. "be creative", "be concise", etc.) to help explore the feature space of potential instructions.  This context is provided to a `prompt_model` which writes high quality instruction candidates.

3) **Find an Optimized Combination of Few-Shot Examples & Instructions**. Finally, we use Bayesian Optimization to choose which combinations of instructions and demonstrations work best for each predictor in our program. This works by running a series of `num_trials` trials, where a new set of prompts are evaluated over our validation set at each trial. The new set of prompts are only evaluated on a minibatch of size `minibatch_size` at each trial (when `minibatch`=`True`). The best averaging set of prompts is then evaluated on the full validation set every `minibatch_full_eval_steps`. At the end of the optimization process, the LM program with the set of prompts that performed best on the full validation set is returned.

For those interested in more details, more information on `MIPROv2` along with a study on `MIPROv2` compared with other DSPy optimizers can be found in [this paper](https://arxiv.org/abs/2406.11695).
