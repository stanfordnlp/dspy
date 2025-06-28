# dspy.MIPROv2

`MIPROv2` (<u>M</u>ultiprompt <u>I</u>nstruction <u>PR</u>oposal <u>O</u>ptimizer Version 2) is an prompt optimizer capable of optimizing both instructions and few-shot examples jointly. It does this by bootstrapping few-shot example candidates, proposing instructions grounded in different dynamics of the task, and finding an optimized combination of these options using Bayesian Optimization. It can be used for optimizing few-shot examples & instructions jointly, or just instructions for 0-shot optimization.

For those interested in more details, more information on `MIPROv2` along with a study on `MIPROv2` compared with other DSPy optimizers can be found in [this paper](https://arxiv.org/abs/2406.11695).

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
<!-- END_API_REF -->

## The Problem We're Solving

Let's start with a specific challenge in prompt optimization: **existing optimizers often only tackle half the problem**.

Here's what typically happens with traditional prompt optimization approaches:

1. **Bootstrap-style optimizers** focus primarily on generating good few-shot examples, but leave instruction wording largely unchanged
2. **Instruction-only optimizers** improve the task description but don't systematically select the best examples to include
3. **Manual approaches** require you to separately optimize instructions and examples, missing potential synergies between them
4. **Simple search methods** test combinations randomly or greedily, often missing better solutions

This creates several specific problems:

- **Suboptimal combinations**: The best instruction might need different examples than you'd expect, and vice versa
- **Inefficient search**: Random or exhaustive testing wastes compute on unlikely-to-succeed combinations
- **Limited exploration**: Without systematic candidate generation, you might never discover effective instruction variants
- **Fragmented optimization**: Optimizing pieces separately misses the interactions between instruction wording and example selection

## The MIPROv2 Solution

MIPROv2 solves this by **automating the entire prompt optimization process**. Think of it as having an AI assistant that:

- **Generates multiple prompt variations** for you to try
- **Tests each variation systematically** on your data
- **Learns from the results** to propose even better variations
- **Finds the optimal combination** of instruction wording and examples

Instead of you manually crafting prompts, MIPROv2 uses your language model itself to suggest improvements, then uses smart search algorithms to find the best combination.

## How MIPROv2 Works Under the Hood

Let's break down MIPROv2's process into three main stages:

### Stage 1: Gathering Building Blocks

**What happens**: MIPROv2 collects potential few-shot examples to use in prompts.

**How it works**:

- **Labeled examples**: Takes some of your existing training examples directly
- **Bootstrapped examples**: Runs your current program on training inputs to generate new examples, but only keeps the ones that score well on your metric

**Why this matters**: Having a diverse pool of high-quality examples gives MIPROv2 raw materials to build effective prompts.

### Stage 2: Generating Instruction Candidates

**What happens**: MIPROv2 creates multiple versions of the instruction text.

**How it works**:

- Uses your language model to propose different ways to phrase the task instructions
- Grounds these proposals in the actual examples and program context
- Generates variations that might emphasize different aspects (e.g., step-by-step reasoning vs. brevity)

**Example**: 

For a Q&A task, it might generate:

- "Answer the question step by step, showing your reasoning"
- "Provide a concise, accurate answer to the question"
- "Based on the given context, answer the following question"

### Stage 3: Smart Search for the Best Combination

**What happens**: MIPROv2 systematically tests different instruction + example combinations.

**How it works**:

1. **Initial trials**: Tests various combinations randomly to get baseline data
2. **Learning phase**: Builds a model of what makes prompts successful
3. **Guided search**: Uses Bayesian optimization to focus on promising combinations
4. **Refinement**: Continues testing until it finds the best-performing prompt

**The key insight**: Instead of trying every possible combination (which would be too expensive), MIPROv2 uses smart search to focus on the most promising options.

## The Technical Details (For the Curious)

### The Math Behind It

From a formal perspective, MIPROv2 is solving this optimization problem:

$$\max_{\theta} \; M(\text{Program}_\theta)$$

Where:

- $\theta$ represents all the prompt parameters (instructions + examples)
- $M$ is your evaluation metric (like accuracy)
- We want to find the $\theta$ that gives the highest score

**The challenge**: This is a "black-box" optimization problem because:

- We can't take gradients (the metric isn't differentiable)
- Small changes in prompts can have unpredictable effects
- We need to balance exploration (trying new things) with exploitation (refining what works)

### The Bayesian Optimization Approach

MIPROv2 tackles this using **Bayesian optimization**, which works like this:

1. **Build a surrogate model**: Creates a statistical model that predicts how well a prompt will perform based on past evaluations
2. **Acquisition function**: Uses this model to decide which prompt to test next (balancing trying promising options vs. exploring unknowns)
3. **Update and repeat**: After each test, updates the model and selects the next candidate

**Why this works**: Bayesian optimization is particularly good at handling noisy evaluations (which language models produce) and finding good solutions with relatively few trials.

### The Meta-Learning Aspect

An advanced feature is that MIPROv2 can **learn how to propose better instructions over time**. As it discovers what types of instructions work well for your task, it can bias future proposals toward similar patterns.

## Using MIPROv2 in Practice

### Example 1: Zero-Shot Optimization (Instructions Only)

Let's say you want to optimize just the instruction without adding examples:

```python
from dspy.teleprompt import MIPROv2

# Set up the optimizer
optimizer = MIPROv2(metric=accuracy_metric, auto="light")

# Optimize only instructions (no examples)
optimized_program = optimizer.compile(
    qa_system, 
    trainset=train_data,
    max_bootstrapped_demos=0,  # No AI-generated examples
    max_labeled_demos=0,       # No manual examples
    requires_permission_to_run=False
)
```

**What this does**:

- Takes your basic Q&A system
- Generates different instruction wordings
- Tests them to find the best one
- Returns an improved version with better instructions

### Example 2: Full Optimization (Instructions + Examples)

For maximum performance, optimize both instructions and examples:

```python
from dspy.teleprompt import MIPROv2

# Configure for more thorough optimization
optimizer = MIPROv2(
    metric=accuracy_metric,
    num_candidates=7,      # Generate 7 instruction variants
    init_temperature=0.5   # Control randomness in proposals
)

# Run full optimization
optimized_program = optimizer.compile(
    qa_system.deepcopy(),
    trainset=train_data,
    max_bootstrapped_demos=3,  # Up to 3 AI-generated examples
    max_labeled_demos=4,       # Up to 4 manual examples
    num_trials=15,             # Test 15 different combinations
    minibatch_size=25,         # Test each on 25 examples
    requires_permission_to_run=False
)
```

**What this does**:

- Creates 7 different instruction variants
- Allows up to 7 examples total in the prompt
- Tests 15 different combinations intelligently
- Returns the best-performing combination

### Key Parameters Explained

- **`auto`**: Quick presets (`"light"`, `"medium"`, `"heavy"`) that balance speed vs. thoroughness
- **`num_candidates`**: How many instruction variations to generate
- **`num_trials`**: How many combinations to test (more trials = better results but higher cost)
- **`minibatch_size`**: How many examples to test each combination on
- **`max_bootstrapped_demos`**: Maximum AI-generated examples to include
- **`max_labeled_demos`**: Maximum manual examples to include

## What You Can Expect

### The Good News

**Significant Performance Improvements**: MIPROv2 often delivers substantial gains:

- Typical improvements range from 5-15% accuracy boost
- Some cases see even larger improvements (up to 20%+ in favorable conditions)
- Works well even with limited training data

**Automated Discovery**: MIPROv2 often finds prompt strategies you wouldn't think of:

- Novel instruction phrasings that work better than obvious approaches
- Unexpected combinations of examples that complement each other well
- Task-specific optimizations tailored to your exact use case

**Flexible Application**: Works for both:

- **Zero-shot tasks**: Where you just want better instructions
- **Few-shot tasks**: Where examples significantly help performance

### The Realistic Expectations

**Cost Considerations**: 

- **Time**: Light runs take ~5-10 minutes, medium runs ~20-30 minutes, heavy runs can take hours
- **Compute**: Benefits from parallel processing if available

**Not Magic**: MIPROv2 has limitations:

- **Can't fix fundamental model limitations**: If your base model isn't capable enough, even perfect prompts won't solve everything
- **Depends on good metrics**: The optimizer is only as good as the evaluation function you provide
- **May overfit**: Can sometimes create prompts too specific to your training examples

**Quality Varies**: Results depend on:

- How much room for improvement exists in your initial prompt
- Quality and representativeness of your training data
- Appropriateness of your evaluation metric
- The specific task type

## Best Practices and Tips

### Setting Up for Success

1. **Start with representative data**: Your training set should reflect real-world usage
2. **Choose good metrics**: Use evaluation functions that capture what you actually care about
3. **Begin with light runs**: Start with `auto="light"` to get quick wins before investing in heavier optimization
4. **Monitor for overfitting**: Review the optimized prompts to ensure they're not too specific to your training data

### Common Pitfalls to Avoid

1. **Inadequate training data**: Too few or non-representative examples lead to poor optimization
2. **Wrong metrics**: Optimizing for the wrong thing (e.g., brevity when you need accuracy)
3. **Insufficient trials**: Stopping optimization too early before finding good solutions
4. **Ignoring costs**: Running unnecessarily expensive optimization when lighter approaches would suffice

### Advanced Usage Tips

**Use different models for different purposes**:
```python
# Use a strong model for generating instructions, smaller one for the task
optimizer = MIPROv2(
    metric=accuracy_metric,
    prompt_model=large_model,  # Strong model for creative instruction generation
    task_model=small_model     # Efficient model for actual task execution
)
```

**Inspect optimization results**:
```python
# Save and examine the optimized program
optimized_program.save("my_optimized_prompt")
print("Final instruction:", optimized_program.predictors[0].signature.instructions)
print("Examples used:", optimized_program.predictors[0].demos)
```

## Understanding the Results

### What MIPROv2 Gives You

After optimization, you get:

- **An improved DSPy program**: Drop-in replacement for your original with better prompts
- **Optimized instructions**: Better-worded task descriptions
- **Curated examples**: Carefully selected few-shot demonstrations
- **Performance metrics**: Data on how much improvement was achieved

### Interpreting Success

**Good signs**:

- Consistent improvement across validation examples
- Instructions that make intuitive sense for your task
- Examples that are representative and high-quality

**Warning signs**:

- Instructions that are overly specific to training data
- Inconsistent performance across different test sets
- Examples that don't generalize well

## Conclusion: When and How to Use MIPROv2

### MIPROv2 is Great For:

- **Complex multi-step programs**: Where small prompt improvements compound across steps
- **Tasks with clear success metrics**: Where you can easily measure what "better" means
- **Scenarios where manual optimization is time-consuming**: Especially complex domains where good prompts aren't obvious
- **When you have some training data**: Even a small representative set helps significantly

### Consider Alternatives When:

- **Your current prompts already work well**: If you're getting 95%+ accuracy, optimization might not be worth the cost
- **You have very limited data**: With fewer than ~10-20 examples, results may be unreliable
- **Time/cost constraints are tight**: Manual tweaking might be faster for simple cases
- **Your task is very simple**: Basic tasks might not benefit much from sophisticated optimization

### Getting Started

1. **Try a light run first**: Use `auto="light"` to get a feel for potential improvements
2. **Evaluate the results**: Test the optimized program on held-out data
3. **Scale up if promising**: Move to medium or heavy runs if initial results are encouraging
4. **Iterate and refine**: Use insights from optimization to inform further improvements

MIPROv2 represents a significant advance in automated prompt engineering. When used appropriately, it can save substantial time while delivering better results than manual optimization. The key is understanding when and how to apply it effectively for your specific use case.
