# dspy.BootstrapFewShotWithRandomSearch

`BootstrapFewShotWithRandomSearch` is a prompt optimizer that automatically discovers the best few-shot examples for your language model. Instead of manually guessing which examples to include in your prompts, this optimizer **bootstraps new examples using your model's own successful outputs** and uses intelligent random search to find the combination that delivers the best performance.

For tasks where you have training data available, this optimizer can significantly boost your model's performance by finding the most effective demonstrations to include in your prompts.

<!-- START_API_REF -->
::: dspy.BootstrapFewShotWithRandomSearch
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

Here's a challenge every prompt engineer faces: **which examples should you include in your few-shot prompts?**

When you're building a language model application, you know that good examples in your prompt can dramatically improve performance. But choosing the right examples is surprisingly difficult:

1. **Manual selection is hit-or-miss**: You might pick examples that seem good to you but don't actually help the model learn the pattern
2. **Limited perspective**: Your training data might not cover all the scenarios where good demonstrations would help
3. **No systematic evaluation**: Without testing different combinations, you're essentially guessing
4. **Time-consuming iteration**: Manually trying different example sets and evaluating them is slow and tedious

This creates several specific problems:

- **Suboptimal demonstrations**: The examples you choose might not be the best teachers for your model
- **Missing coverage**: You might not have examples that show the model how to handle edge cases
- **Wasted potential**: Your model could perform much better with the right examples, but finding them manually is impractical
- **Inconsistent results**: Different people might choose different examples, leading to unpredictable performance

## The BootstrapFewShotWithRandomSearch Solution

BootstrapFewShotWithRandomSearch solves this by **automating the entire example selection process**. Think of it as having an expert prompt engineer that:

- **Generates additional high-quality examples** by letting your model solve problems and keeping the ones it gets right
- **Tests many different combinations** of examples systematically  
- **Finds the optimal set** that makes your model perform best on validation data
- **Uses smart random search** to explore the space of possible prompts efficiently

Instead of you manually crafting example sets, this optimizer uses your model as both teacher and student - it generates new examples with the model, then finds the best combination to teach that same model.

## How BootstrapFewShotWithRandomSearch Works

Let's break down the process into three main stages:

### Stage 1: Building a Pool of Candidate Examples

**What happens**: The optimizer gathers potential examples to use in prompts.

**How it works**:

- **Labeled examples**: Takes examples directly from your training data (up to `max_labeled_demos`)
- **Bootstrapped examples**: Uses your model as a "teacher" to solve training problems, keeping only the ones it gets right (up to `max_bootstrapped_demos`)

**Why this matters**: By combining real examples with model-generated ones, you get a richer pool of demonstrations that covers more scenarios and shows correct reasoning patterns.

**Example**: For a math word problem task, you might have 5 real examples from your training set, plus 3 additional examples where your model solved problems correctly with step-by-step reasoning.

### Stage 2: Creating Multiple Candidate Prompts

**What happens**: The optimizer creates many different prompt variants by combining examples in different ways.

**How it works**:

- Creates `num_candidate_programs` different prompts, each with a different selection and ordering of examples
- Uses randomization to explore different combinations you might never think to try manually
- Always includes some baseline comparisons (like no examples, or just labeled examples)

**The key insight**: Instead of betting everything on one example selection, the optimizer hedges by trying many different approaches simultaneously.

### Stage 3: Testing and Selecting the Best Prompt

**What happens**: Each candidate prompt is evaluated on validation data to find the winner.

**How it works**:

1. **Evaluation**: Each prompt variant is tested on your validation set using your metric
2. **Comparison**: The optimizer compares performance across all candidates
3. **Selection**: The prompt that achieves the highest score becomes your optimized program
4. **Validation**: This ensures the chosen prompt generalizes well, not just fits the training data

## The Technical Details (For the Curious)

### The Optimization Problem

From a formal perspective, BootstrapFewShotWithRandomSearch is solving this optimization challenge:

$$S^* = \arg\max_{S \subseteq D} \; M(\text{model prompted with } S,\; \text{validation data})$$

Where:
- $S$ is a set of examples to include in the prompt
- $D$ is your pool of available examples (labeled + bootstrapped)  
- $M$ is your evaluation metric
- We want to find the $S$ that gives the highest score

**The challenge**: This is computationally hard because there are exponentially many possible subsets to choose from.

### The Random Search Approach

Instead of trying every combination (which would be impossibly expensive), the optimizer uses **intelligent random search**:

1. **Smart sampling**: Generates multiple candidate example sets using randomization
2. **Parallel evaluation**: Tests many candidates simultaneously to find good solutions efficiently
3. **Best selection**: Picks the candidate that performs best on validation data

**Why this works**: Random search is surprisingly effective for this type of problem, especially when you can evaluate many candidates in parallel. It avoids the local optima that greedy approaches might get stuck in.

### The Bootstrapping Process

The "bootstrapping" aspect works like this:

1. **Teacher generation**: Your model (acting as teacher) attempts to solve training problems
2. **Quality filtering**: Only solutions that pass your metric become examples
3. **Demonstration creation**: Correct solutions (including reasoning steps) become new few-shot examples

This creates a **positive feedback loop**: the model generates examples of its own successful problem-solving, which then help it solve similar problems even better.

### Multi-Round Refinement (Optional)

For `max_rounds > 1`, the process iterates:

- **Round 1**: Generate initial bootstrapped examples with base model
- **Round 2+**: Use the improved model from previous round as teacher, potentially finding even better examples
- **Progressive improvement**: Each round can discover new successful patterns

## Using BootstrapFewShotWithRandomSearch in Practice

### Example 1: Text Classification Optimization

Let's start with a simple classification task:

```python
import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

# Define your classification task
TOPICS = ["sports", "politics", "technology"]
classifier = dspy.Predict(f"text -> topic:Literal{{{','.join(TOPICS)}}}")

# Define how to measure success
def accuracy_metric(example, prediction, trace=None) -> bool:
    return example.topic == prediction.topic

# Set up the optimizer
optimizer = BootstrapFewShotWithRandomSearch(
    metric=accuracy_metric,
    max_bootstrapped_demos=3,   # Up to 3 model-generated examples
    max_labeled_demos=3,        # Up to 3 real examples  
    num_candidate_programs=8,   # Try 8 different prompt variants
    max_rounds=1               # Single round of bootstrapping
)

# Optimize your classifier
optimized_classifier = optimizer.compile(
    classifier, 
    trainset=train_data,
    valset=validation_data
)
```

**What this does**:

- Takes your basic classifier and training data
- Generates up to 3 additional examples by having the model classify training texts correctly
- Creates 8 different prompt variants with different example combinations
- Tests each variant and returns the best-performing one

### Example 2: Math Problem Solving with Chain-of-Thought

For more complex reasoning tasks:

```python
from dspy import ChainOfThought, Module

# Configure your language model
dspy.configure(lm=dspy.LM(model='openai/gpt-4o', max_tokens=300))

# Define a chain-of-thought math solver
class MathSolver(Module):
    def __init__(self):
        self.solver = ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.solver(question=question)

# Define success metric for math problems
def math_accuracy(example, prediction, trace=None) -> bool:
    return str(example.answer).strip() == str(prediction.answer).strip()

# Set up more thorough optimization
optimizer = BootstrapFewShotWithRandomSearch(
    metric=math_accuracy,
    max_bootstrapped_demos=5,   # More examples for complex reasoning
    max_labeled_demos=3,        # Some real examples as foundation
    num_candidate_programs=12,  # Try more variants for better results
    max_rounds=2,              # Two rounds for iterative improvement
    num_threads=4              # Parallelize for speed
)

# Optimize the math solver
optimized_solver = optimizer.compile(
    MathSolver(),
    trainset=math_problems,
    valset=validation_problems
)
```

**What this does**:

- Uses chain-of-thought reasoning to solve math problems step-by-step
- Generates examples where the model shows correct reasoning patterns
- Tries 12 different combinations across 2 rounds of refinement
- Returns a solver with optimized demonstrations that improve reasoning

### Key Parameters Explained

- **`max_labeled_demos`**: Maximum real examples from your training data to include
- **`max_bootstrapped_demos`**: Maximum model-generated examples to create
- **`num_candidate_programs`**: How many different prompt variants to test (more = better results but higher cost)
- **`max_rounds`**: Number of iterative improvement rounds (1 is usually sufficient)
- **`num_threads`**: Parallel evaluation threads (higher = faster but more resource usage)
- **`metric`**: Function that determines what "success" means for your task

## What You Can Expect

### The Good News

**Significant Performance Improvements**: 

- Typical improvements range from 10-25% accuracy boost over unoptimized prompts
- Works especially well for reasoning tasks where step-by-step examples help
- Often discovers example combinations that perform better than intuitive manual choices

**Automated Discovery**: 

- Finds effective example combinations you might not think of manually
- Generates high-quality demonstrations by keeping only the model's successful attempts  
- Adapts to your specific task and data characteristics

**Practical Benefits**:

- **Time-saving**: Eliminates manual trial-and-error in example selection
- **Systematic**: Evaluates options objectively using your chosen metric
- **Scalable**: Can handle large datasets and complex reasoning tasks

### The Realistic Expectations

**Cost Considerations**:

- **Time**: Typically takes 10-30 minutes depending on settings and data size
- **API calls**: Makes many model calls during optimization (budget accordingly)
- **Compute**: Benefits from parallel processing when available

**Performance Factors**:

- **Works best with sufficient data**: Needs enough training examples to bootstrap from (ideally 20+ examples)
- **Depends on base model capability**: If your model can't solve training problems correctly, bootstrapping won't generate good examples
- **Quality varies by task**: More effective for tasks where examples significantly help (like reasoning, complex formatting)

**Not Magic**: 

- **Won't fix fundamental issues**: Can't overcome poor model choice or impossible tasks
- **Metric-dependent**: Only as good as your evaluation function
- **May overfit**: Can sometimes find examples too specific to validation data

## Strengths and Limitations

### Key Strengths

**Automatic Example Discovery**: 

- Eliminates the guesswork in selecting few-shot examples
- Uses the model's own successful outputs as teaching examples
- Systematically explores combinations you might miss manually

**Effective Search Strategy**:

- Random search is simple but surprisingly powerful for this problem
- Avoids local optima that greedy selection might get stuck in
- Embarrassingly parallel - can evaluate many candidates simultaneously

**Quality Assurance**:

- Only includes bootstrapped examples that pass your quality metric
- Validates final selection on held-out data to ensure generalization
- Prevents overfitting to specific training examples

**Flexibility**:

- Works with any DSPy module and task type
- Supports custom metrics for different quality measures
- Can be combined with different base models and reasoning strategies

### Key Limitations

**Computational Cost**:

- Requires many model evaluations during optimization
- Can be expensive for large models or extensive search
- Time scales with number of candidates and validation data size

**Bootstrap Dependency**:

- Effectiveness limited by base model's ability to solve training problems
- Very weak models may not generate useful bootstrapped examples
- Very strong models might not benefit much from few-shot examples

**Search Limitations**:

- Random search doesn't guarantee finding the global optimum
- May miss good combinations that require more sophisticated search
- No learning from previous trials to guide future searches

**Data Requirements**:

- Needs sufficient training data to bootstrap effectively
- Requires representative validation data for proper selection
- Quality depends on having a meaningful evaluation metric

## Best Practices and Tips

### Setting Up for Success

1. **Start with good training data**: Ensure your examples are representative and high-quality
2. **Choose meaningful metrics**: Your evaluation function should capture what you actually care about
3. **Begin conservatively**: Start with fewer candidates and rounds, then scale up if promising
4. **Monitor costs**: Keep track of API usage during optimization

### Common Pitfalls to Avoid

1. **Insufficient validation data**: Too small validation sets lead to unreliable optimization
2. **Poor metric design**: Metrics that don't reflect real performance goals mislead optimization
3. **Over-optimization**: Running too many rounds or candidates can lead to overfitting
4. **Ignoring base performance**: Not checking if optimization actually improved over baseline

## Comparison with Other Optimizers

### vs. Manual Example Selection

- **BootstrapFewShot**: Systematic, objective, discovers non-obvious combinations
- **Manual**: Intuitive but subjective, time-consuming, limited exploration

### vs. Simple BootstrapFewShot (without random search)

- **With Random Search**: Tests multiple combinations, more robust results
- **Without Random Search**: Single attempt, may get unlucky with initial selection

### vs. MIPROv2 or Bayesian Optimizers

- **BootstrapFewShot**: Simpler, more straightforward, good baseline performance
- **Advanced optimizers**: More sample-efficient, can optimize instructions too, but more complex

## When to Use BootstrapFewShotWithRandomSearch

### Great For:

- **Tasks where examples significantly help**: Complex reasoning, specific formatting, nuanced classification
- **When you have sufficient training data**: At least 20-50 examples to bootstrap from  
- **Systematic optimization needs**: When manual example selection is too time-consuming
- **Performance-critical applications**: Where the optimization cost is justified by improved results

### Consider Alternatives When:

- **Very limited data**: Fewer than 10-20 examples may not provide enough bootstrapping material
- **Simple tasks**: Basic classification or generation where examples don't help much
- **Tight resource constraints**: When optimization cost exceeds the value of improvement
- **Already high performance**: If your current approach achieves 95%+ on your metric

### Getting Started

1. **Prepare your data**: Ensure you have training and validation sets
2. **Define your metric**: Create a function that measures what success means for your task
3. **Start small**: Begin with `num_candidate_programs=5` and `max_rounds=1`
4. **Evaluate results**: Test the optimized program on held-out data
5. **Scale up if promising**: Increase parameters for potentially better results

BootstrapFewShotWithRandomSearch represents a powerful middle ground in prompt optimization - more sophisticated than manual selection, simpler than advanced Bayesian methods, and effective across a wide range of tasks. When you have good training data and clear success metrics, it can deliver substantial improvements with relatively straightforward setup and reasonable computational cost.
