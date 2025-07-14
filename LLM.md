# DSPy API Reference

**Version:** 3.0.0b2  
**Description:** DSPy is the framework for programming—rather than prompting—language models. Build modular AI systems and optimize their prompts and weights.

## Installation
```bash
pip install dspy
```

## Quick Start
```python
import dspy

# Configure language model
lm = dspy.LM("gpt-4o-mini")
dspy.configure(lm=lm)

# Create a simple predictor
predictor = dspy.Predict("question -> answer")
result = predictor(question="What is 2+2?")
print(result.answer)
```

## Core Imports
```python
# Main imports available from dspy
from dspy import (
    # Core primitives
    Module, Example, Prediction,
    
    # Prediction modules
    Predict, ChainOfThought, ReAct, Tool,
    
    # Signatures
    Signature, InputField, OutputField, make_signature,
    
    # Language models and adapters
    LM, Adapter, ChatAdapter, JSONAdapter, XMLAdapter,
    
    # Optimization
    BootstrapFewShot, MIPROv2, COPRO, GRPO,
    
    # Evaluation
    Evaluate,
    
    # Configuration
    configure, context, settings,
    
    # Utilities
    asyncify, syncify, streamify, load
)
```


## Core Primitives (`dspy.primitives`)

### Module
**Import:** `from dspy import Module`

Base class for all DSPy modules with automatic initialization and predictor management.

**Key Methods:**
- `__call__(*args, **kwargs)` - Execute the module
- `acall(*args, **kwargs)` - Async execution
- `named_predictors()` - Get all named predictors
- `predictors()` - Get all predictors
- `set_lm(lm)` - Set language model
- `batch(examples, num_threads=1)` - Parallel processing of examples
- `map_named_predictors(func)` - Apply function to all predictors

### Example
**Import:** `from dspy import Example`

Container for input/output data with dict-like interface.

**Key Methods:**
- `with_inputs(*keys)` - Create example with only specified inputs
- `inputs()` - Get input fields
- `labels()` - Get output/label fields
- `copy(**kwargs)` - Create copy with updates
- `without(*keys)` - Create copy excluding keys

### Prediction
**Import:** `from dspy import Prediction`

Output container supporting arithmetic operations on score fields.

**Key Methods:**
- `from_completions(list_or_dict, signature=None)` - Create from completions
- `get_lm_usage()` - Get language model usage stats
- Arithmetic operations: `+`, `/`, `<`, `>`, `<=`, `>=` (on score field)


## Prediction Modules (`dspy.predict`)

### Predict
**Import:** `from dspy import Predict`

Basic prediction module for any signature.

```python
predictor = Predict("question -> answer")
result = predictor(question="What is 2+2?")
```

**Key Methods:**
- `__init__(signature, callbacks=None, **config)`
- `forward(**kwargs)` - Synchronous prediction
- `aforward(**kwargs)` - Asynchronous prediction
- `reset()` - Reset internal state
- `dump_state()` / `load_state(state)` - State management

### ChainOfThought
**Import:** `from dspy import ChainOfThought`

Adds step-by-step reasoning to predictions.

```python
cot = ChainOfThought("question -> answer")
result = cot(question="Solve this math problem")
# result.reasoning contains the step-by-step reasoning
```

**Key Methods:**
- `__init__(signature, rationale_field=None, rationale_field_type=str, **config)`
- `forward(**kwargs)` - Returns prediction with reasoning field

### ReAct
**Import:** `from dspy import ReAct`

Reasoning and Acting paradigm for tool-using agents.

```python
def get_weather(city: str) -> str:
    return f"Weather in {city}: sunny"

react = ReAct("question -> answer", tools=[get_weather])
result = react(question="What's the weather in Tokyo?")
```

**Key Methods:**
- `__init__(signature, tools, max_iters=10)`
- `forward(**input_args)` - Execute ReAct loop
- `truncate_trajectory(trajectory)` - Handle context window limits

### Tool
**Import:** `from dspy import Tool`

Wrapper for functions to be used with ReAct.

```python
tool = Tool(get_weather)  # Automatically extracts name and signature
```


## Signature System (`dspy.signatures`)

### Signature
**Import:** `from dspy import Signature`

Defines input/output structure for modules.

```python
class QA(dspy.Signature):
    """Answer questions based on context."""
    question: str = InputField()
    context: str = InputField()
    answer: str = OutputField()
```

**Key Methods:**
- `with_instructions(instructions)` - Add/update instructions
- `with_updated_fields(name, type_=None, **kwargs)` - Update field properties
- `prepend(name, field, type_=None)` - Add field at beginning
- `append(name, field, type_=None)` - Add field at end
- `delete(name)` - Remove field
- `insert(index, name, field, type_=None)` - Insert field at position

**Properties:**
- `input_fields` - Dictionary of input fields
- `output_fields` - Dictionary of output fields
- `instructions` - Signature instructions
- `signature` - String representation (e.g., "question, context -> answer")

### Field Types
**Import:** `from dspy import InputField, OutputField`

```python
question = InputField(desc="The question to answer")
answer = OutputField(desc="The answer", prefix="Answer:")
```

### Utility Functions

#### make_signature
**Import:** `from dspy import make_signature`

Create signatures from strings or dictionaries.

```python
# String format
sig = make_signature("question, context -> answer")

# With instructions
sig = make_signature("question -> answer", "Answer the question briefly")

# With custom types
sig = make_signature("input: MyType -> output", custom_types={"MyType": MyType})
```

#### ensure_signature
**Import:** `from dspy import ensure_signature`

Convert string or signature to proper Signature class.

```python
sig = ensure_signature("question -> answer")  # Returns Signature class
```


## Language Model Clients (`dspy.clients`)

### LM
**Import:** `from dspy import LM`

Main language model client supporting chat/text completion requests.

```python
lm = LM("gpt-4o-mini", temperature=0.1, max_tokens=2000)
dspy.configure(lm=lm)
```

**Constructor:**
```python
LM(model: str, model_type: Literal["chat", "text"] = "chat", 
   temperature: float = 0.0, max_tokens: int = 4000, 
   cache: bool = True, cache_in_memory: bool = True,
   callbacks: list[BaseCallback] | None = None, num_retries: int = 3,
   provider: Provider | None = None)
```

**Key Methods:**
- `forward(prompt=None, messages=None, **kwargs)` - Synchronous completion
- `aforward(prompt=None, messages=None, **kwargs)` - Async completion  
- `finetune(train_kwargs)` - Start finetuning job
- `reinforce(train_kwargs)` - Start reinforcement learning job

### BaseLM
**Import:** `from dspy import BaseLM`

Base class for custom LLM implementations. Override `forward()` method.

### Embedder
**Import:** `from dspy import Embedder`

Unified interface for text embeddings from hosted/local models.


## Adapters (`dspy.adapters`)

### Adapter
**Import:** `from dspy import Adapter`

Base adapter class for format conversion between DSPy signatures and LLM APIs.

**Constructor:**
```python
Adapter(callbacks: list[BaseCallback] | None = None, 
        use_native_function_calling: bool = False)
```

**Key Methods:**
- `format(signature, demos, inputs)` - Format input for LLM
- `parse(signature, completion)` - Parse LLM response
- `__call__(lm, lm_kwargs, signature, demos, inputs)` - Main adapter logic

### ChatAdapter
**Import:** `from dspy import ChatAdapter`

Formats interactions as chat messages with field markers `[[ ## field_name ## ]]`.

### JSONAdapter
**Import:** `from dspy import JSONAdapter`

Extends ChatAdapter with JSON schema validation and native function calling.

```python
adapter = JSONAdapter(use_native_function_calling=True)
dspy.configure(adapter=adapter)
```

### XMLAdapter
**Import:** `from dspy import XMLAdapter`

Uses XML tags for field formatting: `<field_name>content</field_name>`.

### TwoStepAdapter
**Import:** `from dspy import TwoStepAdapter`

Two-phase processing for reasoning models - separate reasoning and output steps.


## Optimization (`dspy.teleprompt`)

### BootstrapFewShot
**Import:** `from dspy import BootstrapFewShot`

Composes demos/examples from labeled training data and bootstrapped examples.

```python
optimizer = BootstrapFewShot(metric=my_metric, max_bootstrapped_demos=8)
optimized_program = optimizer.compile(student=program, trainset=train_data)
```

**Key Methods:**
- `__init__(metric=None, metric_threshold=None, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1)`
- `compile(student, teacher=None, trainset)` - Optimizes student program using teacher guidance

### MIPROv2
**Import:** `from dspy import MIPROv2`

Multi-stage Instruction Proposal and Refinement Optimizer v2 for advanced prompt optimization.

```python
optimizer = MIPROv2(metric=my_metric, auto="medium")
optimized_program = optimizer.compile(student=program, trainset=train_data)
```

**Key Methods:**
- `__init__(metric, auto="light"|"medium"|"heavy", num_candidates=None, init_temperature=1.0)`
- `compile(student, trainset, valset=None, num_trials=None)`

### COPRO
**Import:** `from dspy import COPRO`

Coordinate Ascent Prompt Optimization for systematic instruction improvement.

**Key Methods:**
- `__init__(metric, breadth=10, depth=3, init_temperature=1.4, track_stats=True)`
- `compile(program, trainset, eval_kwargs=None)`

### GRPO
**Import:** `from dspy import GRPO`

Generalized Reward-based Prompt Optimization for fine-tuning with reinforcement learning.

**Key Methods:**
- `__init__(metric, multitask=True, train_kwargs=None)`
- `compile(student, trainset, valset=None)`

### Other Optimizers
- `BootstrapFewShotWithRandomSearch` - Bootstrap with random search over hyperparameters
- `Ensemble` - Creates ensemble of multiple optimized programs
- `KNNFewShot` - K-nearest neighbors based few-shot example selection


## Evaluation (`dspy.evaluate`)

### Evaluate
**Import:** `from dspy import Evaluate`

Main evaluation class for DSPy programs with parallel processing support.

```python
evaluator = Evaluate(devset=dev_data, metric=my_metric, num_threads=4)
result = evaluator(optimized_program)
print(f"Score: {result.score}")
```

**Key Methods:**
- `__init__(devset, metric=None, num_threads=None, display_progress=False, display_table=False)`
- `__call__(program, metric=None, devset=None, num_threads=None)` - Returns `EvaluationResult`

### EvaluationResult
Container for evaluation results with:
- `score: float` - Overall performance percentage
- `results: list` - List of (example, prediction, score) tuples

### Metrics
- `EM` - Exact Match with normalization
- `answer_exact_match` / `answer_passage_match` - Specialized QA matching
- `SemanticF1` - LLM-based semantic F1 scoring
- `CompleteAndGrounded` - Multi-dimensional response quality assessment


## Configuration (`dspy.settings`)

### configure
**Import:** `from dspy import configure`

Set global configuration (one thread only).

```python
dspy.configure(lm=lm, adapter=adapter, cache=True)
```

### context
**Import:** `from dspy import context`

Thread-local configuration overrides.

```python
with dspy.context(temperature=0.5):
    # Higher temperature for this block
    pass
```

### settings
**Import:** `from dspy import settings`

Thread-safe global configuration manager.

**Key Methods:**
- `get(key, default=None)` - Retrieve configuration value

**Common Configuration:**
- `lm` - Default language model
- `adapter` - Default adapter instance
- `cache` - Caching settings
- `callbacks` - Global callbacks


## Utilities

### Async/Sync Utilities
- `asyncify` - Convert sync functions to async
- `syncify` - Convert async functions to sync
- `streamify` - Add streaming support to modules

### Data Management
- `load` - Load saved DSPy programs/data
- `track_usage` - Track language model usage

### Logging
- `configure_dspy_loggers` - Configure DSPy logging
- `enable_logging` / `disable_logging` - Control logging

### Caching
- `DSPY_CACHE` - Global cache instance
- `cache` - Alias for DSPY_CACHE


## Complete Usage Examples

### Basic Module Creation
```python
import dspy

class QAModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("question, context -> answer")
    
    def forward(self, question, context):
        return self.predictor(question=question, context=context)

# Usage
lm = dspy.LM("gpt-4o-mini")
dspy.configure(lm=lm)

qa = QAModule()
result = qa(question="What is DSPy?", context="DSPy is a framework...")
```

### Chain of Thought with Custom Signature
```python
class DetailedQA(dspy.Signature):
    """Provide detailed answers with reasoning."""
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning")
    answer: str = dspy.OutputField()

cot = dspy.ChainOfThought(DetailedQA)
result = cot(question="Complex question", context="Relevant context")
print(result.reasoning)
print(result.answer)
```

### Tool-Using Agent
```python
def search_web(query: str) -> str:
    return f"Search results for: {query}"

def calculate(expression: str) -> str:
    return str(eval(expression))  # Use safely in practice

agent = dspy.ReAct(
    "question -> answer",
    tools=[search_web, calculate],
    max_iters=5
)

result = agent(question="What is 15 * 23 and find recent news about it?")
```

### Optimization Pipeline
```python
# Define metric
def accuracy_metric(example, prediction, trace=None):
    return example.answer.lower() == prediction.answer.lower()

# Create and optimize program
program = QAModule()
optimizer = dspy.BootstrapFewShot(metric=accuracy_metric)
optimized_program = optimizer.compile(student=program, trainset=train_data)

# Evaluate
evaluator = dspy.Evaluate(devset=dev_data, metric=accuracy_metric)
result = evaluator(optimized_program)
print(f"Accuracy: {result.score}%")
```

### Advanced Configuration
```python
# Custom adapter with JSON output
adapter = dspy.JSONAdapter(use_native_function_calling=True)
lm = dspy.LM("gpt-4o", temperature=0.1, max_tokens=2000)

dspy.configure(lm=lm, adapter=adapter)

# Context-specific overrides
with dspy.context(temperature=0.8, max_tokens=1000):
    creative_result = predictor(question="Write a creative story")

# Async usage
async def async_prediction():
    result = await predictor.aforward(question="Async question")
    return result
```

