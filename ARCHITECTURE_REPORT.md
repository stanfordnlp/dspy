# DSPy Library Architecture & Codebase Exploration

## Executive Summary

DSPy is a framework for **programming language models** rather than prompting them. It provides a structured, modular approach to building and optimizing AI systems. The library follows a declarative programming model where you compose modules and let DSPy handle prompt optimization, few-shot learning, and instruction generation.

**Version**: 3.0.4 (beta)
**Python**: 3.10-3.14
**Key Dependencies**: Pydantic 2.0+, LiteLLM, Optuna, OpenAI, gepa[dspy]

---

## 1. OVERALL DIRECTORY STRUCTURE

```
dspy/
├── __init__.py              # Main package entry point (30 lines - exports all public APIs)
├── __metadata__.py          # Version and metadata
├── primitives/              # Core building blocks
│   ├── base_module.py       # BaseModule - save/load, parameter tracking
│   ├── module.py            # Module (with metaclass) - basic executable unit
│   ├── example.py           # Example - flexible data container for training
│   └── prediction.py        # Prediction - output with scoring capabilities
├── signatures/              # Task specification system
│   ├── signature.py         # Signature class - defines input/output fields
│   ├── field.py             # InputField, OutputField definitions
│   └── utils.py             # Signature utilities
├── predict/                 # Prediction modules
│   ├── predict.py           # Predict - core LM calling module
│   ├── parameter.py         # Parameter - marker interface for optimizable components
│   ├── chain_of_thought.py  # ChainOfThought - reasoning module
│   ├── parallel.py          # Parallel - multi-threaded execution
│   ├── best_of_n.py         # BestOfN - selection strategy
│   ├── retry.py             # Retry - failure recovery
│   ├── refine.py            # Refine - iterative improvement
│   ├── react.py             # ReAct - reasoning + acting with tools
│   ├── code_act.py          # CodeAct - code execution module
│   ├── program_of_thought.py # ProgramOfThought - symbolic reasoning
│   ├── knn.py               # KNN - few-shot selection
│   ├── multi_chain_comparison.py
│   └── aggregation.py       # Majority voting strategies
├── clients/                 # Language Model interfaces
│   ├── base_lm.py           # BaseLM - abstract LM interface
│   ├── lm.py                # LM - main implementation (LiteLLM integration)
│   ├── lm_local.py          # Local model support
│   ├── cache.py             # Response caching layer
│   ├── provider.py          # Provider abstractions
│   ├── embedding.py         # Embedding clients
│   ├── openai.py            # OpenAI-specific implementations
│   ├── databricks.py        # Databricks integration
│   └── __init__.py          # Exports and DSPY_CACHE
├── adapters/                # LM Output Formatting
│   ├── base.py              # Adapter - base adapter class
│   ├── chat_adapter.py      # ChatAdapter - chat format handling
│   ├── json_adapter.py      # JSONAdapter - JSON parsing
│   ├── xml_adapter.py       # XMLAdapter - XML parsing
│   ├── two_step_adapter.py  # TwoStepAdapter - multi-step format
│   ├── baml_adapter.py      # BAMLAdapter - BAML format
│   ├── utils.py             # Adapter utilities
│   ├── types/               # Type definitions for adapters
│   │   ├── base_type.py
│   │   ├── tool.py          # Tool & ToolCalls
│   │   ├── history.py       # Conversation history
│   │   └── __init__.py
│   └── __init__.py
├── teleprompt/              # Optimization Algorithms (Teleprompters)
│   ├── teleprompt.py        # Teleprompter base class
│   ├── bootstrap.py         # BootstrapFewShot - few-shot demo selection
│   ├── copro_optimizer.py   # CoprOptimizer - prompt optimization
│   ├── mipro_optimizer_v2.py# MipROv2 - modern prompt optimizer
│   ├── simba.py             # SIMBA - instruction optimization
│   ├── ensemble.py          # Ensemble methods
│   ├── random_search.py     # Random search baseline
│   ├── vanilla.py           # Vanilla few-shot
│   ├── bootstrap_finetune.py# Fine-tuning integration
│   ├── avatar_optimizer.py  # Avatar optimizer
│   ├── grpo.py              # GRPO - group reasoning
│   ├── knn_fewshot.py       # KNN-based few-shot
│   ├── signature_opt.py     # Signature optimization
│   ├── infer_rules.py       # Rule inference
│   ├── bettertogether.py    # Multi-optimizer ensemble
│   ├── gepa/                # GEPA optimizer (external dependency)
│   ├── utils.py             # Teleprompter utilities
│   └── __init__.py
├── evaluate/                # Evaluation Framework
│   ├── evaluate.py          # Evaluate - core evaluation class
│   ├── metrics.py           # Built-in metrics
│   ├── auto_evaluation.py   # Automatic evaluation
│   └── __init__.py
├── retrievers/              # Information Retrieval
│   ├── retrieve.py          # Retrieve - base retriever module
│   ├── embeddings.py        # Embedding-based retrieval
│   ├── databricks_rm.py     # Databricks retrieval
│   ├── weaviate_rm.py       # Weaviate integration
│   └── __init__.py
├── datasets/                # Data Loading
│   ├── dataset.py           # Dataset base class
│   ├── dataloader.py        # DataLoader utilities
│   ├── hotpotqa.py          # HotpotQA dataset
│   ├── gsm8k.py             # GSM8K math dataset
│   ├── math.py              # Math datasets
│   ├── colors.py            # Color reasoning task
│   ├── alfworld/            # AlfWorld simulator
│   └── __init__.py
├── propose/                 # Instruction Generation
│   ├── propose_base.py      # Proposer - base for instruction generation
│   ├── grounded_proposer.py # Grounded instruction proposer
│   ├── dataset_summary_generator.py
│   ├── utils.py
│   └── __init__.py
├── retrievers/              # Retrieval Systems
│   └── (see above)
├── dsp/                     # Legacy DSP Support
│   ├── utils/
│   │   ├── settings.py      # Global configuration (thread-safe)
│   │   ├── utils.py         # Utility functions (dotdict, etc.)
│   │   └── dpr.py           # Dense passage retrieval
│   ├── colbertv2.py         # ColBERT retriever
│   └── __init__.py          # Empty, legacy module
├── utils/                   # Utility Functions
│   ├── saving.py            # Save/load programs
│   ├── magicattr.py         # Magic attribute access
│   ├── callback.py          # Callback system
│   ├── caching.py           # Caching utilities
│   ├── asyncify.py          # Async conversion utilities
│   ├── syncify.py           # Sync conversion utilities
│   ├── unbatchify.py        # Batch processing utilities
│   ├── annotation.py        # Field annotations
│   ├── exceptions.py        # Custom exceptions
│   ├── hasher.py            # Hashing utilities
│   ├── inspect_history.py   # History inspection
│   ├── logging_utils.py     # Logging configuration
│   ├── parallelizer.py      # Parallel execution
│   ├── usage_tracker.py     # Token usage tracking
│   ├── mcp.py               # MCP protocol support
│   ├── dummies.py           # Dummy implementations
│   ├── langchain_tool.py    # LangChain integration
│   └── __init__.py
├── streaming/               # Streaming Outputs
│   ├── streamify.py         # Streamify - enable streaming
│   ├── streaming_listener.py# Listener pattern
│   ├── messages.py          # Message handling
│   └── __init__.py
├── experimental/            # Experimental Features
│   └── __init__.py          # Citations and experimental APIs
└── primitives/              # (see above)

```

---

## 2. CORE COMPONENTS & THEIR PURPOSES

### A. PRIMITIVES - Foundation Layer

**Location**: `dspy/primitives/`

#### Module Class (`module.py`)
```
Purpose: Base class for all DSPy programs/components
Key Features:
  - Metaclass-based initialization (ProgramMeta)
  - Manages callbacks and execution history
  - Supports batch processing
  - LM usage tracking
  - Named parameter access
  - Module composition
  
Key Methods:
  __call__()        - Execute module (with callback support)
  acall()           - Async execution
  forward()         - Main logic (overridden by subclasses)
  batch()           - Process multiple examples in parallel
  named_predictors() - Find all Predict modules in tree
  set_lm()          - Configure language model
  inspect_history() - Debug execution traces
```

#### BaseModule Class (`base_module.py`)
```
Purpose: Core functionality for save/load and parameter management
Key Features:
  - Deep copying with smart parameter handling
  - Parameter tracking via named_parameters()
  - Sub-module discovery
  - State serialization (JSON/pickle)
  - Dependency version tracking
  
Key Methods:
  named_parameters()  - Recursively find all Parameter instances
  named_sub_modules() - Find all sub-modules
  save()             - Save state or full program
  load()             - Load state into module
  dump_state()       - Serialize parameters
  load_state()       - Deserialize parameters
```

#### Example Class (`example.py`)
```
Purpose: Flexible data container for training/eval examples
Key Features:
  - Dictionary-like access (item, attribute access)
  - Input/output field separation (with_inputs())
  - Immutable operations (copy(), without())
  - Serialization support
  
Key Methods:
  inputs()  - Get only input fields
  labels()  - Get only output fields
  copy()    - Create modified copy
  toDict()  - Serialize to dict
```

#### Prediction Class (`prediction.py`)
```
Purpose: Output container with scoring capabilities
Inherits: Example
Key Features:
  - Score-based comparisons (<, >, <=, >=)
  - Multiple completions support
  - LM usage tracking
  - Arithmetic operations on scores
```

### B. SIGNATURES - Task Specification

**Location**: `dspy/signatures/`

#### Signature Class (`signature.py`)
```
Purpose: Declarative specification of input/output fields
Based On: Pydantic BaseModel (automatic validation)
Key Features:
  - Input/output field separation via InputField/OutputField
  - String-based signature creation ("input1, input2 -> output1")
  - Type annotations with custom type support
  - Field prefix/description inference
  - Instructions (docstring = instructions)
  - Signature manipulation (prepend, append, insert, delete)
  
Key Methods:
  with_instructions()   - Create variant with new instructions
  with_updated_fields() - Modify field metadata
  prepend/append/insert - Add fields at specific positions
  delete()             - Remove field
  equals()             - Compare two signatures
  dump_state()         - Serialize signature
  load_state()         - Deserialize signature
```

**Signature Metaclass (`SignatureMeta`)**:
- Intercepts Signature() calls to generate new signature types
- Validates all fields are InputField or OutputField
- Generates default instructions from field names
- Handles custom type resolution via frame introspection

#### Field Classes (`field.py`)
```
InputField   - Marks input parameters
OutputField  - Marks output/response fields

Both use Pydantic Field() under the hood with:
  - desc: Description for LM
  - prefix: Label for formatting
  - annotation: Type hint
```

### C. PREDICT - Core LM Interaction

**Location**: `dspy/predict/`

#### Predict Class (`predict.py`)
```
Purpose: Core module for calling language models
Signature: signature (input/output spec) -> predictions
Key Features:
  - LM-agnostic (works with any BaseLM)
  - Adapter-based output parsing
  - Few-shot demo management
  - Temperature/config per-call
  - Callback integration
  - State save/load (traces, demos, signature)
  
Key Attributes:
  signature - Input/output specification
  lm        - Language model instance
  demos     - Few-shot examples
  traces    - Execution traces for optimization
  config    - LM parameters (temperature, max_tokens, etc.)
  
Key Methods:
  __call__(**kwargs) -> Prediction
  forward()  - Main execution logic
  dump_state() - Save signature, demos, config
  load_state() - Restore from saved state
```

#### Parameter Class (`parameter.py`)
```
Purpose: Marker interface for optimizable components
Used By: Teleprompters to find what can be optimized
Actual Implementation: Just a pass class (marker pattern)
```

#### Important Predict Variants:
- **ChainOfThought**: Prepends reasoning field to signature
- **Parallel**: Execute multiple modules in threads
- **BestOfN**: Try N times, pick best via metric
- **Retry**: Retry on failure with backoff
- **Refine**: Iterative refinement loop
- **ReAct**: Reasoning + Acting with tool use
- **ProgramOfThought**: Symbolic reasoning
- **KNN**: Few-shot demo selection via similarity

### D. CLIENTS - Language Models

**Location**: `dspy/clients/`

#### BaseLM Class (`base_lm.py`)
```
Purpose: Abstract interface for LM providers
Key Features:
  - Unified response processing
  - History tracking
  - Cost calculation
  - Response parsing
  
Key Methods:
  forward()  - Main LM call (to be implemented)
  _process_lm_response() - Standardize response format
```

#### LM Class (`lm.py`)
```
Purpose: Main LM implementation using LiteLLM
Features:
  - Multi-provider support (OpenAI, Claude, Cohere, etc.)
  - Chat and text completion modes
  - Request caching
  - Retry with exponential backoff
  - Fine-tuning support
  - Token usage tracking
  - Reasoning model support (o1, gpt-5)
  
Provider Integration:
  Uses LiteLLM under hood for provider abstraction
```

#### Cache Layer (`cache.py`)
```
Purpose: Transparent request caching
Features:
  - Disk-based caching (diskcache)
  - Cache bypass with rollout_id
  - Temperature-aware caching
```

### E. ADAPTERS - Output Formatting

**Location**: `dspy/adapters/`

#### Adapter Base Class (`base.py`)
```
Purpose: Transform LM outputs to structured Predictions
Key Flow: LM prompt -> format() -> LM response -> parse()
Key Features:
  - Custom type handling (Tool, ToolCalls, Citations)
  - Native function calling support
  - Conversation history management
  - Few-shot example formatting
  - Callback integration
  
Key Methods:
  format(signature, inputs, demos) -> prompt_string
  parse(signature, response) -> dict
```

#### Adapter Implementations:
- **ChatAdapter**: For chat-based LMs
- **JSONAdapter**: Parses JSON responses
- **XMLAdapter**: Parses XML responses  
- **TwoStepAdapter**: Multi-stage output generation
- **BAMLAdapter**: BAML format parsing

### F. TELEPROMPT - Optimizers

**Location**: `dspy/teleprompt/`

#### Teleprompter Base (`teleprompt.py`)
```
Purpose: Base class for all optimizers
Key Method:
  compile(student, trainset, teacher=None, valset=None) -> optimized_student
  
The Pattern:
  Takes unoptimized program + training data
  Returns optimized version with better prompts/demos
```

#### Key Optimizers:
1. **BootstrapFewShot**: 
   - Selects best examples from training set as demos
   - Validates with metric function
   
2. **CopROOptimizer**:
   - Optimizes prompts using contrastive learning
   
3. **MIPROv2**:
   - Modern multi-stage prompt optimizer
   - Uses LLM to suggest instruction improvements
   
4. **SIMBA**:
   - Optimizes field descriptions
   
5. **GRPO**:
   - Group-wise reasoning optimization
   
6. **BootstrapFinetune**:
   - Prepares data for model fine-tuning

### G. EVALUATE - Evaluation Framework

**Location**: `dspy/evaluate/`

#### Evaluate Class (`evaluate.py`)
```
Purpose: Test program performance on dataset
Key Features:
  - Parallel evaluation via ThreadPoolExecutor
  - Metric function support
  - Result tracking and display
  - CSV/JSON export
  - Error handling
  
Usage:
  evaluator = dspy.Evaluate(devset=examples, metric=my_metric)
  result = evaluator(program)
```

### H. SETTINGS & CONFIGURATION

**Location**: `dspy/dsp/utils/settings.py`

#### Settings Singleton (`Settings` class)
```
Purpose: Thread-safe global configuration
Key Features:
  - Single owner thread for dspy.configure()
  - Context managers for thread-local overrides
  - Configuration visibility across threads
  - Async-safe via contextvars
  
Key Attributes:
  lm              - Global language model
  adapter         - Global adapter
  rm              - Global retriever/reranker
  callbacks       - Global callbacks
  track_usage     - Token usage tracking
  async_max_workers - Async concurrency
  
Key Methods:
  configure()     - Set global config (single owner)
  context()       - Thread-local override context
```

---

## 3. ARCHITECTURE PATTERNS & DESIGN

### A. Core Design Principles

```
1. COMPOSITION OVER INHERITANCE
   - Modules contain Predict instances, not inherit from them
   - Adapters are pluggable, not subclass-specific
   
2. DECLARATION OVER CONFIGURATION
   - Signatures declare I/O structure declaratively
   - Instructions are docstrings on signatures
   - No complex config files needed
   
3. OPTIMIZATION AS COMPILATION
   - Teleprompters "compile" unoptimized programs
   - Output: optimized module with same interface
   - No rewrite of program code needed
   
4. SEPARATION OF CONCERNS
   - Modules: business logic
   - Signatures: task specification  
   - Adapters: format handling
   - Clients: LM backend
   - Optimizers: automatic improvement
```

### B. Key Architectural Patterns

#### 1. Module Pattern
```
Every DSPy program is a Module that:
  - Can contain other Modules
  - Has a forward() method (main logic)
  - Tracks parameters automatically
  - Can be saved/loaded
  - Has execution history
  
Example:
  class RAG(dspy.Module):
      def __init__(self):
          self.retrieve = dspy.Retrieve(k=3)
          self.answer = dspy.ChainOfThought("context -> answer")
      
      def forward(self, question):
          context = self.retrieve(question).passages
          return self.answer(question=question, context=context)
```

#### 2. Signature Pattern
```
Signatures specify "contracts" for LM calls:
  - All inputs must be provided
  - All outputs will be generated
  - Instructions guide the LM
  - Type hints enable validation
  
Example:
  class QA(dspy.Signature):
      """Answer questions about the topic."""
      question: str = dspy.InputField()
      answer: str = dspy.OutputField()
```

#### 3. Parameter & Optimization Pattern
```
Optimize what the LM sees:
  1. Identify "parameters" (via Parameter marker)
  2. Generate variations
  3. Evaluate against metric
  4. Keep best version
  
Optimizable Elements:
  - Instruction text (signature.__doc__)
  - Field descriptions
  - Few-shot examples (demos)
  - Field order/structure
```

#### 4. Adapter Pattern
```
Transform between program and LM interfaces:
  format(): Program inputs -> LM prompt
  parse(): LM response -> Program outputs
  
Allows:
  - Same program with different output formats
  - Type-safe parsing
  - Native LM features (function calling, citations)
```

#### 5. Thread-Safe Settings Pattern
```
Global state without race conditions:
  - One owner thread for configuration
  - Other threads see configured values
  - Context managers for thread-local overrides
  - Async-safe via contextvars
```

### C. Data Flow Diagram

```
Input Data
    ↓
Example (input fields only)
    ↓
dspy.Signature (declares transformation)
    ↓
dspy.Predict.forward()
    │
    ├─→ Adapter.format()  (apply instructions, demos)
    │
    ├─→ LM.forward()      (call model)
    │
    └─→ Adapter.parse()   (extract outputs)
    ↓
Prediction (output fields populated)
    ↓
Metric function (e.g., for evaluation)
    ↓
Score / Feedback
    ↓
Teleprompter (optimizer)
    └─→ Modifies signature/demos → better program
```

---

## 4. KEY ENTRY POINTS & EXECUTION FLOW

### A. Typical Usage Pattern

```python
# 1. Define task via Signature
class MyTask(dspy.Signature):
    input_text: str = dspy.InputField()
    output_text: str = dspy.OutputField()

# 2. Create Module
class MyProgram(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(MyTask)
    
    def forward(self, input_text):
        return self.predict(input_text=input_text)

# 3. Configure LM
dspy.configure(lm=dspy.LM("openai/gpt-4o"))

# 4. Create instance
program = MyProgram()

# 5. Execute
result = program(input_text="...")

# 6. Evaluate
evaluator = dspy.Evaluate(devset=examples, metric=my_metric)
score = evaluator(program)

# 7. Optimize
optimizer = dspy.BootstrapFewShot(metric=my_metric)
optimized = optimizer.compile(program, trainset=train_examples)

# 8. Use optimized program
better_result = optimized(input_text="...")
```

### B. Module Execution Flow

When you call a module:

```
module(input1=x, input2=y)
    ↓
Module.__call__ (defined in module.py)
    ├─→ Register module in call stack
    ├─→ Track token usage (if enabled)
    ├─→ Execute callbacks (before)
    ├─→ Call forward(input1=x, input2=y)
    ├─→ Execute callbacks (after)
    ├─→ Track LM usage on Prediction
    └─→ Return Prediction object
```

### C. Predict Execution Flow

```
predict(input1=x, input2=y)
    ↓
Predict.__call__
    ├─→ Validate inputs against signature.input_fields
    ├─→ Get demos (few-shot examples) from self.demos
    ├─→ Get current LM from dspy.settings.lm
    ├─→ Get current Adapter from dspy.settings.adapter
    ├─→ Call Adapter.format(signature, inputs, demos)
    │   └─→ Create prompt with instructions + examples
    ├─→ Call LM.forward(prompt)
    │   └─→ Call provider API (OpenAI, etc.)
    │   └─→ Cache result if enabled
    │   └─→ Track usage
    ├─→ Call Adapter.parse(signature, response)
    │   └─→ Extract fields from LM response
    └─→ Return Prediction(output_field=value, ...)
```

### D. Optimization Flow

```
optimizer.compile(student, trainset, teacher)
    ↓
Create copies:
  - student_copy (for modifying)
  - teacher (if not provided, copy of student)
    ↓
Run training rounds:
    For each example in trainset:
        ├─→ Execute teacher (with temperature=1.0 for variance)
        ├─→ Evaluate output with metric function
        ├─→ If metric passes:
        │   └─→ Collect as bootstrap demo
        └─→ Accumulate demos for each Predict module
    ↓
Install demos:
    For each Predict in student_copy:
        ├─→ student_copy.predict.demos = collected_demos
        └─→ Mark as _compiled=True
    ↓
Return optimized student_copy
```

---

## 5. MODULE ORGANIZATION & EXPORTS

### Public API (via `dspy/__init__.py`)

```python
# Core classes
from dspy.predict import *              # Predict, ChainOfThought, etc.
from dspy.primitives import *           # Module, Example, Prediction
from dspy.retrievers import *           # Retrieve
from dspy.signatures import *           # Signature, InputField, OutputField
from dspy.teleprompt import *           # All optimizers

# Utilities
from dspy.evaluate import Evaluate      # Evaluation framework
from dspy.clients import *              # LM, BaseLM
from dspy.adapters import (             # Adapter, ChatAdapter, etc.
    Adapter, ChatAdapter, JSONAdapter,
    XMLAdapter, TwoStepAdapter
)
from dspy.utils.logging_utils import *  # Logging configuration
from dspy.utils.asyncify import asyncify
from dspy.utils.syncify import syncify
from dspy.utils.saving import load      # Load saved programs
from dspy.streaming.streamify import streamify

# Settings
from dspy.dsp.utils.settings import settings
configure = settings.configure          # Main config function
context = settings.context              # Context manager

# Caching
from dspy.clients import DSPY_CACHE
cache = DSPY_CACHE
```

### Key Classes by Category

```
CORE ABSTRACTIONS:
  - dspy.Module              (all programs)
  - dspy.Signature           (task spec)
  - dspy.Predict             (LM calling)
  - dspy.Example             (data)
  - dspy.Prediction          (output)

PREDICTORS:
  - dspy.Predict             (basic)
  - dspy.ChainOfThought      (reasoning)
  - dspy.ReAct               (reasoning + tools)
  - dspy.ProgramOfThought    (symbolic)
  - dspy.CodeAct             (code execution)
  - dspy.Refine              (iterative)
  - dspy.BestOfN             (selection)
  - dspy.Parallel            (concurrent)
  - dspy.KNN                 (few-shot selection)

OPTIMIZERS (Teleprompters):
  - dspy.BootstrapFewShot    (demo selection)
  - dspy.BootstrapFinetune   (prep for fine-tuning)
  - dspy.CopROOptimizer      (prompt optimization)
  - dspy.MIPROv2             (multi-stage optimizer)
  - dspy.SIMBA               (field optimization)
  - dspy.Ensemble            (combine models)
  - dspy.RandomSearch        (baseline)

LANGUAGE MODELS:
  - dspy.LM                  (main, LiteLLM-based)
  - dspy.BaseLM              (abstract)

ADAPTERS:
  - dspy.Adapter             (base)
  - dspy.ChatAdapter         (chat format)
  - dspy.JSONAdapter         (JSON parsing)
  - dspy.XMLAdapter          (XML parsing)
  - dspy.TwoStepAdapter      (multi-stage)

RETRIEVAL:
  - dspy.Retrieve            (retrieval module)
  - Various retriever implementations

EVALUATION:
  - dspy.Evaluate            (framework)

UTILITIES:
  - dspy.asyncify()          (sync -> async)
  - dspy.syncify()           (async -> sync)
  - dspy.streamify()         (enable streaming)
  - dspy.load()              (load saved program)
  - dspy.configure()         (global settings)
```

---

## 6. IMPORTANT DESIGN PATTERNS USED

### A. Metaclass Pattern
- **ProgramMeta**: Ensures all Module instances have proper initialization
- **SignatureMeta**: Intercepts Signature() calls to dynamically create signature classes

### B. Marker Pattern
- **Parameter**: Empty class that marks optimizable components
- Used to identify what Teleprompters can optimize

### C. Singleton Pattern
- **Settings**: Thread-safe singleton for global configuration

### D. Adapter Pattern
- **Adapter hierarchy**: ChatAdapter -> JSONAdapter, XMLAdapter, etc.
- Allows pluggable output formatting

### E. Template Method Pattern
- **Teleprompter.compile()**: Abstract method each optimizer implements
- **Module.forward()**: Subclasses implement their logic

### F. Strategy Pattern
- **LM implementations**: Different providers implement same interface
- **Adapters**: Different output formats, same format/parse interface

### G. Callback/Observer Pattern
- **Callbacks**: Can inject custom logic before/after key operations
- Used in Predict, Adapter, Module execution

### H. Factory Pattern
- **Signature()**: Creates new signature classes from strings or dicts
- **make_signature()**: Factory function for signatures

### I. Context Manager Pattern
- **Settings.context()**: Temporarily override global settings
- **Usage tracking**: with track_usage() context managers

---

## 7. THREADING & ASYNC SUPPORT

### Threading Safety

```
Global State (thread-safe):
  - Settings (owned by single thread)
  - LM cache (uses locks)
  - Module.history (can accumulate across threads)

Thread Usage:
  - Parallel module runs multiple threads
  - Evaluate runs evaluation in threads
  - ParallelExecutor manages thread pool
  - Context overrides propagate to spawned threads
```

### Async Support

```
Async Methods:
  - Module.acall() - async execution
  - Predict.aforward() - async LM calls (if supported)

Async Utilities:
  - asyncify() - convert sync module to async
  - syncify() - convert async to sync

Implementation:
  - Uses asyncer for sync/async bridging
  - Supports concurrent LM calls
  - Context-var based configuration propagation
```

---

## 8. CONFIGURATION & SETUP PATTERNS

### Global Configuration

```python
# Configure language model
dspy.configure(lm=dspy.LM("openai/gpt-4o"))

# Configure retriever/reranker
dspy.configure(rm=my_retriever)

# Configure adapter
dspy.configure(adapter=dspy.JSONAdapter())

# Configure callbacks
dspy.configure(callbacks=[my_callback])

# Enable features
dspy.configure(track_usage=True)
dspy.configure(disable_history=False)
```

### Per-Module Configuration

```python
# Override per module
module.set_lm(different_lm)

# Per-call config
predict(input1=x, config={
    "temperature": 0.7,
    "max_tokens": 500
})
```

### Context-Based Configuration

```python
# Thread-local overrides
with dspy.context(lm=test_lm):
    result = program(...)  # Uses test_lm
    # Other threads unaffected
```

---

## 9. EXTENSION POINTS FOR CUSTOM IMPLEMENTATIONS

### A. Custom Language Models

```python
class MyLM(dspy.BaseLM):
    def forward(self, prompt, messages=None, **kwargs):
        # Your implementation
        response = my_api_call(...)
        return response  # Must match OpenAI format
```

### B. Custom Modules

```python
class MyProgram(dspy.Module):
    def __init__(self):
        self.predict1 = dspy.Predict(Signature1)
        self.predict2 = dspy.Predict(Signature2)
    
    def forward(self, input1, input2):
        # Any logic you want
        result = self.predict1(input1=input1)
        return self.predict2(input2=input2, context=result.output)
```

### C. Custom Optimizers

```python
class MyOptimizer(dspy.Teleprompter):
    def compile(self, student, *, trainset, teacher=None, **kwargs):
        # Your optimization logic
        # Modify student's Predict modules
        return optimized_student
```

### C. Custom Adapters

```python
class MyAdapter(dspy.Adapter):
    def format(self, signature, inputs, demos):
        # Create your custom prompt format
        return prompt_string
    
    def parse(self, signature, response):
        # Extract outputs from response
        return {"field1": value1, "field2": value2}
```

---

## 10. KEY FILES REFERENCE

| File | Purpose | Key Classes |
|------|---------|-------------|
| `primitives/module.py` | Program base class | Module, ProgramMeta |
| `primitives/base_module.py` | Save/load, parameters | BaseModule |
| `primitives/example.py` | Data container | Example |
| `primitives/prediction.py` | LM output | Prediction, Completions |
| `signatures/signature.py` | Task spec | Signature, SignatureMeta |
| `signatures/field.py` | Field definitions | InputField, OutputField |
| `predict/predict.py` | Core LM caller | Predict |
| `predict/parameter.py` | Optimization marker | Parameter |
| `predict/chain_of_thought.py` | Reasoning | ChainOfThought |
| `clients/base_lm.py` | LM interface | BaseLM |
| `clients/lm.py` | Main LM impl | LM |
| `clients/cache.py` | Response caching | request_cache |
| `adapters/base.py` | Format handling | Adapter |
| `adapters/chat_adapter.py` | Chat format | ChatAdapter |
| `adapters/json_adapter.py` | JSON format | JSONAdapter |
| `teleprompt/teleprompt.py` | Optimizer base | Teleprompter |
| `teleprompt/bootstrap.py` | Demo selection | BootstrapFewShot |
| `evaluate/evaluate.py` | Testing framework | Evaluate |
| `dsp/utils/settings.py` | Global config | Settings |
| `retrievers/retrieve.py` | Information retrieval | Retrieve |

---

## 11. DEPENDENCY GRAPH

```
Core Dependencies:
  Pydantic 2.0+      → Type validation for signatures
  LiteLLM 1.64+      → Multi-provider LM support
  Optuna 3.4+        → Hyperparameter optimization
  Openai 0.28+       → OpenAI API compatibility
  
Utility Dependencies:
  tqdm               → Progress bars
  requests           → HTTP
  diskcache 5.6+     → Disk-based caching
  cloudpickle 3.0+   → Serialize complex objects
  tenacity 8.2+      → Retry logic
  
Specialized Dependencies:
  gepa[dspy] 0.0.22  → GEPA optimizer
  anthropic          → Anthropic models (optional)
  weaviate           → Vector search (optional)
  langchain_core     → LangChain integration (optional)
  mcp                → Model Context Protocol (optional)
```

---

## 12. SUMMARY TABLE

| Aspect | Description |
|--------|-------------|
| **Type** | Framework for programming (not prompting) language models |
| **Core Unit** | dspy.Module (composable, optimizable programs) |
| **Task Spec** | dspy.Signature (declarative I/O contracts) |
| **LM Calling** | dspy.Predict (adapter-based, parameterized) |
| **Optimization** | Teleprompters (automatic prompt/demo improvement) |
| **Evaluation** | dspy.Evaluate (parallel, metric-driven) |
| **Threading** | Thread-safe settings, parallel modules, async support |
| **State** | Save/load entire programs or just parameters |
| **Extensibility** | Custom LMs, modules, optimizers, adapters |
| **Philosophy** | Composition, declaration, compilation paradigm |

