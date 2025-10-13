# dspy.GEPA - Advanced Features

## Custom Instruction Proposers

### What is instruction_proposer?

The `instruction_proposer` is the component responsible for invoking the `reflection_lm` and proposing new prompts during GEPA optimization. When GEPA identifies underperforming components in your DSPy program, the instruction proposer analyzes execution traces, feedback, and failures to generate improved instructions tailored to the observed issues.

### Default Implementation

By default, GEPA uses the built-in instruction proposer from the [GEPA library](https://github.com/gepa-ai/gepa), which implements the [`ProposalFn`](https://github.com/gepa-ai/gepa/blob/main/src/gepa/core/adapter.py). The [default proposer](https://github.com/gepa-ai/gepa/blob/main/src/gepa/proposer/reflective_mutation/reflective_mutation.py#L53-L75) uses this prompt template:

````
I provided an assistant with the following instructions to perform a task for me:
```
<curr_instructions>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<inputs_outputs_feedback>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks.
````

This template is automatically filled with:

- `<curr_instructions>`: The current instruction being optimized
- `<inputs_outputs_feedback>`: Structured markdown containing predictor inputs, generated outputs, and evaluation feedback

Example of default behavior:

```python
# Default instruction proposer is used automatically
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key),
    auto="medium"
)
optimized_program = gepa.compile(student, trainset=examples)
```

### When to Use Custom instruction_proposer

**Note:** Custom instruction proposers are an advanced feature. Most users should start with the default proposer, which works well for most text-based optimization tasks.

Consider implementing a custom instruction proposer when you need:

- **Multi-modal handling**: Process images (dspy.Image) alongside textual information in your inputs
- **Nuanced control on limits and length constraints**: Have more fine-grained control over instruction length, format, and structural requirements
- **Domain-specific information**: Inject specialized knowledge, terminology, or context that the default proposer lacks and cannot be provided via feedback_func. This is an advanced feature, and most users should not need to use this.
- **Provider-specific prompting guides**: Optimize instructions for specific LLM providers (OpenAI, Anthropic, etc.) with their unique formatting preferences
- **Coupled component updates**: Handle situations where 2 or more components need to be updated together in a coordinated manner, rather than optimizing each component independently (refer to component_selector parameter, in [Custom Component Selection](#custom-component-selection) section, for related functionality)
- **External knowledge integration**: Connect to databases, APIs, or knowledge bases during instruction generation

### Available Options

**Built-in Options:**

- **Default Proposer**: The standard GEPA instruction proposer (used when `instruction_proposer=None`). The default instruction proposer IS an instruction proposer as well! It is the most general one, that was used for the diverse experiments reported in the GEPA paper and tutorials.
- **MultiModalInstructionProposer**: Handles `dspy.Image` inputs and structured multimodal content.

```python
from dspy.teleprompt.gepa.instruction_proposal import MultiModalInstructionProposer

# For tasks involving images or multimodal inputs
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key),
    instruction_proposer=MultiModalInstructionProposer(),
    auto="medium"
)
```

We invite community contributions of new instruction proposers for specialized domains as the [GEPA library](https://github.com/gepa-ai/gepa) continues to grow.

### How to Implement Custom Instruction Proposers

Custom instruction proposers must implement the `ProposalFn` protocol by defining a callable class or function. GEPA will call your proposer during optimization:

```python
from dspy.teleprompt.gepa.gepa_utils import ReflectiveExample

class CustomInstructionProposer:
    def __call__(
        self,
        candidate: dict[str, str],                          # Candidate component name -> instruction mapping to be updated in this round
        reflective_dataset: dict[str, list[ReflectiveExample]],  # Component -> examples with structure: {"Inputs": ..., "Generated Outputs": ..., "Feedback": ...}
        components_to_update: list[str]                     # Which components to improve
    ) -> dict[str, str]:                                    # Return new instruction mapping only for components being updated
        # Your custom instruction generation logic here
        return updated_instructions

# Or as a function:
def custom_instruction_proposer(candidate, reflective_dataset, components_to_update):
    # Your custom instruction generation logic here
    return updated_instructions
```

**Reflective Dataset Structure:**

- `dict[str, list[ReflectiveExample]]` - Maps component names to lists of examples
- `ReflectiveExample` TypedDict contains:
  - `Inputs: dict[str, Any]` - Predictor inputs (may include dspy.Image objects)
  - `Generated_Outputs: dict[str, Any] | str` - Success: output fields dict, Failure: error message
  - `Feedback: str` - Always a string from metric function or auto-generated by GEPA

#### Basic Example: Word Limit Proposer

```python
import dspy
from gepa.core.adapter import ProposalFn
from dspy.teleprompt.gepa.gepa_utils import ReflectiveExample

class GenerateWordLimitedInstruction(dspy.Signature):
    """Given a current instruction and feedback examples, generate an improved instruction with word limit constraints."""

    current_instruction = dspy.InputField(desc="The current instruction that needs improvement")
    feedback_summary = dspy.InputField(desc="Feedback from examples that might include both positive and negative cases")
    max_words = dspy.InputField(desc="Maximum number of words allowed in the new instruction")

    improved_instruction = dspy.OutputField(desc="A new instruction that fixes the issues while staying under the max_words limit")

class WordLimitProposer(ProposalFn):
    def __init__(self, max_words: int = 1000):
        self.max_words = max_words
        self.instruction_improver = dspy.ChainOfThought(GenerateWordLimitedInstruction)

    def __call__(self, candidate: dict[str, str], reflective_dataset: dict[str, list[ReflectiveExample]], components_to_update: list[str]) -> dict[str, str]:
        updated_components = {}

        for component_name in components_to_update:
            if component_name not in candidate or component_name not in reflective_dataset:
                continue

            current_instruction = candidate[component_name]
            component_examples = reflective_dataset[component_name]

            # Create feedback summary
            feedback_text = "\n".join([
                f"Example {i+1}: {ex.get('Feedback', 'No feedback')}"
                for i, ex in enumerate(component_examples)  # Limit examples to prevent context overflow
            ])

            # Use the module to improve the instruction
            result = self.instruction_improver(
                current_instruction=current_instruction,
                feedback_summary=feedback_text,
                max_words=self.max_words
            )

            updated_components[component_name] = result.improved_instruction

        return updated_components

# Usage
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key),
    instruction_proposer=WordLimitProposer(max_words=700),
    auto="medium"
)
```

#### Advanced Example: RAG-Enhanced Instruction Proposer

```python
import dspy
from gepa.core.adapter import ProposalFn
from dspy.teleprompt.gepa.gepa_utils import ReflectiveExample

class GenerateDocumentationQuery(dspy.Signature):
    """Analyze examples with feedback to identify common issue patterns and generate targeted database queries for retrieving relevant documentation.

    Your goal is to search a document database for guidelines that address the problematic patterns found in the examples. Look for recurring issues, error types, or failure modes in the feedback, then craft specific search queries that will find documentation to help resolve these patterns."""

    current_instruction = dspy.InputField(desc="The current instruction that needs improvement")
    examples_with_feedback = dspy.InputField(desc="Examples with their feedback showing what issues occurred and any recurring patterns")

    failure_patterns: str = dspy.OutputField(desc="Summarize the common failure patterns identified in the examples")

    retrieval_queries: list[str] = dspy.OutputField(desc="Specific search queries to find relevant documentation in the database that addresses the common issue patterns identified in the problematic examples")

class GenerateRAGEnhancedInstruction(dspy.Signature):
    """Generate improved instructions using retrieved documentation and examples analysis."""

    current_instruction = dspy.InputField(desc="The current instruction that needs improvement")
    relevant_documentation = dspy.InputField(desc="Retrieved guidelines and best practices from specialized documentation")
    examples_with_feedback = dspy.InputField(desc="Examples showing what issues occurred with the current instruction")

    improved_instruction: str = dspy.OutputField(desc="Enhanced instruction that incorporates retrieved guidelines and addresses the issues shown in the examples")

class RAGInstructionImprover(dspy.Module):
    """Module that uses RAG to improve instructions with specialized documentation."""

    def __init__(self, retrieval_model):
        super().__init__()
        self.retrieve = retrieval_model  # Could be dspy.Retrieve or custom retriever
        self.query_generator = dspy.ChainOfThought(GenerateDocumentationQuery)
        self.generate_answer = dspy.ChainOfThought(GenerateRAGEnhancedInstruction)

    def forward(self, current_instruction: str, component_examples: list):
        """Improve instruction using retrieved documentation."""

        # Let LM analyze examples and generate targeted retrieval queries
        query_result = self.query_generator(
            current_instruction=current_instruction,
            examples_with_feedback=component_examples
        )

        results = self.retrieve.query(
            query_texts=query_result.retrieval_queries,
            n_results=3
        )

        relevant_docs_parts = []
        for i, (query, query_docs) in enumerate(zip(query_result.retrieval_queries, results['documents'])):
            if query_docs:
                docs_formatted = "\n".join([f"  - {doc}" for doc in query_docs])
                relevant_docs_parts.append(
                    f"**Search Query #{i+1}**: {query}\n"
                    f"**Retrieved Guidelines**:\n{docs_formatted}"
                )

        relevant_docs = "\n\n" + "="*60 + "\n\n".join(relevant_docs_parts) + "\n" + "="*60

        # Generate improved instruction with retrieved context
        result = self.generate_answer(
            current_instruction=current_instruction,
            relevant_documentation=relevant_docs,
            examples_with_feedback=component_examples
        )

        return result

class DocumentationEnhancedProposer(ProposalFn):
    """Instruction proposer that accesses specialized documentation via RAG."""

    def __init__(self, documentation_retriever):
        """
        Args:
            documentation_retriever: A retrieval model that can search your specialized docs
                                   Could be dspy.Retrieve, ChromadbRM, or custom retriever
        """
        self.instruction_improver = RAGInstructionImprover(documentation_retriever)

    def __call__(self, candidate: dict[str, str], reflective_dataset: dict[str, list[ReflectiveExample]], components_to_update: list[str]) -> dict[str, str]:
        updated_components = {}

        for component_name in components_to_update:
            if component_name not in candidate or component_name not in reflective_dataset:
                continue

            current_instruction = candidate[component_name]
            component_examples = reflective_dataset[component_name]

            result = self.instruction_improver(
                current_instruction=current_instruction,
                component_examples=component_examples
            )

            updated_components[component_name] = result.improved_instruction

        return updated_components

import chromadb

client = chromadb.Client()
collection = client.get_collection("instruction_guidelines")

gepa = dspy.GEPA(
    metric=task_specific_metric,
    reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key),
    instruction_proposer=DocumentationEnhancedProposer(collection),
    auto="medium"
)
```

#### Integration Patterns

**Using Custom Proposer with External LM:**

```python
class ExternalLMProposer(ProposalFn):
    def __init__(self):
        # Manage your own LM instance
        self.external_lm = dspy.LM('gemini/gemini-2.5-pro')

    def __call__(self, candidate, reflective_dataset, components_to_update):
        updated_components = {}

        with dspy.context(lm=self.external_lm):
            # Your custom logic here using self.external_lm
            for component_name in components_to_update:
                # ... implementation
                pass

        return updated_components

gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=None,  # Optional when using custom proposer
    instruction_proposer=ExternalLMProposer(),
    auto="medium"
)
```

**Best Practices:**

- **Use the full power of DSPy**: Leverage DSPy components like `dspy.Module`, `dspy.Signature`, and `dspy.Predict` to create your instruction proposer rather than direct LM calls. Consider `dspy.Refine` for constraint satisfaction, `dspy.ChainOfThought` for complex reasoning tasks, and compose multiple modules for sophisticated instruction improvement workflows
- **Enable holistic feedback analysis**: While dspy.GEPA's `GEPAFeedbackMetric` processes one (gold, prediction) pair at a time, instruction proposers receive all examples for a component in batch, enabling cross-example pattern detection and systematic issue identification.
- **Mind data serialization**: Serializing everything to strings might not be ideal - handle complex input types (like `dspy.Image`) by maintaining their structure for better LM processing
- **Test thoroughly**: Test your custom proposer with representative failure cases

## Custom Component Selection

### What is component_selector?

The `component_selector` parameter controls which components (predictors) in your DSPy program are selected for optimization at each GEPA iteration. Instead of the default round-robin approach that updates one component at a time, you can implement custom selection strategies that choose single or multiple components based on optimization state, performance trajectories, and other contextual information.

### Default Behavior

By default, GEPA uses a **round-robin strategy** (`RoundRobinReflectionComponentSelector`) that cycles through components sequentially, optimizing one component per iteration:

```python
# Default round-robin component selection
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key),
    # component_selector="round_robin"  # This is the default
    auto="medium"
)
```

### Built-in Selection Strategies

**String-based selectors:**

- `"round_robin"` (default): Cycles through components one at a time
- `"all"`: Selects all components for simultaneous optimization

```python
# Optimize all components simultaneously
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=reflection_lm,
    component_selector="all",  # Update all components together
    auto="medium"
)

# Explicit round-robin selection
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=reflection_lm,
    component_selector="round_robin",  # One component per iteration
    auto="medium"
)
```

### When to Use Custom Component Selection

Consider implementing custom component selection when you need:

- **Dependency-aware optimization**: Update related components together (e.g., a classifier and its input formatter)
- **LLM-driven selection**: Let an LLM analyze trajectories and decide which components need attention
- **Resource-conscious optimization**: Balance optimization thoroughness with computational budget

### Custom Component Selector Protocol

Custom component selectors must implement the [`ReflectionComponentSelector`](https://github.com/gepa-ai/gepa/blob/main/src/gepa/proposer/reflective_mutation/base.py) protocol by defining a callable class or function. GEPA will call your selector during optimization:

```python
from dspy.teleprompt.gepa.gepa_utils import GEPAState, Trajectory

class CustomComponentSelector:
    def __call__(
        self,
        state: GEPAState,                    # Complete optimization state with history
        trajectories: list[Trajectory],      # Execution traces from the current minibatch
        subsample_scores: list[float],       # Scores for each example in the current minibatch
        candidate_idx: int,                  # Index of the current program candidate being optimized
        candidate: dict[str, str],           # Component name -> instruction mapping
    ) -> list[str]:                          # Return list of component names to optimize
        # Your custom component selection logic here
        return selected_components

# Or as a function:
def custom_component_selector(state, trajectories, subsample_scores, candidate_idx, candidate):
    # Your custom component selection logic here
    return selected_components
```

### Custom Implementation Example

Here's a simple function that alternates between optimizing different halves of your components:

```python
def alternating_half_selector(state, trajectories, subsample_scores, candidate_idx, candidate):
    """Optimize half the components on even iterations, half on odd iterations."""
    components = list(candidate.keys())

    # If there's only one component, always optimize it
    if len(components) <= 1:
        return components

    mid_point = len(components) // 2

    # Use state.i (iteration counter) to alternate between halves
    if state.i % 2 == 0:
        # Even iteration: optimize first half
        return components[:mid_point]
    else:
        # Odd iteration: optimize second half
        return components[mid_point:]

# Usage
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=reflection_lm,
    component_selector=alternating_half_selector,
    auto="medium"
)
```

### Integration with Custom Instruction Proposers

Component selectors work seamlessly with custom instruction proposers. The selector determines which components to update, then the instruction proposer generates new instructions for those components:

```python
# Combined custom selector + custom proposer
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=reflection_lm,
    component_selector=alternating_half_selector,
    instruction_proposer=WordLimitProposer(max_words=500),
    auto="medium"
)
```

## Tool Description Optimization

### What is optimize_tool_descriptions?

The `optimize_tool_descriptions` parameter enables GEPA to optimize tool descriptions in addition to signature instructions. This is particularly valuable for ReAct agents and other tool-using systems, where the quality of tool descriptions directly impacts the agent's ability to select appropriate tools for each task.

Unlike signature instructions that guide reasoning strategies, tool descriptions serve a different purpose: they help agents decide **which tool to use** in a given situation. GEPA applies a specialized reflection prompt tailored for tool selection decisions.

### Tool-Specific Reflection Prompt

GEPA uses a dedicated prompt for optimizing tool descriptions. The prompt receives the complete ReAct trajectory (all thoughts, actions, observations) from executions that used the tool being optimized:

```python
class GenerateImprovedToolDescriptionFromFeedback(dspy.Signature):
    """You are refining a tool description that the assistant currently uses.

    Review the current description along with examples of the assistant's tool decisions 
    and the feedback those decisions received.

    Read them together and refine the description.
    So the agent understands when this tool actually helps, what argument or result matters, 
    and what misuse the feedback exposed. Keep the tool's voice and only change what the 
    evidence justifies.

    Return a refined description that helps the assistant quickly recognize good 
    opportunities for the tool."""

    current_tool_description = dspy.InputField(desc="The current description of the tool")
    examples_with_feedback = dspy.InputField(
        desc="Examples showing tool usage decisions and feedback on correctness"
    )

    improved_tool_description = dspy.OutputField(
        desc="An improved description that guides correct tool selection and usage"
    )
```

The `examples_with_feedback` contains full ReAct trajectories showing the complete context in which each tool was selected and used, enabling the reflection LM to understand tool selection patterns.

**Example: Writing Tool-Aware Metrics**

To provide effective feedback for tool optimization, write metrics that examine the trajectory:

```python
def tool_feedback_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Metric that provides tool-specific feedback for GEPA optimization."""
    correct = prediction.answer == example.answer
    score = 1.0 if correct else 0.0
    
    # Generate tool-specific feedback if available
    if hasattr(prediction, 'trajectory'):
        tools_used = [
            prediction.trajectory[key] 
            for key in prediction.trajectory 
            if key.startswith('tool_name_') and prediction.trajectory[key] != 'finish'
        ]
        feedback = f"{'Correct' if correct else 'Wrong'}. Tools: {', '.join(tools_used)}"
    else:
        feedback = "Correct" if correct else "Wrong"
    
    return dspy.Prediction(score=score, feedback=feedback)
```

This produces feedback like:
```
[Tool 'calculator' from 'agent'] Correct. Tools: calculator
[Tool 'search' from 'agent'] Wrong. Tools: search, calculator
```

The tool-specific prefix `[Tool 'calculator' from 'agent']` is automatically added by GEPA to focus the reflection LM on optimizing that particular tool's description.

**Note:** Tool descriptions are treated as components in GEPA's optimization process. The `component_selector` parameter applies to both signature instructions and tool descriptions. For example, `component_selector="all"` optimizes all signatures and tools together, while `component_selector="round_robin"` cycles through them one at a time.

### Default Behavior

By default, GEPA only optimizes signature instructions (`optimize_tool_descriptions=False`):

```python
# Default behavior: only signature optimization
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key),
    # optimize_tool_descriptions=False  # This is the default
    auto="medium"
)
optimized_program = gepa.compile(student, trainset=examples)
```

### When to Use optimize_tool_descriptions

Consider enabling `optimize_tool_descriptions=True` when:

- **Building ReAct agents**: ReAct agents rely on tool descriptions to make action selection decisions
- **Multi-agent systems**: Systems with nested agents and delegated tools benefit from holistic optimization
- **Poor tool selection**: Your agent frequently selects wrong tools or overlooks appropriate ones
- **Complex tool sets**: When managing many tools with overlapping capabilities
- **Domain-specific tools**: Tools requiring specialized knowledge or context for proper usage

### How It Works

When enabled, GEPA:

1. **Discovers all tools**: Traverses your program including nested sub-modules to find all `dspy.Tool` instances
2. **Categorizes components**: Separates tools (identified by `tool:` prefix) from signature instructions
3. **Routes components appropriately**:
   - Signature instructions → Default or custom instruction proposer
   - Tool descriptions → ToolProposer (receives ReAct's reflective data with tool-specific annotation)
4. **Optimizes holistically**: Treats tool descriptions as first-class components in the optimization process

### Implementation Details

**Reflective Dataset Construction:**

GEPA constructs the reflective dataset for tool optimization in two passes:

**Pass 1: Build reflective examples for predictors (used by instruction proposer)**

For each predictor (including ReAct modules), GEPA creates reflective examples containing:
- **Inputs**: The predictor's input fields (e.g., `{"question": "..."}`)
- **Generated Outputs**: ALL of the predictor's output fields converted to strings
  - For ReAct: This includes both `answer` AND `trajectory` fields
  - The trajectory contains the complete execution trace with all thoughts, actions, and observations
- **Feedback**: Text feedback returned by your metric function

These examples are used by the instruction proposer to optimize signature instructions.

**Pass 2: Copy reflective examples to tools with annotation (used by tool proposer)**

For each tool being optimized, GEPA:
- Identifies ALL ReAct predictors (across all nested modules) that have this tool in their toolset
- Takes ALL reflective examples from those predictors and makes a deep copy for the tool
- Annotates the feedback: `[Tool 'tool_name' from 'predictor_key'] {original_feedback}`
- If multiple ReAct modules use the same tool, their reflective examples are aggregated together

These annotated examples are used by the tool proposer (with the tool-specific reflection prompt shown above) to optimize tool descriptions.

This means:
- A tool receives the FULL ReAct trajectory (thoughts, actions, observations) in the "Generated Outputs" field
- The metric can optionally examine the trajectory and include tool-specific insights in the feedback text
- The reflection LM sees complete context about how and when the tool was used

**Component Identification & Proposer Routing:**

Tools are identified with a `tool:` prefix (e.g., `tool:calculator`). GEPA uses independent proposers:
- **Signature instructions** → Custom instruction proposer (if provided) OR default GEPA proposer
- **Tool descriptions** (prefixed with `tool:`) → Built-in `ToolProposer` (always used, not customizable)

The custom instruction proposer affects ONLY signature instructions. Tools always use the built-in `ToolProposer`, regardless of whether you provide a custom instruction proposer.

### Usage Examples

#### Basic ReAct Agent

```python
import dspy

def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation here
    return search_results

def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    # Implementation here
    return result

# Create ReAct agent with tools
search_tool = dspy.Tool(search_web, name="search", desc="Search the web")
calc_tool = dspy.Tool(calculate, name="calculator", desc="Do math")

agent = dspy.ReAct("question -> answer", tools=[search_tool, calc_tool])

# Enable tool optimization
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key),
    optimize_tool_descriptions=True,  # Enable tool optimization
    auto="medium"
)

optimized_agent = gepa.compile(agent, trainset=train_examples, valset=val_examples)
```

#### Multi-Agent System

For systems with nested agents, GEPA automatically discovers and optimizes all tools:

```python
import dspy

def search_web(query: str) -> str:
    """Search the web."""
    # Implementation here
    return results

def calculate(expression: str) -> float:
    """Evaluate math expression."""
    # Implementation here
    return result

# Define tools
search_tool = dspy.Tool(search_web, name="search", desc="Searches web")
calc_tool = dspy.Tool(calculate, name="calculator", desc="Does math")

class ResearchAssistant(dspy.Module):
    def __init__(self):
        super().__init__()
        # Sub-agent with search tool
        self.researcher = dspy.ReAct("query -> findings", tools=[search_tool])
        
        # Delegation tool wraps sub-agent
        def delegate_research(query: str) -> str:
            return self.researcher(query=query).findings
        
        research_tool = dspy.Tool(delegate_research, name="research", desc="Research things")
        
        # Main agent with calculator and research delegation
        self.assistant = dspy.ReAct("question -> answer", tools=[research_tool, calc_tool])
    
    def forward(self, question):
        return self.assistant(question=question)

# GEPA optimizes ALL tools (calculator, research, search) together
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key),
    optimize_tool_descriptions=True,
    auto="medium"
)

optimized_system = gepa.compile(ResearchAssistant(), trainset=train, valset=val)
```

### Inspecting Optimized Tool Descriptions

After optimization, tool descriptions are automatically updated in your program. Access them directly through your module structure:

```python
optimized_agent = gepa.compile(agent, trainset=train, valset=val)

# Access tools directly - descriptions are already updated
print(optimized_agent.tools["search"].desc)
print(optimized_agent.tools["calculator"].desc)
```

For multi-agent systems, access nested tools through your module hierarchy:

```python
optimized_system = gepa.compile(ResearchAssistant(), trainset=train, valset=val)

# Access tools at different levels
print(optimized_system.researcher.tools["search"].desc)  # Sub-agent tool
print(optimized_system.assistant.tools["research"].desc)  # Main agent tool
print(optimized_system.assistant.tools["calculator"].desc)
```

### Compatibility with Custom Instruction Proposers

Tool optimization works seamlessly with custom instruction proposers. When both are provided:

- Signature instructions → Custom instruction proposer
- Tool descriptions → Built-in `ToolProposer`

```python
from dspy.teleprompt.gepa.instruction_proposal import MultiModalInstructionProposer

gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key),
    instruction_proposer=MultiModalInstructionProposer(),  # For signatures
    optimize_tool_descriptions=True,  # Enables ToolProposer for tools
    auto="medium"
)
```

**Note:** Tool optimization is fully backward compatible. Existing programs without tools, or with `optimize_tool_descriptions=False`, continue to work exactly as before.
