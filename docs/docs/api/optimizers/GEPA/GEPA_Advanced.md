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

## ReAct Component Optimization

### What is optimize_react_components?

Enable `optimize_react_components=True` to apply specialized optimization to `dspy.ReAct` modules while using default optimization for other modules.

A [`dspy.ReAct`](../../learn/programming/tools.md#approach-1-using-dspyreact-fully-managed) module has three parts: a **react predictor** (iteratively reasons and selects tools), an **extract predictor** (extracts final answers from trajectories), and **tools** with their schemas.

**What gets optimized for ReAct modules:**

GEPA can improve textual components across all parts:
- **React instruction** - Guides reasoning and tool selection (always optimized)
- **Extract instruction** - Guides answer extraction from trajectories (optional)
- **Tool descriptions** - Describes what each tool does (optional)
- **Tool argument descriptions** - Describes tool parameters (optional)

The reflection LM decides which optional components to improve based on observed failures. Non-ReAct modules in your program are optimized using GEPA's default signature optimization.

**Why this matters:**

Unlike optimizing signature instructions alone (which improves individual predictors), ReAct optimization improves the **entire agent workflow** - from initial reasoning through tool execution to final answer extraction.

ReAct agents often fail when their components contradict each other. A clear tool description doesn't help if the react instruction never considers using that tool. GEPA analyzes execution traces to learn how all components should work together.

### ReAct Optimization Prompt

GEPA uses a specialized prompt to jointly optimize all ReAct components. The prompt receives complete ReAct trajectories and current component texts:

```python
class GenerateImprovedReActDescriptionsFromFeedback(dspy.Signature):
    """Improve a ReAct agent based on execution examples and feedback.
    
    Analyze the trajectories to identify successful patterns and failure causes.
    Generate improved texts to help the agent succeed on similar tasks. 
    Place improved texts at their appropriate level of abstraction and specificity.
    """

    current_react_instruction = dspy.InputField(
        desc="Current ReAct module instruction guiding the ReAct agent's reasoning and tool selection"
    )
    current_extract_instruction = dspy.InputField(
        desc="Current Extract module instruction for extracting final answers from trajectories"
    )
    current_tools = dspy.InputField(
        annotation=list[dspy.Tool],
        desc="Available tools with their complete schemas"
    )
    examples_with_feedback = dspy.InputField(
        desc="Execution examples with feedback showing successes and failures"
    )

    improved_react_instruction = dspy.OutputField(
        desc="Improved ReAct module instruction"
    )
    improved_extract_instruction = dspy.OutputField(
        desc="Improved Extract module instruction",
        default=""
    )
    # Note: Tool descriptions and arg descriptions are added dynamically via signature.append()
```

The reflection LM receives all current components and execution traces, then decides which components to improve. Tool-specific fields (`improved_tool_{name}_desc`, `improved_tool_{name}_arg_{param}_desc`) are generated dynamically for each tool and parameter.

**Writing Metrics for ReAct Optimization**

GEPA optimizes ReAct modules more effectively when metrics provide feedback about the agent's execution. Here's how to write metrics that help:

```python
def react_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    """Evaluate ReAct agent performance with trajectory feedback."""
    # Check if the answer is correct
    answer_match = pred.answer == example.answer
    score = 1.0 if answer_match else 0.0
    
    # Provide feedback to help GEPA understand what happened
    feedback = "Correct answer" if answer_match else "Incorrect answer"
    
    return dspy.Prediction(score=score, feedback=feedback)
```

You can make feedback more informative by examining the trajectory:

```python
def react_metric_with_trajectory(example, pred, trace=None, pred_name=None, pred_trace=None):
    """Evaluate with trajectory analysis."""
    # Check if the answer is correct
    answer_match = pred.answer == example.answer
    score = 1.0 if answer_match else 0.0
    
    # Access the ReAct trajectory to understand agent behavior
    trajectory = getattr(pred, 'trajectory', {})
    
    # Extract tool names from trajectory (excluding 'finish')
    tools_used = []
    for key in trajectory:
        if key.startswith('tool_name_'):
            tool_name = trajectory[key]
            if tool_name != 'finish':
                tools_used.append(tool_name)
    
    # Build feedback message
    if answer_match:
        feedback = "Correct answer"
    else:
        feedback = "Incorrect answer"
    
    if tools_used:
        feedback += f". Tools: {', '.join(tools_used)}"
    
    return dspy.Prediction(score=score, feedback=feedback)
```

The trajectory contains the agent's step-by-step execution. Use it to provide feedback about:

- **Tool selection**: Were appropriate tools chosen?
- **Reasoning quality**: Did the agent think through the problem?
- **Efficiency**: Were there unnecessary steps?

The reflection LM uses your feedback to jointly improve react instructions, tool descriptions, and extraction logic.

### How It Works

When `optimize_react_components=True`, GEPA:

1. **Discovers ReAct modules** - Finds all `dspy.ReAct` instances in your program (including nested modules)
2. **Extracts components** - Collects react instructions, extract instructions, and tool schemas from each ReAct module
3. **Routes to proposers** - Separates components by type and routes them appropriately:
   - **With custom `instruction_proposer`**: Your custom proposer receives all components (both regular instructions and ReAct components) and handles the optimization logic
   - **With default proposer**: Regular instructions use default instruction proposer, ReAct components use specialized `ReActModuleProposer`
4. **Optimizes jointly** - ReAct proposer improves all four components together based on execution feedback
5. **Applies updates** - Updates your ReAct modules with improved instructions and tool descriptions

Non-ReAct modules (like `dspy.Predict` or `dspy.ChainOfThought`) continue using standard GEPA optimization.

### When to Use optimize_react_components

Enable `optimize_react_components=True` when you use `dspy.ReAct` in your program and need better agent performance. GEPA jointly optimizes all ReAct components (react instruction, extract instruction, tool descriptions, tool argument descriptions) based on execution feedback. Common scenarios:

1. **Agent loops with repeated tool calls** - Agent keeps calling `web_search` multiple times with similar queries instead of synthesizing information. GEPA improves react instruction to encourage synthesis and tool descriptions to clarify when searches are sufficient.

2. **Wrong tool selection** - Agent with `search` and `calculator` tools keeps searching when it should calculate, or vice versa. GEPA refines react instruction and tool descriptions to clarify "use search for factual queries, calculator for numerical analysis."

3. **Agent gives up without trying tools** - Agent responds "I don't know" without using available tools that could answer the question. GEPA improves react instruction to be more proactive about tool usage.

4. **Extraction failures** - Agent executes tools correctly but fails to extract the final answer from the trajectory. GEPA improves extract instruction to better identify and format answers from tool outputs.

5. **Multi-agent delegation issues** - Parent agent has delegation tools to specialized sub-agents but doesn't understand when to use each. GEPA optimizes all ReAct components across both parent and sub-agent modules for coherent delegation.

See the usage examples below for basic ReAct agents and multi-agent systems.

### Usage Examples

#### Basic ReAct Agent

```python
import dspy

def search_web(query: str) -> str:
    return f"Search results for: {query}"

def calculate(expression: str) -> float:
    return eval(expression)

# Create ReAct agent with tools (poor initial descriptions)
search_tool = dspy.Tool(search_web, name="search", desc="Finds things")
calc_tool = dspy.Tool(calculate, name="calculator", desc="Does calculations")

agent = dspy.ReAct("question -> answer", tools=[search_tool, calc_tool])

# Enable tool optimization
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5-mini"),
    optimize_react_components=True,
    component_selector="all",  # Optimize all components together
    auto="medium"
)

optimized_agent = gepa.compile(agent, trainset=train_examples, valset=val_examples)

# View optimized tool descriptions
print("Optimized search tool:", optimized_agent.tools["search"].desc)
print("Optimized calculator tool:", optimized_agent.tools["calculator"].desc)
```

**Example output after optimization:**
```
Optimized search tool: Use when you need to find current information, facts, or data 
    from external sources. Provide specific search queries to get relevant results.

Optimized calculator tool: Use for arithmetic operations and mathematical expressions. 
    Accepts Python-compatible expressions with numbers and operators (+, -, *, /, **). 
    Do not use for date calculations or string manipulations.
```

#### Multi-Agent System

GEPA automatically discovers and optimizes tools in nested agents:

```python
import dspy

def search_web(query: str) -> str:
    return f"Search results for: {query}"

def calculate(expression: str) -> float:
    return eval(expression)

search_tool = dspy.Tool(search_web, name="search", desc="Searches")
calc_tool = dspy.Tool(calculate, name="calculator", desc="Computes")

class ResearchAssistant(dspy.Module):
    def __init__(self):
        super().__init__()
        self.researcher = dspy.ReAct("query -> findings", tools=[search_tool])
        
        def delegate_research(query: str) -> str:
            return self.researcher(query=query).findings
        
        research_tool = dspy.Tool(delegate_research, name="research", desc="Helps with questions")
        self.assistant = dspy.ReAct("question -> answer", tools=[research_tool, calc_tool])
    
    def forward(self, question):
        return self.assistant(question=question)

# Optimizes ALL tools: calculator, research, search
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5-mini"),
    optimize_react_components=True,
    component_selector="all",
    auto="medium"
)

optimized_system = gepa.compile(ResearchAssistant(), trainset=train, valset=val)

# View optimized nested tool descriptions
print(optimized_system.researcher.tools["search"].desc)
print(optimized_system.assistant.tools["research"].desc)
print(optimized_system.assistant.tools["calculator"].desc)
```

### Inspecting Optimized ReAct Components

After optimization, all ReAct components are automatically updated in your program. Access them directly:

```python
optimized_agent = gepa.compile(agent, trainset=train, valset=val)

# ReAct instruction (guides reasoning and tool selection)
print("React instruction:", optimized_agent.react.signature.instructions)

# Extract instruction (guides answer extraction from trajectory)
print("Extract instruction:", optimized_agent.extract.predict.signature.instructions)

# Tool descriptions
for tool_name, tool in optimized_agent.tools.items():
    if tool_name != 'finish':  # Skip the built-in finish tool
        print(f"Tool '{tool_name}' description:", tool.desc)
        # Tool argument descriptions
        print(f"  Argument descriptions:", tool.arg_desc)
```

### Custom Instruction Proposers and ReAct Optimization

**Important:** When you provide a custom `instruction_proposer`, it receives ALL components (regular predictors AND ReAct modules). You must set `optimize_react_components=True` to enable ReAct module discovery and serialization, then handle the optimization logic yourself.

**How it works internally:**

1. **Component Discovery** - GEPA discovers components in your program:
   - Regular predictors → keys like `"predict"`, `"chain_of_thought"`
   - ReAct modules → keys like `"react_module"` or `"react_module:agent_name"`

2. **ReAct Serialization** - When `optimize_react_components=True`, GEPA serializes ReAct modules as JSON:
   ```json
   {
     "react": "instruction for reasoning and tool selection",
     "extract": "instruction for answer extraction",
     "tools": {
       "tool_name": {
         "desc": "what the tool does",
         "args": {"param": {"type": "string"}},
         "arg_desc": {"param": "description of param"}
       }
     }
   }
   ```

3. **Custom Proposer Receives**:
   - `candidate: dict[str, str]` - **All values are strings**
     - Regular component: `candidate["predict"]` → `"Your instruction here"`
     - ReAct component: `candidate["react_module"]` → `'{"react": "...", "extract": "...", "tools": {...}}'` (JSON as a string)
   - `reflective_dataset: dict[str, list[ReflectiveExample]]` - **GEPA provides this** 
     - Contains execution traces: inputs, outputs (including full ReAct trajectory), and your metric's feedback
     - For ReAct: `Generated_Outputs` includes the entire trajectory with all tool calls and reasoning
     - Use this to understand what went wrong and guide your improvements
   - `components_to_update: list[str]` - Component keys to optimize this round

4. **Your Responsibility**:
   - For ReAct components: Use `json.loads()` to parse, improve all 4 parts, use `json.dumps()` to return
   - For regular components: Improve the instruction string directly
   - Return `dict[str, str]` with same keys

**What this means:**
- Your custom proposer receives ALL components: regular signatures AND ReAct modules
- GEPA still does discovery and JSON serialization, but YOU handle the optimization logic
- ReAct components are passed with keys like `"react_module"` or `"react_module:agent_name"`

#### Implementing a Custom Proposer for ReAct

If you need custom logic, you can start with the existing implementation at [`ReActModuleProposer`](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/gepa/instruction_proposal.py). This reference implementation shows how to:

- Parse ReAct JSON configurations with `json.loads()`
- Build dynamic signatures for tools and parameters
- Call the reflection LM to optimize all components jointly
- Handle optional improvements (reflection LM returns `None` to keep originals)
- Serialize improved components back to JSON with `json.dumps()`

**Key concepts for custom proposers:**

ReAct components are JSON strings containing 4 parts:
```json
{
  "react": "instruction for reasoning and tool selection",
  "extract": "instruction for answer extraction",
  "tools": {
    "tool_name": {
      "desc": "what the tool does",
      "args": {"param": {"type": "string"}},
      "arg_desc": {"param": "description of param"}
    }
  }
}
```

Your proposer receives:
- `candidate: dict[str, str]` - Component names to instructions (ReAct values are JSON strings)
- `reflective_dataset: dict[str, list[ReflectiveExample]]` - Execution traces with feedback
- `components_to_update: list[str]` - Which components to optimize this round

Your proposer returns:
- `dict[str, str]` - Same keys with improved instructions (ReAct as JSON strings)
