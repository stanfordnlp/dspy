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

## Tool Optimization

### What is enable_tool_optimization?

Many DSPy programs use tools, with modules like `dspy.ReAct` as canonical examples. When `enable_tool_optimization=True`, GEPA jointly optimizes tool-using modules as a whole: predictor instructions and tool descriptions and argument descriptions are updated together, instead of being tuned in isolation. This lets the model learn better patterns for when to call a tool and how to use it from the same execution traces and feedback that drive core GEPA.

### Usage and constraints

- **Expose tools as `dspy.Tool` in signatures and examples.** GEPA only optimizes tools that are represented as `dspy.Tool` and actually passed as `dspy.Tool` objects into your modules.
- **Treat `Tool.name` as a stable identifier.** `Tool.name` is the tool's name, and GEPA uses it to attach improved descriptions and argument descriptions. If you reuse the same `Tool.name` for different tools, they will share the same text updates.
- **Avoid custom tools named `"finish"`.** The built-in ReAct `"finish"` tool is reserved and excluded from optimization. Custom tools with the name `"finish"` are also not optimized.
- **Custom instruction proposers handle all modules and tool updates.** When you provide an `instruction_proposer`, GEPA routes every optimized module through your proposer instead of the built-in instruction proposer. If `enable_tool_optimization=True`, modules that call tools are still included, and your proposer is also responsible for updating their tool descriptions and argument descriptions.

### Tool Module Optimization Prompt

GEPA uses `ToolModuleProposer` to optimize tool-using modules when `enable_tool_optimization=True`. For each module, the proposer builds a dynamic signature from the base `GenerateImprovedToolModuleDescriptionsFromFeedback` signature shown below, then appends output fields for each tool description and each tool argument description in that module. For ReAct modules, the proposer also appends input and output fields for the extract instruction.

```python
class GenerateImprovedToolModuleDescriptionsFromFeedback(dspy.Signature):
    """I provided an assistant with predictor instructions and tool descriptions,
    but its performance needs improvement based on the examples_with_feedback below.

    Your task is to propose better predictor instructions, tool descriptions, and
    tool argument descriptions that address the issues shown in these examples.
    Focus on reinforcing patterns that clearly improve the assistant's performance
    on similar tasks, rather than rewriting everything from scratch unless necessary.
    These components are progressively optimized - refine only what needs to change.

    Analyze the examples_with_feedback to identify success and failure patterns,
    and write improved instructions and descriptions at their appropriate level
    of abstraction and/or specificity, so that each layer plays a clear,
    complementary role without unnecessary repetition or verbosity unless
    redundancy clearly helps the assistant's performance.
    """

    current_predictor_instruction = dspy.InputField(
        desc="Current instruction guiding the predictor"
    )
    current_tools = dspy.InputField(
        annotation=list[dspy.Tool],
        desc="Available tools with their complete schemas"
    )
    examples_with_feedback = dspy.InputField(
        desc="Execution examples with feedback showing successes and failures"
    )

    improved_predictor_instruction: str | None = dspy.OutputField(
        desc="Improved instruction for the predictor",
        default=None
    )

    # GEPA appends output fields dynamically for each tool and argument:
    # - improved_tool_{name}_desc with desc="Improved description of tool '{name}'"
    # - improved_tool_{name}_arg_{param}_desc with desc="Improved description of the argument '{param}' of tool '{name}'"
    # For ReAct modules, GEPA also appends:
    # - current_extract_instruction (input) with desc="Current instruction for extraction predictor"
    # - improved_extract_instruction (output) with desc="Improved instruction for extraction"
```

The reflection LM uses this dynamically-built signature to jointly propose updates across predictor instructions, tool descriptions, and argument descriptions based on execution feedback. Updates are coordinated rather than made in isolation: the LM sees all current components together and can selectively update any subset by returning new text, or return `None` to keep a component unchanged.

### How Tool Optimization Works

When `enable_tool_optimization=True`, GEPA:

1. **Discovers tool-using modules** - Identifies any module that uses `dspy.Tool` instances, including `dspy.ReAct` and generic predictors with `dspy.Tool` as an input field type in their signatures
2. **Treats them as joint optimization units** - Instead of only optimizing predictor instructions, GEPA optimizes predictor instructions and tool descriptions together as a coordinated set; for ReAct this includes both the react and extract instructions
3. **Routes to specialized proposer** - Separates components by type and routes them appropriately:
   - **With custom `instruction_proposer`**: Your custom proposer receives both tool-using modules and plain predictors, and is responsible for updating all components
   - **With default proposer**: Plain predictors use the default instruction proposer; tool-using modules use `ToolModuleProposer`, which employs the dynamic signature mechanism described above
4. **Optimizes jointly** - `ToolModuleProposer` improves predictor instructions and tool descriptions together based on execution feedback, coordinating updates across all components rather than tuning them in isolation
5. **Applies updates** - Improved instructions update predictor signatures; improved tool descriptions and argument descriptions update all `dspy.Tool` objects with matching tool names throughout the program

Modules without tools (like `dspy.Predict` or `dspy.ChainOfThought`) continue using standard GEPA instruction-only optimization.

### When to Use Tool Optimization

Enable `enable_tool_optimization=True` when tools are central to your program's behavior and you want GEPA to jointly optimize predictor instructions and tool descriptions together. Common scenarios:

1. **Wrong tool selection** - Predictor with `search` and `calculator` tools keeps searching when it should calculate, or vice versa. GEPA refines predictor instructions and tool descriptions to clarify "use search for factual queries, calculator for numerical analysis."

2. **Underused tools** - Predictor responds "I don't know" without using available tools that could answer the question. GEPA improves predictor instructions to be more proactive about tool usage.

3. **Tool call loops** - Agent keeps calling `web_search` multiple times with similar queries instead of synthesizing information. GEPA improves instructions to encourage synthesis and tool descriptions to clarify when searches are sufficient.

4. **Extraction failures (ReAct)** - Agent executes tools correctly but fails to extract the final answer from the trajectory. GEPA improves extract instruction to better identify and format answers from tool outputs.

5. **Multi-agent delegation** - Parent agent has delegation tools to specialized sub-agents but doesn't understand when to use each. GEPA optimizes instructions and tool descriptions across both parent and sub-agent modules for coherent delegation.

See the usage examples below for tool-using programs.

### Usage Examples

#### Custom Tool-Using Agent

```python
import dspy

def search_web(query: str) -> str:
    """Search for information."""
    return f"Search results for: {query}"

def finish_task(final_answer: str) -> str:
    """Signal completion and return final answer."""
    return final_answer

# Signature with tools, history tracking, and tool_calls output
class AgentSignature(dspy.Signature):
    task: str = dspy.InputField()
    tools: list[dspy.Tool] = dspy.InputField()
    max_iters: int = dspy.InputField()
    history: list[dict] = dspy.InputField()
    outputs: dspy.ToolCalls = dspy.OutputField()

class CustomAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.max_iters = 3
        self.tools = {
            "search_web": dspy.Tool(search_web, name="search_web", desc="Search tool"),
            "finish_task": dspy.Tool(finish_task, name="finish_task", desc="Finish tool"),  # Avoid "finish" tool name if program uses ReAct
        }
        self.predictor = dspy.Predict(AgentSignature)
    
    def forward(self, task: str):
        history = []
        for iteration in range(self.max_iters):
            response = self.predictor(
                task=task,
                tools=self.tools.values(),
                max_iters=self.max_iters,
                history=history,
            )
            for call in response.outputs.tool_calls:
                result = call.execute()
                if call.name == "finish_task":
                    return dspy.Prediction(answer=result)
                history.append({
                    "tool_call_name": call.name,
                    "tool_call_args": call.args,
                    "tool_call_result": result,
                })
        return dspy.Prediction(answer="No answer")

program = CustomAgent()

# Enable tool optimization
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5-mini"),
    enable_tool_optimization=True,
    auto="medium"
)

optimized_program = gepa.compile(program, trainset=train_examples, valset=val_examples)
```

#### ReAct Agent

```python
import dspy

def search_web(query: str) -> str:
    return f"Search results for: {query}"

def calculate(expression: str) -> float:
    return eval(expression)

# Create tools with basic descriptions
search_tool = dspy.Tool(search_web, name="search_web", desc="Search tool")
calc_tool = dspy.Tool(calculate, name="calculate", desc="Calculator tool")

program = dspy.ReAct("question -> answer", tools=[search_tool, calc_tool])

# Enable tool optimization
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5-mini"),
    enable_tool_optimization=True,
    auto="medium"
)

optimized_program = gepa.compile(program, trainset=train_examples, valset=val_examples)
```

### Inspecting Optimized Programs

View optimization results and metadata (requires `track_stats=True`):

```python
# High-level optimization metadata
optimized_program.detailed_results
```

Access optimized instructions and tool descriptions directly:

```python
# Predictor instructions
for name, predictor in optimized_program.named_predictors():
    print(f"{name}: {predictor.signature.instructions}")

# Tool descriptions and argument descriptions
for tool_name, tool in optimized_program.tools.items():
    print(f"{tool_name}: {tool.desc}")
    for arg_name, arg_schema in tool.args.items():
        print(f"  {arg_name}: {arg_schema.get('description', 'N/A')}")
```

### Custom Instruction Proposers with Tool Optimization

When you provide a custom `instruction_proposer`, GEPA routes **all components** to your proposer, including both plain predictors and tool-using modules. Your proposer must handle both.

**What your proposer receives:**

- **Plain predictors**: instruction strings keyed by predictor name
- **Tool-using modules**: JSON strings keyed by module identifier, containing predictor instructions and tool schemas
  - Generic tool modules: `f"{TOOL_MODULE_PREFIX}:{predictor_name}"`
  - ReAct modules: `f"{REACT_MODULE_PREFIX}:{extract_predictor_name}"`

**Your proposer's responsibilities:**

```python
import json
from dspy.teleprompt.gepa.gepa_utils import REACT_MODULE_PREFIX, TOOL_MODULE_PREFIX

def custom_proposer(candidate, reflective_dataset, components_to_update):
    """Custom instruction proposer for GEPA with tool optimization.
    
    Args:
        candidate: dict[str, str] - All components in the program
            {
                "predictor_name": "instruction string",
                "tool_module:pred_name": '{"pred_name": "...", "tools": {...}}',
                "react_module:extract_name": '{"react_name": "...", "extract_name": "...", "tools": {...}}'
            }
        reflective_dataset: dict[str, list[dict]] - Execution examples with feedback per component
        components_to_update: list[str] - Component keys to optimize in this call
    
    Returns:
        dict[str, str]: Improved instructions for components_to_update keys only
    """
    improved_components = {}
    
    for component_key in components_to_update:
        if component_key.startswith(REACT_MODULE_PREFIX) or component_key.startswith(TOOL_MODULE_PREFIX):
            config = json.loads(candidate[component_key])
            # Example: {"pred": "instruction", "tools": {"search": {"desc": "...", "args": {...}}}}
            
            # Find predictor names (predictor keys with string values and "tools" is a dict)
            predictor_keys = [k for k, v in config.items() if isinstance(v, str)]
            for pred_name in predictor_keys:
                config[pred_name] = "improved predictor instruction"
            
            # Update tool descriptions and argument descriptions
            for tool_name, tool_info in config.get("tools", {}).items():
                tool_info["desc"] = "improved tool description"
                for arg_name in tool_info.get("args", {}):
                    tool_info["args"][arg_name]["description"] = "improved argument description"
            
            improved_components[component_key] = json.dumps(config)
        else:
            # Plain predictor: improve instruction string only
            improved_components[component_key] = "improved instruction"
    
    return improved_components
```

Your proposer can use any optimization approach: custom prompts, LM calls, heuristics, or rule-based logic.

**Tool module JSON structure:**

Generic:
```json
{
  "predictor_name": "instruction",
  "tools": {
    "search": {
      "desc": "...",
      "args": {"query": {"type": "string", "description": "..."}}
    }
  }
}
```

ReAct:
```json
{
  "react_name": "react instruction",
  "extract_name": "extract instruction",
  "tools": { ... }
}
```

**What to update:**
- `config[predictor_name] = "proposed predictor instruction"`
- `config["tools"][tool_name]["desc"] = "proposed tool description"`
- `config["tools"][tool_name]["args"][arg_name]["description"] = "proposed argument description"`

**What to preserve:**
- `config["tools"][tool_name]["args"][arg_name]["type"]` and other schema metadata (changing these breaks the tool since they must match the underlying function's parameter types)

See [`ToolModuleProposer`](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/gepa/instruction_proposal.py) for reference.
