# dspy.GEPA - Advanced Features

## Custom Instruction Proposers

### What is instruction_proposer?

The `instruction_proposer` is the component responsible for invoking the `reflection_lm` and proposing new prompts during GEPA optimization. When GEPA identifies underperforming components in your DSPy program, the instruction proposer analyzes execution traces, feedback, and failures to generate improved instructions tailored to the observed issues.

### Default Implementation

By default, GEPA uses `InstructionProposer` from `dspy.teleprompt.gepa`. When you pass no `instruction_proposer`, GEPA builds `InstructionProposer()` with its defaults. The proposer passes reflective examples as a list of dictionaries to `dspy.Predict`. The adapter renders their inputs, outputs, and feedback, including history and multimodal values. The proposer asks the `reflection_lm` for a new instruction through `dspy.Predict` with a `JSONAdapter`. Inputs that are `dspy.Type` instances, such as `dspy.Image`, reach the reflection LM as structured content.

The default prompt is the `ProposeInstruction` signature. Its instructions are:

````
I provided an assistant with instructions to perform a task for me. You are given those instructions, along with examples of different task inputs provided to the assistant, the assistant's response for each of them, and some feedback on how the assistant's response could be better.

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.
````

The signature has two input fields and one output field:

- `current_instruction`: The current instruction being optimized
- `examples_with_feedback`: A list of dictionaries containing predictor inputs, generated outputs, and evaluation feedback
- `new_instruction`: The proposed instruction

Example of default behavior:

```python
# Default instruction proposer is used automatically
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5", max_tokens=32000, api_key=api_key),
    auto="medium"
)
optimized_program = gepa.compile(student, trainset=examples)
```

#### Configuring the default proposer

Pass an `InstructionProposer(...)` instance to configure the default behavior. `skills`, `additional_instructions`, and `max_chars` each add one input field to the proposal prompt when set. `InstructionProposer()` sends only the two base fields.

| Option | Effect |
|---|---|
| `skills` | Reference material shown to the reflection LM, as one source or a sequence of sources. Each entry is a path to a markdown or text file, a directory holding `SKILL.md` (the Agent Skills layout), or an inline string. A path that does not exist raises at construction. A leading YAML frontmatter block supplies `name` and `description`, including multiline descriptions. Empty bodies and invalid frontmatter raise at construction. |
| `additional_instructions` | Guidance applied to every proposal, such as "Write instructions in imperative voice." |
| `base_instructions` | Replaces the prompt text above. The input and output fields stay the same. |
| `max_chars` | Maximum Unicode characters in each proposed instruction after stripping outer whitespace. Pydantic validates this limit. `None` means no limit. |
| `truncate_history_outputs` | When `True`, long tool results inside `dspy.History` inputs (for example from `dspy.ReActV2`) and long outputs inside `REPLHistory` inputs are cut to 500 characters before the examples are rendered. Nothing else is shortened. |
| `adapter` | The adapter for the proposer's own LM calls. Defaults to `JSONAdapter()`. |

When a proposal exceeds `max_chars`, the proposer makes one more call asking the reflection LM to shorten the draft. Pydantic validates the result and raises `ValidationError` if it is empty or still too long. The proposer never truncates an instruction. Validation runs after adapter parsing, so the limit is enforced with custom adapters too.

```python
from dspy.teleprompt.gepa import InstructionProposer

gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5", max_tokens=32000, api_key=api_key),
    instruction_proposer=InstructionProposer(
        skills=["./skills/prompt-engineering", "./skills/prompt-engineering/models/openai.md"],
        additional_instructions="Write instructions in imperative voice.",
        max_chars=1500,
        truncate_history_outputs=True,
    ),
    auto="medium",
)
```

Exceptions raised while proposing propagate to GEPA for proposal failure handling.

### When to Use Custom instruction_proposer

**Note:** Custom instruction proposers are an advanced feature. Most users should start with the default proposer, which works well for most optimization tasks, and reach for its options (`skills`, `additional_instructions`, `base_instructions`, `max_chars`, and `truncate_history_outputs`) before writing their own.

Consider implementing a custom instruction proposer when you need:

- **Nuanced control on format and structure**: Requirements on instruction format or structure that go beyond what `additional_instructions` and `max_chars` express
- **Coupled component updates**: Handle situations where 2 or more components need to be updated together in a coordinated manner, rather than optimizing each component independently (refer to component_selector parameter, in [Custom Component Selection](#custom-component-selection) section, for related functionality)
- **External knowledge integration**: Connect to databases, APIs, or knowledge bases during instruction generation

### Available Options

**Built-in Options:**

- **InstructionProposer**: The default (used when `instruction_proposer=None`), configurable as shown above. It uses GEPA's standard reflection prompt, which was used for the diverse experiments reported in the GEPA paper and tutorials, and it sends `dspy.Image` inputs to the reflection LM as structured content.
- **MultiModalInstructionProposer**: An earlier proposer for `dspy.Image` inputs with its own vision-oriented prompt. The default proposer now handles images too, so this class is kept for comparison and may be deprecated.

```python
from dspy.teleprompt.gepa.instruction_proposal import MultiModalInstructionProposer

# A vision-specific prompt for tasks involving images
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5", max_tokens=32000, api_key=api_key),
    instruction_proposer=MultiModalInstructionProposer(),
    auto="medium"
)
```

We invite community contributions of new instruction proposers for specialized domains.

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

#### Basic Example: Character Limit

A character limit needs no custom proposer. The default proposer validates it with Pydantic and allows one compression call when a draft is too long:

```python
from dspy.teleprompt.gepa import InstructionProposer

gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5", max_tokens=32000, api_key=api_key),
    instruction_proposer=InstructionProposer(max_chars=3500),
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
    reflection_lm=dspy.LM(model="gpt-5", max_tokens=32000, api_key=api_key),
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

## Custom Code Proposers

### What is code_proposer?

`code_proposer` lets you supply your own function for rewriting the source code of a [`dspy.Flex`](../../../diving-deeper/flex.md) submodule during GEPA optimization. GEPA calls it each reflection round with the current source and the examples it ran on, and it returns a revised `dspy.Module` class.

DSPy's GEPA adapter sorts the optimizable parts of a program into two kinds of component. A **code component** is a `Flex` submodule, and its optimizable value is a whole `dspy.Module` source. An **instruction component** is any other predictor, and its value is an instruction string. `code_proposer` replaces the default proposer for code components, and `instruction_proposer` replaces it for instruction components. You can set either one without the other.

### The contract

A code proposer is a callable taking five keyword arguments:

| Argument | Meaning |
|---|---|
| `candidate` | A dict keyed by component name. Each value is that component's current value: the module source for a `Flex`, or the instruction string for an ordinary predictor. |
| `reflective_dataset` | A dict keyed by component name. Each value is the list of reflective records for that component from this round's minibatch. Each record is a dict with `Inputs`, `Generated Outputs`, and `Feedback`. |
| `components_to_update` | A list of the component names to rewrite this round, filtered to `Flex` submodules. |
| `task_descriptions` | A dict keyed by component name. Each value is a text rendering of that `Flex`'s signature: its name, objective, and input and output fields. |
| `context_blurbs` | A dict keyed by component name. Each value is a text block listing the tools passed to that `Flex` and the sandbox rules for using them. |

Note that `candidate` carries **every** component, including instruction components you aren't being asked to touch. `reflective_dataset` covers only the components the component selector picked this round, and omits a code component that produced no records. `components_to_update` is the authoritative list; use `reflective_dataset.get(name, [])` rather than indexing.

A code proposer returns a dict keyed by component name. Each value is the **complete replacement source** for that component. The source is one `dspy.Module` subclass that defines `forward`. An `__init__` is optional and is needed only if the module constructs predictors. Do not return a patch or a partial class.

Four things to know:

- **Records are whole-program, not per-predictor.** A `Flex`'s own predictors are part of what gets rewritten, so its reflective records hold the module's inputs, its final prediction, and the metric feedback, under the keys `Inputs`, `Generated Outputs`, and `Feedback`. Every example in the minibatch is included, not just the low-scoring ones; GEPA skips reflection only when the whole minibatch scores perfectly.
- **Strip markdown fences.** Whatever you return is bound as source verbatim. The built-in proposer strips fences from the LM's output; a fenced string returned from yours raises `SyntaxError` when GEPA binds it.
- **You own your failures.** The built-in proposer falls back to the original source when a proposal fails (except LM errors, which propagate). A custom proposer that raises propagates unconditionally — return `candidate[name]` unchanged if you want the same fallback.
- **A bad proposal is safe, just wasteful.** Source that doesn't parse is scored at the failure score and the search continues; it costs a step, not the run.

Your proposer runs inside the `reflection_lm` context, so a bare `dspy.Predict` inside it uses the reflection LM with no extra wiring. To use a different model, wrap your calls in `dspy.context(lm=...)`. Note that `code_proposer` does not by itself satisfy GEPA's reflection-provider requirement: you still need to pass `reflection_lm` (or an `instruction_proposer`).

### Example

````python
import dspy

class ProposeCode(dspy.Signature):
    """Rewrite the module to fix the observed failures."""
    task_description: str = dspy.InputField()
    current_source: str = dspy.InputField()
    failures: str = dspy.InputField()
    revised_source: str = dspy.OutputField(desc="One complete dspy.Module subclass.")

def _unfence(src):
    src = src.strip()
    if src.startswith("```"):          # drop a ```python ... ``` wrapper
        src = src.split("\n", 1)[-1].rsplit("```", 1)[0]
    return src.strip()

def my_code_proposer(*, candidate, reflective_dataset, components_to_update,
                     task_descriptions, context_blurbs):
    propose = dspy.Predict(ProposeCode)
    proposals = {}
    for name in components_to_update:
        failures = "\n\n".join(
            f"Inputs: {r['Inputs']}\nOutputs: {r['Generated Outputs']}\nFeedback: {r['Feedback']}"
            for r in reflective_dataset.get(name, [])
        )
        out = propose(
            task_description=task_descriptions.get(name, name),
            current_source=candidate[name],
            failures=failures,
        )
        proposals[name] = _unfence(out.revised_source)
    return proposals

gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5"),
    code_proposer=my_code_proposer,
    auto="medium",
)
````

## Custom Component Selection

### What is component_selector?

The `component_selector` parameter controls which components (predictors) in your DSPy program are selected for optimization at each GEPA iteration. Instead of the default round-robin approach that updates one component at a time, you can implement custom selection strategies that choose single or multiple components based on optimization state, performance trajectories, and other contextual information.

### Default Behavior

By default, GEPA uses a **round-robin strategy** (`RoundRobinReflectionComponentSelector`) that cycles through components sequentially, optimizing one component per iteration:

```python
# Default round-robin component selection
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5", max_tokens=32000, api_key=api_key),
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
from dspy.teleprompt.gepa import InstructionProposer

# Combined custom selector and configured instruction proposer
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=reflection_lm,
    component_selector=alternating_half_selector,
    instruction_proposer=InstructionProposer(max_chars=2500),
    auto="medium"
)
```
