# dspy.GEPA: Reflective Prompt Optimizer

**GEPA** (Genetic-Pareto) is a reflective optimizer proposed in "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning" (Agrawal et al., 2025, [arxiv:2507.19457](https://arxiv.org/abs/2507.19457)), that adaptively evolves _textual components_ (such as prompts) of arbitrary systems. In addition to scalar scores returned by metrics, users can also provide GEPA with a text feedback to guide the optimization process. Such textual feedback provides GEPA more visibility into why the system got the score that it did, and then GEPA can introspect to identify how to improve the score. This allows GEPA to propose high performing prompts in very few rollouts.

<!-- START_API_REF -->
::: dspy.GEPA
    handler: python
    options:
        members:
            - compile
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

One of the key insights behind GEPA is its ability to leverage domain-specific textual feedback. Users should provide a feedback function as the GEPA metric, which has the following call signature:
<!-- START_API_REF -->
::: dspy.teleprompt.gepa.gepa.GEPAFeedbackMetric
    handler: python
    options:
        members:
            - __call__
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

When `track_stats=True`, GEPA returns detailed results about all of the proposed candidates, and metadata about the optimization run. The results are available in the `detailed_results` attribute of the optimized program returned by GEPA, and has the following type:
<!-- START_API_REF -->
::: dspy.teleprompt.gepa.gepa.DspyGEPAResult
    handler: python
    options:
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

## Usage Examples

See GEPA usage tutorials in [GEPA Tutorials](../../tutorials/gepa_ai_program/index.md).

### Inference-Time Search

GEPA can act as a test-time/inference search mechanism. By setting your `valset` to your _evaluation batch_ and using `track_best_outputs=True`, GEPA produces for each batch element the highest-scoring outputs found during the evolutionary search.

```python
gepa = dspy.GEPA(metric=metric, track_stats=True, ...)
new_prog = gepa.compile(student, trainset=my_tasks, valset=my_tasks)
highest_score_achieved_per_task = new_prog.detailed_results.highest_score_achieved_per_val_task
best_outputs = new_prog.detailed_results.best_outputs_valset
```

## Custom Instruction Proposers

### What is instruction_proposer?

The `instruction_proposer` is the component responsible for invoking the `reflection_lm` and proposing new prompts during GEPA optimization. When GEPA identifies underperforming components in your DSPy program, the instruction proposer analyzes execution traces, feedback, and failures to generate improved instructions tailored to the observed issues.

### Default Implementation

By default, GEPA uses the built-in instruction proposer from the [GEPA library](https://github.com/gepa-ai/gepa), which implements the `InstructionProposalSignature`. This default proposer:

- Uses a comprehensive prompt template that analyzes task context, inputs, outputs, and feedback
- Extracts domain-specific details and generalizable strategies from execution traces  
- Generates improved instructions within structured code blocks for easy parsing
- Works well for most text-based optimization tasks

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

Consider implementing a custom instruction proposer when you need:

- **Multi-modal handling**: Process images (dspy.Image) alongside textual information in your inputs
- **Nuanced control on limits and length constraints**: Have more fine-grained control over instruction length, format, and structural requirements
- **Domain-specific information**: Inject specialized knowledge, terminology, or context that the default proposer lacks
- **Provider-specific prompting guides**: Optimize instructions for specific LLM providers (OpenAI, Anthropic, etc.) with their unique formatting preferences
- **Coupled component updates**: Handle situations where 2 or more components need to be updated together in a coordinated manner, rather than optimizing each component independently
- **External knowledge integration**: Connect to databases, APIs, or knowledge bases during instruction generation

### Available Options

**Built-in Options:**
- **Default Proposer**: The standard GEPA instruction proposer (used when `instruction_proposer=None`). The default instruction proposer IS an instruction proposer as well! It is the most general one, that was used for the diverse experiments reported in the GEPA paper and tutorials.
- **MultiModalInstructionProposer**: Handles `dspy.Image` inputs and structured multimodal content. This proposer has been specifically optimized for tasks that include one or more `dspy.Image` inputs.

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

We invite community contributions of new instruction proposers for specialized domains as the [GEPA library](https://github.com/gepa-ai/gepa) continues to grow. The GEPA repository contains multiple adapters (like `DefaultAdapter`, `DSPyFullProgramAdapter`, `TerminalBenchAdapter`, etc.) that, while not instruction proposers themselves, provide good examples and inspiration for implementing your own instruction proposer.

### How to Implement Custom Proposer

Custom instruction proposers must implement the `ProposalFn` protocol:

```python
def __call__(
    self,
    candidate: dict[str, str],           # Current component name -> instruction mapping
    reflective_dataset: dict[str, list[dict[str, Any]]],  # Component -> failed examples  
    components_to_update: list[str]      # Which components to improve
) -> dict[str, str]:                     # Updated component name -> new instruction mapping
```

#### Basic Example: Word Limit Proposer

```python
import dspy
from gepa.core.adapter import ProposalFn

class GenerateWordLimitedInstruction(dspy.Signature):
    """Given a current instruction and feedback examples, generate an improved instruction with word limit constraints."""
    
    current_instruction = dspy.InputField(desc="The current instruction that needs improvement")
    issues_found = dspy.InputField(desc="Feedback and issues identified from failed examples")
    max_words = dspy.InputField(desc="Maximum number of words allowed in the new instruction")
    
    improved_instruction = dspy.OutputField(desc="A new instruction that fixes the issues while staying under the word limit")

class WordLimitedInstructionImprover(dspy.Module):
    """Module for improving instructions with word limit constraints using dspy.Refine."""
    
    def __init__(self, max_words: int = 1000):
        super().__init__()
        self.max_words = max_words
        self.base_improver = dspy.Predict(GenerateWordLimitedInstruction)
        
        # Define reward function that checks word count
        def word_limit_reward(args, pred):
            word_count = len(pred.improved_instruction.split())
            return 1.0 if word_count <= self.max_words else 0.0
        
        # Use dspy.Refine to try up to 3 times to meet word limit
        self.improver = dspy.Refine(
            module=self.base_improver,
            N=3,
            reward_fn=word_limit_reward,
            threshold=1.0
        )
    
    def forward(self, current_instruction: str, issues_found: str):
        """Generate improved instruction within word limit."""
        return self.improver(
            current_instruction=current_instruction,
            issues_found=issues_found,
            max_words=str(self.max_words)
        )

class WordLimitProposer(ProposalFn):
    def __init__(self, max_words: int = 1000):
        self.max_words = max_words
        self.instruction_improver = WordLimitedInstructionImprover(max_words=max_words)
    
    def __call__(self, candidate, reflective_dataset, components_to_update):
        updated_components = {}
        
        for component_name in components_to_update:
            if component_name not in candidate or component_name not in reflective_dataset:
                continue
                
            current_instruction = candidate[component_name]
            component_examples = reflective_dataset[component_name]
            
            # Create feedback summary
            feedback_text = "\n".join([
                f"Example {i+1}: {ex.get('Feedback', 'No feedback')}"
                for i, ex in enumerate(component_examples[:3])  # Limit examples to prevent context overflow
            ])
            
            # Use the module to improve the instruction
            result = self.instruction_improver(
                current_instruction=current_instruction,
                issues_found=feedback_text
            )
            
            updated_components[component_name] = result.improved_instruction
            
        return updated_components

# Usage
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key),
    instruction_proposer=WordLimitProposer(max_words=30),
    auto="medium"
)
```

#### Advanced Example: RAG-Enhanced Instruction Proposer

```python
import dspy
from gepa.core.adapter import ProposalFn

class GenerateDocumentationQuery(dspy.Signature):
    """Analyze examples with feedback to identify common issue patterns and generate targeted database queries for retrieving relevant documentation.
    
    Your goal is to search a document database for guidelines that address the problematic patterns found in the examples. Look for recurring issues, error types, or failure modes in the feedback, then craft specific search queries that will find documentation to help resolve these patterns."""
    
    current_instruction = dspy.InputField(desc="The current instruction that needs improvement")
    examples_with_feedback = dspy.InputField(desc="Examples with their feedback showing what issues occurred and any recurring patterns")

    failure_patterns: str = dspy.OutputField(desc="Summarize the common failure patterns identified in the examples")
    
    retrieval_queries: list[str] = dspy.OutputField(desc="Specific search query to find relevant documentation in the database that addresses the common issue patterns identified in the problematic examples")

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
        
        # Retrieve relevant documentation using multiple queries
        all_retrieved_docs = []
        for query in query_result.retrieval_queries:
            retrieved_docs = self.retrieve(query)
            if hasattr(retrieved_docs, 'passages'):
                all_retrieved_docs.extend(retrieved_docs.passages)
            else:
                all_retrieved_docs.append(str(retrieved_docs))
        
        # Format retrieved documentation
        relevant_docs = "\n\n".join([
            f"Guideline {i+1}: {doc}" 
            for i, doc in enumerate(all_retrieved_docs)
        ])
        
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
    
    def __call__(self, candidate, reflective_dataset, components_to_update):
        updated_components = {}
        
        for component_name in components_to_update:
            if component_name not in candidate or component_name not in reflective_dataset:
                continue
                
            current_instruction = candidate[component_name]
            component_examples = reflective_dataset[component_name]
            
            # Use RAG-enhanced improvement
            result = self.instruction_improver(
                current_instruction=current_instruction,
                component_examples=component_examples
            )
            
            updated_components[component_name] = result.improved_instruction
            
        return updated_components

# Usage - Setup specialized documentation retriever
import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM

# Setup retriever for your specialized documentation
doc_retriever = ChromadbRM(
    collection_name="instruction_guidelines",
    persist_directory="./specialized_docs_db",
    k=5  # Retrieve top 5 relevant documents
)

# Configure the RAG-enhanced proposer
gepa = dspy.GEPA(
    metric=task_specific_metric,
    reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key),
    instruction_proposer=DocumentationEnhancedProposer(doc_retriever),
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

# Can work without GEPA's reflection_lm!
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=None,  # Optional when using custom proposer
    instruction_proposer=ExternalLMProposer(),
    auto="medium"
)
```

**Best Practices:**
- **Use the full power of DSPy**: Leverage DSPy components like `dspy.Module`, `dspy.Signature`, and `dspy.Predict` to create your instruction proposer rather than direct LM calls. Consider `dspy.Refine` for constraint satisfaction, `dspy.ChainOfThought` for complex reasoning tasks, and compose multiple modules for sophisticated instruction improvement workflows
- **Validate component existence**: Always validate that required components exist in `candidate` and `reflective_dataset`  
- **Enable holistic feedback analysis**: While dspy.GEPA's `GEPAFeedbackMetric` processes one (gold, prediction) pair at a time, instruction proposers receive all examples for a component in batch, enabling cross-example pattern detection and systematic issue identification.
- **Mind data serialization**: Serializing everything to strings might not be ideal - handle complex input types (like `dspy.Image`) by maintaining their structure for better LM processing
- **Create detailed signatures**: Write comprehensive signature docstrings with step-by-step analysis guidance and clear requirements for the instruction improvement task
- **Test thoroughly**: Test your custom proposer with representative failure cases
- **Control context size**: Mind the number of examples analyzed to prevent context overflow

## How Does GEPA Work?

### 1. **Reflective Prompt Mutation**

GEPA uses LLMs to _reflect_ on structured execution traces (inputs, outputs, failures, feedback), targeting a chosen module and proposing a new instruction/program text tailored to real observed failures and rich textual/environmental feedback.

### 2. **Rich Textual Feedback as Optimization Signal**

GEPA can leverage _any_ textual feedback available—not just scalar rewards. This includes evaluation logs, code traces, failed parses, constraint violations, error message strings, or even isolated submodule-specific feedback. This allows actionable, domain-aware optimization. 

### 3. **Pareto-based Candidate Selection**

Rather than evolving just the _best_ global candidate (which leads to local optima or stagnation), GEPA maintains a Pareto frontier: the set of candidates which achieve the highest score on at least one evaluation instance. In each iteration, the next candidate to mutate is sampled (with probability proportional to coverage) from this frontier, guaranteeing both exploration and robust retention of complementary strategies.

### Algorithm Summary

1. **Initialize** the candidate pool with the the unoptimized program.
2. **Iterate**:
   - **Sample a candidate** (from Pareto frontier).
   - **Sample a minibatch** from the train set.
   - **Collect execution traces + feedbacks** for module rollout on minibatch.
   - **Select a module** of the candidate for targeted improvement.
   - **LLM Reflection:** Propose a new instruction/prompt for the targeted module using reflective meta-prompting and the gathered feedback.
   - **Roll out the new candidate** on the minibatch; **if improved, evaluate on Pareto validation set**.
   - **Update the candidate pool/Pareto frontier.**
   - **[Optionally] System-aware merge/crossover**: Combine best-performing modules from distinct lineages.
3. **Continue** until rollout or metric budget is exhausted. 
4. **Return** candidate with best aggregate performance on validation.

## Implementing Feedback Metrics

A well-designed metric is central to GEPA's sample efficiency and learning signal richness. GEPA expects the metric to returns a `dspy.Prediction(score=..., feedback=...)`. GEPA leverages natural language traces from LLM-based workflows for optimization, preserving intermediate trajectories and errors in plain text rather than reducing them to numerical rewards. This mirrors human diagnostic processes, enabling clearer identification of system behaviors and bottlenecks.

Practical Recipe for GEPA-Friendly Feedback:

- **Leverage Existing Artifacts**: Use logs, unit tests, evaluation scripts, and profiler outputs; surfacing these often suffices.
- **Decompose Outcomes**: Break scores into per-objective components (e.g., correctness, latency, cost, safety) and attribute errors to steps.
- **Expose Trajectories**: Label pipeline stages, reporting pass/fail with salient errors (e.g., in code generation pipelines).
- **Ground in Checks**: Employ automatic validators (unit tests, schemas, simulators) or LLM-as-a-judge for non-verifiable tasks (as in PUPA).
- **Prioritize Clarity**: Focus on error coverage and decision points over technical complexity.

### Examples

- **Document Retrieval** (e.g., HotpotQA): List correctly retrieved, incorrect, or missed documents, beyond mere Recall/F1 scores.
- **Multi-Objective Tasks** (e.g., PUPA): Decompose aggregate scores to reveal contributions from each objective, highlighting tradeoffs (e.g., quality vs. privacy).
- **Stacked Pipelines** (e.g., code generation: parse → compile → run → profile → evaluate): Expose stage-specific failures; natural-language traces often suffice for LLM self-correction.

## Further Reading

- [GEPA Paper: arxiv:2507.19457](https://arxiv.org/abs/2507.19457)
- [GEPA Github](https://github.com/gepa-ai/gepa) - This repository provides the core GEPA evolution pipeline used by `dspy.GEPA` optimizer.
- [DSPy Tutorials](../../tutorials/gepa_ai_program/index.md)
