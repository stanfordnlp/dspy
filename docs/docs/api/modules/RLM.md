# dspy.RLM

`RLM` (Recursive Language Model) is a DSPy module that lets LLMs programmatically explore large contexts through a sandboxed Python REPL. Instead of feeding huge contexts directly into the prompt, RLM treats context as external data that the LLM examines via code execution and recursive sub-LLM calls.

This implements the approach described in ["Recursive Language Models" (Zhang, Kraska, Khattab, 2025)](https://arxiv.org/abs/2512.24601).

## When to Use RLM

As contexts grow, LLM performance degrades — a phenomenon known as [context rot](https://research.trychroma.com/context-rot). RLMs address this by separating the _variable space_ (information stored in the REPL) from the _token space_ (what the LLM actually processes). The LLM dynamically loads only the context it needs, when it needs it.

Use RLM when:

- Your context is **too large** to fit in the LLM's context window effectively
- The task benefits from **programmatic exploration** (searching, filtering, aggregating, chunking)
- You need the LLM to decide **how to decompose** the problem, not you

## Basic Usage

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-5"))

# Create an RLM module
rlm = dspy.RLM("context, query -> answer")

# Call it like any other module
result = rlm(
    context="...very long document or data...",
    query="What is the total revenue mentioned?"
)
print(result.answer)
```

## Deno Installation

RLM relies on [Deno](https://deno.land/) and (Pyodide)[tocite] to create a WASM sandbox locally that 

You can install Deno with: `curl -fsSL https://deno.land/install.sh | sh` on macOS and Linux. See the [Deno Installation Docs](https://docs.deno.com/runtime/getting_started/installation/) for more details. Make sure to accept the prompt when it asks to add it to your shell profile. 

After you have installed Deno, **Make sure to restart your shell**

Then you can run dspy.RLM.

User's have reported issues with the Deno cache not being found by DSPy. We are actively investigating these issues, and your feedback is greatly appreciated.

You can also work with an external sandbox provider. We are still working on creating an example of using external sandbox providers.


## How It Works

RLM operates in an iterative REPL loop:

1. The LLM receives **metadata** about the context (type, length, preview) but not the full context
2. The LLM writes **Python code** to explore the data (print samples, search, filter)
3. Code executes in a **sandboxed interpreter**, and the LLM sees the output
4. The LLM can call `llm_query(prompt)` to run **sub-LLM calls** for semantic analysis on snippets
5. When done, the LLM calls `SUBMIT(output)` to return the final answer

#### What the LLM sees (step-by-step trace):

##### Step 1: Initial Metadata (no direct access to full context)
```python
# Step 1: Peek at the data
print(context[:2000])
```
_Output shown to the LLM:_
```
[Preview of the first 2000 characters of the document]
```

##### Step 2: Write Code to Explore Context
```python
# Step 2: Search for relevant sections
import re
matches = re.findall(r'revenue.*?\$[\d,]+', context, re.IGNORECASE)
print(matches)
```
_Output shown to the LLM:_
```
['Revenue in Q4: $5,000,000', 'Total revenue: $20,000,000']
```

##### Step 3: Trigger Sub-LLM Calls
```python
# Step 3: Use sub-LLM for semantic extraction
result = llm_query(f"Extract the total revenue from: {matches[0]}")
print(result)
```
_Output shown to the LLM:_
```
$5,000,000
```

##### Step 4: Submit Final Answer
```python
# Step 4: Return final answer
SUBMIT(result)
```
_Output shown to the user:_
```
$5,000,000
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| Signature` | required | Defines inputs and outputs (e.g., `"context, query -> answer"`) |
| `max_iterations` | `int` | `20` | Maximum REPL interaction loops before fallback extraction |
| `max_llm_calls` | `int` | `50` | Maximum `llm_query`/`llm_query_batched` calls per execution |
| `max_output_chars` | `int` | `100_000` | Maximum characters to include from REPL output |
| `verbose` | `bool` | `False` | Log detailed execution info |
| `tools` | `dict[str, Callable]` | `None` | Additional tool functions callable from interpreter code |
| `sub_lm` | `dspy.LM` | `None` | LM for sub-queries. Defaults to `dspy.settings.lm`. Use a cheaper model here. |
| `interpreter` | `CodeInterpreter` | `None` | Custom interpreter. Defaults to `PythonInterpreter` (Deno/Pyodide WASM). |

## Built-in Tools

Inside the REPL, the LLM has access to:

| Tool | Description |
|------|-------------|
| `llm_query(prompt)` | Query a sub-LLM for semantic analysis (~500K char capacity) |
| `llm_query_batched(prompts)` | Query multiple prompts concurrently (faster for batch operations) |
| `print()` | Print output (required to see results) |
| `SUBMIT(...)` | Submit final output and end execution |
| Standard library | `re`, `json`, `collections`, `math`, etc. |

## Examples

### Long Document Q&A

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-5"))

rlm = dspy.RLM("document, question -> answer", max_iterations=10)

with open("large_report.txt") as f:
    document = f.read()  # 500K+ characters

result = rlm(
    document=document,
    question="What were the key findings from Q3?"
)
print(result.answer)
```

### Using a Cheaper Sub-LM

```python
import dspy

main_lm = dspy.LM("openai/gpt-5")
cheap_lm = dspy.LM("openai/gpt-5-nano")

dspy.configure(lm=main_lm)

# Root LM (gpt-4o) decides strategy; sub-LM (gpt-4o-mini) handles extraction
rlm = dspy.RLM("data, query -> summary", sub_lm=cheap_lm)
```

### Multiple Typed Outputs

```python
rlm = dspy.RLM("logs -> error_count: int, critical_errors: list[str]")

result = rlm(logs=server_logs)
print(f"Found {result.error_count} errors")
print(f"Critical: {result.critical_errors}")
```

### Custom Tools

```python
def fetch_metadata(doc_id: str) -> str:
    """Fetch metadata for a document ID."""
    return database.get_metadata(doc_id)

rlm = dspy.RLM(
    "documents, query -> answer",
    tools={"fetch_metadata": fetch_metadata}
)
```

### Async Execution

```python
import asyncio

rlm = dspy.RLM("context, query -> answer")

async def process():
    result = await rlm.aforward(context=data, query="Summarize this")
    return result.answer

answer = asyncio.run(process())
```

### Inspecting the Trajectory

```python
result = rlm(context=data, query="Find the magic number")

# See what code the LLM executed
for step in result.trajectory:
    print(f"Code:\n{step['code']}")
    print(f"Output:\n{step['output']}\n")
```

## Output

RLM returns a `Prediction` with:

- **Output fields** from your signature (e.g., `result.answer`)
- **`trajectory`**: List of dicts with `reasoning`, `code`, `output` for each step
- **`final_reasoning`**: The LLM's reasoning on the final step

## Notes

!!! warning "Experimental"
    RLM is marked as experimental. The API may change in future releases.

!!! note "Thread Safety"
    RLM instances are not thread-safe when using a custom interpreter. Create separate instances for concurrent use, or use the default `PythonInterpreter` which creates a fresh instance per `forward()` call.

!!! note "Interpreter Requirements"
    The default `PythonInterpreter` requires [Deno](https://deno.land/) to be installed for the Pyodide WASM sandbox.

## API Reference

<!-- START_API_REF -->
::: dspy.RLM
    handler: python
    options:
        members:
            - __init__
            - __call__
            - forward
            - aforward
            - tools
            - batch
            - deepcopy
            - dump_state
            - get_lm
            - load
            - load_state
            - named_parameters
            - named_predictors
            - parameters
            - predictors
            - reset_copy
            - save
            - set_lm
        show_source: true
        show_root_heading: true
        heading_level: 3
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
:::
<!-- END_API_REF -->
