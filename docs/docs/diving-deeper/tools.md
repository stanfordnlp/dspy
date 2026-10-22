# Tools and MCP

## Intent

A tool lets a DSPy program call Python code chosen by a language model. The model sees a name, description, and argument schema; it selects the tool and supplies arguments; DSPy executes the callable and makes the result available to the program.

Tools are a shared DSPy primitive, not a ReAct-specific feature. `dspy.ReAct` and `dspy.ReActV2` use them in agent loops, `dspy.RLM` exposes them inside its interpreter, `dspy.Flex` can wire them into optimized programs, and adapters use the same tool schemas for native provider function calling. `dspy.utils.mcp` bridges remote MCP tools into this interface.

Read this page when you want to define, wrap, validate, or import tools. For the agent loop that decides when to call them, see [ReAct and ReActV2](react.md).

## Defining a tool

A DSPy tool can start as an ordinary typed Python function:

```python
def search(query: str, limit: int = 5) -> list[str]:
    """Search the knowledge base for relevant passages."""
    return index.search(query, limit=limit)
```

Modules that accept `tools=` convert callables to `dspy.Tool` automatically:

```python
agent = dspy.ReAct("question -> answer", tools=[search])
```

Wrap the function explicitly when you need to inspect or override its metadata:

```python
search_tool = dspy.Tool(
    search,
    name="search_docs",
    desc="Search the product documentation.",
)
```

Good names, descriptions, parameter names, and type hints matter: they are the interface the model uses to decide whether and how to call the function.

## How `dspy.Tool` works

### Schema and metadata are inferred

`dspy.Tool` stores the callable in `func` and derives `name`, `desc`, `args`, `arg_types`, and `arg_desc` from the function signature, type hints, and docstring. Pydantic annotations are converted to JSON schema and local `$ref` paths are resolved so the model receives a complete argument shape. Explicit constructor values override inference field by field.

### Arguments are validated before execution

`Tool.__call__(**kwargs)` and `Tool.acall(**kwargs)` validate arguments against the JSON schema and use Pydantic to coerce nested annotated values. The synchronous path calls the function directly. The asynchronous path awaits coroutine results and also accepts ordinary synchronous functions.

Calling an asynchronous tool from synchronous code raises by default. Opt into conversion only when your runtime permits it:

```python
async_search_tool = dspy.Tool(async_search)

with dspy.context(allow_tool_async_sync_conversion=True):
    result = async_search_tool(query="DSPy")
```

The conversion is explicit because driving an async tool from an existing event loop can deadlock in some environments.

### Adapters choose text or native formatting

The tool itself is provider-neutral. An adapter decides whether to render its schema into text or send it through a provider's native function-calling API. For example:

```python
adapter = dspy.ChatAdapter(use_native_function_calling=True)

with dspy.context(adapter=adapter):
    result = agent(question="What changed in DSPy 3.3?")
```

When native calling is active and the LM supports it, the adapter converts each tool with `Tool.format_as_litellm_function_call()` and sends the resulting descriptors in the LM request. Otherwise, it keeps tool selection in DSPy's normal adapter-formatted fields. The same `Tool` works in either mode.

## Structured tool calls and results

`dspy.ToolCalls` represents model-requested calls independently of any provider's wire format. Each `ToolCalls.ToolCall` carries an optional provider call ID, a tool name, and an argument dictionary:

```python
calls = dspy.ToolCalls.from_dict_list([
    {"name": "search", "args": {"query": "DSPy tools"}},
])
```

The validator accepts DSPy's `{name, args}` shape and common provider-style function-call shapes. Adapters handle conversion at the provider boundary; application code can keep using the DSPy representation.

Tool results are paired to calls by ID, name, value, and an error flag in `ToolCallResults`. This pairing lets adapters replay a native assistant tool call followed by the matching provider `tool` message. It also lets an agent report an unknown tool or execution exception to the model as a result instead of losing the failed turn.

## MCP tools

MCP servers publish tools with JSON schemas. `dspy.Tool.from_mcp_tool(session, tool)` is the canonical bridge from a live `mcp.ClientSession` and an MCP tool definition to a DSPy tool:

```python
dspy_tool = dspy.Tool.from_mcp_tool(session, mcp_tool)
```

The bridge:

1. Converts the MCP input schema into DSPy's `args`, `arg_types`, and `arg_desc`.
2. Creates an async callable that invokes `session.call_tool(...)`.
3. Unpacks MCP text content into a string or list and preserves non-text content.
4. Raises an execution error when the MCP response has `isError=True`.

The bridge supports both the camelCase result fields in MCP SDK v1 and their snake_case replacements in v2 without changing text or non-text result behavior. Pass `result_mode="structured"` to return `structuredContent` when available, falling back to the default conversion when it is absent.

MCP tools are asynchronous because `mcp.ClientSession` is asynchronous. Use a module's async entry point, such as `acall`, or explicitly enable async-to-sync conversion when appropriate.

`Tool.from_langchain(tool)` provides the equivalent bridge for LangChain `BaseTool` objects.

## API walkthrough

**`dspy.Tool(func, name=None, desc=None, args=None, arg_types=None, arg_desc=None)`**
Wraps a callable and infers any metadata not supplied explicitly.

**`Tool.__call__(**kwargs)` / `Tool.acall(**kwargs)`**
Validates, coerces, and executes tool arguments through synchronous or asynchronous entry points.

**`Tool.format_as_litellm_function_call()`** → `dict`
Returns the OpenAI/LiteLLM-style function descriptor used by adapters for native calling.

**`Tool.from_mcp_tool(session, tool, *, result_mode="text")`** → `Tool`
Wraps a remote MCP tool as an asynchronous DSPy tool. Set `result_mode="structured"` to return structured MCP results when available.

**`Tool.from_langchain(tool)`** → `Tool`
Wraps a LangChain tool in the same DSPy interface.

**`dspy.ToolCalls(tool_calls=[...])`**
Stores one or more requested calls in a provider-neutral form.

## Cross-links

- [ReAct and ReActV2](react.md) — how DSPy's agent modules choose, execute, record, and replay tools.
- [Adapters: how signatures become prompts](adapters.md) — where tool schemas and `ToolCalls` become provider requests and responses.
- [RLM: exploring large contexts with code](rlm.md) — how RLM makes supplied tools callable inside its sandbox.
- [Built-in module variants](built-in-module-variants.md) — other modules that consume tools.
- [`dspy.Tool` API reference](../api/primitives/Tool.md)
- [`dspy.ToolCalls` API reference](../api/primitives/ToolCalls.md)
