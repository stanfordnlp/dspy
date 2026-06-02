# Tools, ReAct, and MCP

## Intent

A DSPy program “uses a tool” when an LM picks a named Python function from a list, supplies arguments to it, sees the result, and reasons over the result on the next turn. `dspy.Tool` is the wrapper that makes a plain callable usable this way; `dspy.ReAct` is the module that runs the thought / tool / observation loop; and `dspy.utils.mcp` (with `Tool.from_mcp_tool`) bridges remote MCP servers into the same interface.

Read this when you want to give your program tools, when you want to understand the ReAct loop’s mechanics (trajectory, finish step, truncation), or when you want to attach an MCP server’s tools to a DSPy program.

## Design decisions

### 1. A `Tool` is a Pydantic model over a callable, not a wrapper class

`Tool` is a `dspy.Type` (which is itself a `pydantic.BaseModel`), and `func`, `name`, `desc`, `args`, `arg_types`, and `arg_desc` are model fields. This matters because the same `Tool` instance is what adapters read to render the prompt, what `ToolCall.execute` resolves by name, and what serializes when a program with tools is saved. Being a Pydantic model plugs into all of that without bespoke code.

### 2. Schema, types, and description are introspected by default

The constructor walks `inspect.signature(func)` and `get_type_hints(func)` to build the JSON schema (`args`), the type-hint dict (`arg_types`), and the description (from the docstring). Pydantic models in the type hints are inlined: their JSON schemas replace the `$ref` paths so the LM sees a flat schema. Explicit values you pass to the constructor override inference, field by field — hand-tune one and the rest stay inferred.

### 3. Async vs sync is detected at call time, not at construction

`Tool.__call__` invokes `func(**kwargs)` and checks the result with `asyncio.iscoroutine`. The check lives at call time because the same `func` could be wrapped, decorated, or be a generic callable whose async-ness isn’t visible from the outside. A runtime check is the only signal that always works.

### 4. `allow_tool_async_sync_conversion` is the opt-in bridge

An async tool called from a sync entry point raises by default. Setting `settings.allow_tool_async_sync_conversion=True` (or scoping it via `dspy.context`) makes `__call__` run the coroutine through `asyncio.run` or the active loop’s `run_until_complete`. Opt-in matters because sync-from-async conversion is a deadlock risk inside many runtime environments; consent is explicit.

### 5. ReAct’s loop is thought → tool name → tool args → observation

On each iteration the LM produces three output fields: `next_thought` (free text), `next_tool_name` (a `Literal` over the registered tools), and `next_tool_args` (a `dict`). ReAct looks up the tool, calls it with `next_tool_args` as kwargs, appends four trajectory entries (`thought_i`, `tool_name_i`, `tool_args_i`, `observation_i`), and loops. The `Literal` type on `next_tool_name` means the adapter constrains the LM to known tool names, not free-text guesses.

### 6. The `finish` tool is auto-added and is how the loop terminates

ReAct registers an extra `finish` tool — a no-arg callable — and the loop exits when the LM selects it. Making termination a tool rather than a special-cased flag keeps the LM’s choice uniform: pick from the same list every turn, no extra parsing logic. A separate `max_iters` cap is the hard upper bound for cases where the LM never selects `finish`.

### 7. The trajectory is a dict; the adapter decides how it’s rendered

`ReAct._format_trajectory` builds a throwaway signature with one field per trajectory key, then asks the active adapter to render it via `format_user_message_content`. With `ChatAdapter` the trajectory comes out with `[[ ## thought_i ## ]]` markers; with `JSONAdapter` it comes out as a JSON object; with `XMLAdapter` it comes out as tags. ReAct never hand-formats — the adapter does, and the format stays consistent with the rest of the prompt.

### 8. Tool execution errors become observations, not crashes

A raised exception from a tool is caught inside `forward` and recorded as `observation_i = "Execution error in <tool>: <traceback>"`. The LM sees the error text on the next iteration and can react to it. Surfacing failures into the LM’s reasoning loop, rather than terminating the program, is the default an agent expects — the model is supposed to recover.

### 9. A separate extractor module produces the declared outputs

After the loop ends, ReAct hands the trajectory to a `dspy.ChainOfThought` over a fallback signature that includes the original output fields plus a `trajectory` input. The extractor’s job is to read the trajectory and produce the signature’s declared outputs in their correct types. Decoupling navigation (the loop) from extraction (the typed answer) makes both halves easier to optimize and easier to debug.

### 10. Trajectory truncation drops the oldest tool call on context overflow

When `_call_with_potential_trajectory_truncation` catches a `ContextWindowExceededError`, it calls `truncate_trajectory`, which removes the first four keys (one full tool-call’s worth) and retries — up to three rounds. The heuristic: recent observations are most informative for the next step, older context can go. Override `truncate_trajectory` in a subclass to keep summaries, slide a window, or anything else.

### 11. MCP tools are always async; `Tool.from_mcp_tool` is the canonical bridge

`convert_mcp_tool` builds an async `func` that awaits `session.call_tool(name, args)`. Sync access goes through `acall` or the `allow_tool_async_sync_conversion` switch. MCP is async-only because the underlying `mcp.ClientSession` is built on `anyio` task groups and exposes an async API; bridging it sync would block the I/O loop.

### 12. Native function calling is opt-in on adapters, not on tools

Whether the LM gets a `tools=` payload (OpenAI-style function calling) or a text-rendered prompt is the adapter’s decision, not the tool’s. `Tool.format_as_litellm_function_call()` produces the function-calling descriptor; `ChatAdapter`‘s `use_native_function_calling=True` constructor flag opts in. The split exists because the same `Tool` object should work whether or not the LM supports native tool calling — only the wire format changes.

## API walkthrough

Grouped by what you’re trying to do.

### Wrapping a callable as a tool

**`dspy.Tool(func, name=None, desc=None, args=None, arg_types=None, arg_desc=None)`**
A Pydantic model with `func` as its primary field. The constructor calls `_parse_function`, which walks the callable’s signature with `inspect.signature` and `get_type_hints`, derives the JSON schema with `pydantic.TypeAdapter`, and resolves `$ref` paths into inline schemas via `_resolve_json_schema_reference`. Explicit kwargs take precedence: pass `args={...}` and the inference for `args` is skipped, but `arg_types` and `desc` still come from inspection.

**`Tool.__call__(**kwargs)` / `Tool.acall(**kwargs)`**
Both go through `_validate_and_parse_args`, which JSON-schema-validates each kwarg then wraps it in a one-field Pydantic model to coerce nested types like `list[list[MyModel]]`. `__call__` runs the function and inspects the return: if it’s a coroutine, behavior depends on `settings.allow_tool_async_sync_conversion`. `acall` awaits a coroutine return or returns the synchronous value directly — so the async path tolerates sync tools.

**`Tool.format_as_litellm_function_call()`** → `dict`
Returns the OpenAI/LiteLLM tool descriptor: `{"type": "function", "function": {"name", "description", "parameters": {"type": "object", "properties": args, "required": list(args)}}}`. Every declared arg is marked `required` — Pydantic-level defaults aren’t surfaced as optional in the function descriptor.

**`Tool.from_mcp_tool(session, tool)`** → `Tool`
Class method. Delegates to `dspy.utils.mcp.convert_mcp_tool`. Takes a live `mcp.ClientSession` and an `mcp.types.Tool` and returns an async DSPy tool wired to that session.

**`Tool.from_langchain(tool)`** → `Tool`
Class method. Wraps a LangChain `BaseTool` so it can be passed to DSPy modules that accept a `tools=` list. Implementation lives in `dspy.utils.langchain_tool`.

### The ReAct module

**`dspy.ReAct(signature, tools, max_iters=20)`**
On construction, ReAct coerces every entry in `tools` to a `dspy.Tool`, registers the auto-generated `finish` tool, and builds two internal modules: a `dspy.Predict` over a signature whose outputs are `next_thought`, `next_tool_name` (a `Literal` over the tool names), and `next_tool_args` (a `dict`); and a `dspy.ChainOfThought` extractor whose signature contains the original output fields plus a `trajectory` input. The system instructions are assembled with a numbered tool list and embedded into the predict’s signature.

**`ReAct.forward(**inputs)` / `ReAct.aforward(**inputs)`**
Runs the loop described above. Each iteration formats the trajectory through the current adapter and passes it as the `trajectory` input to the predict; tool exceptions are caught and stringified into the trajectory; the loop exits when the LM picks `finish` or `max_iters` is reached. The extractor then reads the final trajectory and produces the declared outputs. Returns a `dspy.Prediction` carrying both the trajectory and the extracted fields.

**`ReAct.truncate_trajectory(trajectory)`**
Removes the first four keys of the trajectory dict — one complete tool call. Called from `_call_with_potential_trajectory_truncation` after a `ContextWindowExceededError`, with up to three rounds before the module raises. Override to keep summaries, slide a window, or implement any other policy.

### Tool calls in the LM response

When an adapter supports native function calling, the LM’s output contains structured tool-call objects rather than text markers. These types model that surface.

**`dspy.ToolCalls(tool_calls: list[ToolCall])`**
A `pydantic.BaseModel` whose single field is a list of `ToolCall` items. A validator accepts several input shapes — a list of dicts, a `{"tool_calls": [...]}` dict, or a single `{"name", "args"}` dict — so adapters parsing from different providers can route through the same constructor.

**`ToolCalls.ToolCall(name, args)`**
A single named function call with its arguments. Nested class, accessible as `dspy.ToolCalls.ToolCall`.

**`ToolCall.execute(functions=None)`** → `Any`
Looks up the named function in a `{name: func}` dict, a list of `Tool` objects, or — when `functions=None` — the caller’s locals and globals (resolved via `inspect.currentframe().f_back`). Raises `ValueError` when the name isn’t found and `RuntimeError` when the underlying function throws.

**`ToolCalls.from_dict_list(tool_calls_dicts)`** → `ToolCalls`
Constructor helper. Builds a `ToolCalls` from a list of `{"name", "args"}` dicts — the shape adapters produce from a provider’s native response.

### MCP bridge

**`dspy.utils.mcp.convert_mcp_tool(session, tool)`** → `Tool`
Builds a `Tool` whose `func` is an `async def` closure: it awaits `session.call_tool(tool.name, arguments=kwargs)` and unpacks the result. The unpacker pulls text from any `TextContent` entries (returning a single string when there’s one, a list otherwise) and returns non-text entries as-is. An `isError=True` response raises `RuntimeError`. The MCP tool’s `inputSchema` is mapped to DSPy’s `args`/`arg_types`/`arg_desc` via `convert_input_schema_to_tool_args`.

**`dspy.utils.mcp.convert_input_schema_to_tool_args(schema)`** → `(args, arg_types, arg_desc)`
Walks an MCP-style JSON schema and produces the three dicts a `Tool` needs. Resolves `$defs`, picks up `description` fields as `arg_desc`, and translates JSON-schema types to Python types via a `_TYPE_MAPPING` table. Exported in case callers want to write their own bridge against a non-MCP protocol with similar JSON-schema arguments.

## Cross-links

- [Adapters: how signatures become prompts](adapters.md) — where `Tool` and `ToolCalls` are formatted onto the wire, and where native function calling is opted into.
- [Built-in module variants](built-in-module-variants.md) — `CodeAct` and `RLM` use the same tool-wrapping machinery in a code-execution context.
- [RLM: exploring large contexts with code](rlm.md) — how RLM exposes built-in `llm_query` tools and your own tools inside its sandbox.
- [Modules: composing your own](modules.md) — `ReAct` is a `Module`, so trace, history, and `set_lm` all apply.
