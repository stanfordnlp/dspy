# ReAct

## Intent

ReAct is DSPy's general-purpose tool-using agent loop: the language model reasons about a task, chooses tools, observes their results, and repeats until it can produce the signature's outputs.

In DSPy 3.5, `dspy.ReAct` uses structured `dspy.History`, typed output submission, and synchronous or asynchronous execution. A model turn can request multiple tool calls, which execute sequentially by default. Async callers can opt into bounded concurrency. `dspy.ReActV2` remains a deprecated compatibility subclass that warns on construction and will be removed in DSPy 3.6.

Read this page to understand how ReAct stores and formats history, calls tools, finishes a run, and interacts with provider prompt caching. For changes from DSPy 3.4 and earlier, see [Migration to DSPy 3.5](#migration-to-dspy-35). For defining tools or importing them from MCP, see [Tools and MCP](tools.md).

## Getting started

The constructor accepts a task signature, tool list, and iteration limit:

```python
import dspy

def lookup(query: str) -> str:
    """Look up information relevant to a query."""
    return search_index(query)

agent = dspy.ReAct("question -> answer", tools=[lookup], max_iters=10)
result = agent(question="What is DSPy?")
```

## Three use cases for the new loop

| Use case | ReAct in DSPy 3.4 and earlier | ReAct in DSPy 3.5 |
| --- | --- | --- |
| Return a typed invoice total, such as `invoice -> total: float` | The agent calls `finish`, then a separate extractor produces the final fields. | The agent calls `submit(total=125.50)` directly. Successful submission returns the typed value without another extraction call. |
| Compare prices from two independent lookup tools | Each turn selects one tool, so two lookups require two model turns. | A capable model can request both lookups in one turn. ReAct runs them sequentially and preserves each result's call ID for the comparison. |
| Investigate a document over several turns, then ask a follow-up | Prior actions are reformatted into a trajectory input; continuation requires application-specific handling. | Pass `prediction.history` into the next call. Native assistant/tool messages retain their IDs and form reusable request prefixes for provider prompt caching. |

These are execution and API differences, not accuracy or latency benchmarks. Multiple calls per turn depend on model support and choices; provider cache hits depend on the provider's caching policy.

## Structured-history execution

### How history is stored

ReAct owns a `dspy.History`, whose `messages` list contains one structured event per loop turn. An event can contain:

- Original signature inputs, included on the first new turn only.
- `next_thought`, when the model returns one.
- A `dspy.ToolCalls` object containing every requested call and its call ID.
- Attached `ToolCallResults`, pairing each ID with the tool name, value, and error status.
- The signature's final output fields when the turn successfully calls `submit`.

Conceptually, a turn looks like:

```python
{
    "question": "What is DSPy?",       # first new turn only
    "next_thought": "I should look it up.",
    "tool_calls": dspy.ToolCalls(
        tool_calls=[
            dspy.ToolCalls.ToolCall(
                id="call_123",
                name="lookup",
                args={"query": "DSPy"},
            )
        ],
        tool_call_results=...,
    ),
}
```

The returned prediction exposes this object as `prediction.history`. Pass that `dspy.History`, or its serialized representation, back as `history=` to continue from an earlier run.

### How history is formatted for the model

The adapter turns each history event into multiple model messages rather than flattening all prior activity into one trajectory field.

With native function calling enabled, a completed turn is replayed as:

```text
user       original input fields, when present
assistant  next_thought plus native tool_calls
tool       one result message per call, matched by tool_call_id
```

Call IDs are preserved from the provider. ReAct generates deterministic IDs when non-native model output does not supply them.

With native function calling disabled, history remains structured inside DSPy, but the adapter renders the assistant fields in its normal text/JSON/XML format and renders tool results as a following user message. This lets ReAct work with models that do not expose a native tool API while retaining one internal history representation.

Native calling is an adapter setting:

```python
adapter = dspy.ChatAdapter(
    use_native_function_calling=True,
    parallel_tool_calls=True,
)

with dspy.context(adapter=adapter):
    result = agent(question="Compare two releases.")
```

Provider support varies. `JSONAdapter` enables native function calling by default; `ChatAdapter` requires the explicit setting shown above.

### How tools are called

The internal predictor returns `dspy.ToolCalls`, so one model turn can request multiple tools. ReAct preserves each requested call/result pair by ID. The `parallel_tool_calls` adapter option asks a capable provider to generate multiple independent calls in the same turn. ReAct executes the returned calls sequentially by default; `max_tool_concurrency` separately controls execution in the async loop.

Callables supplied to the constructor are converted to `dspy.Tool`. Unknown tool names and execution exceptions become error results in history so the next model turn can respond to them.

ReAct reserves `submit` as an internal tool. Its argument schema is generated from the original signature's output fields, so final values travel through the same structured calling path as every other action.

### What happens after each turn and at termination

After executing the calls in a turn, ReAct attaches their results to `ToolCalls` and appends the event to history. If a call successfully invokes `submit`, ReAct returns its typed fields with `termination_reason="submit"`; there is no separate extraction module. Otherwise, it invokes the model again with no duplicate copy of the original inputs.

If the loop reaches `max_iters`, returns no calls, fails to parse, or exceeds the context window, ReAct makes one final LM call requesting `submit`. With native function calling enabled, it sets `tool_choice` to `submit`, so the provider enforces that choice. In non-native mode, the adapter removes `tool_choice` and requests `submit` through the formatted prompt instead. A successful fallback returns `termination_reason="forced_submit"`. If forced submission fails, ReAct raises rather than returning an incomplete prediction without the declared output fields.

ReAct does not automatically truncate old history events on context overflow.

### Async execution

Use `await agent.acall(...)` for asynchronous execution. Tool calls execute sequentially by default. For independent async tools, opt into bounded concurrency:

```python
agent = dspy.ReAct("question -> answer", tools=[lookup], max_tool_concurrency=4)
result = await agent.acall(question="Compare the independent results.")
```

`max_tool_concurrency` must be a positive integer and defaults to `1`. It bounds in-flight tool calls within each model turn, while returned history retains the model's requested order. Ordinary tool errors remain observations. Cancelling the run cancels and awaits active and queued tool tasks before returning control.

Only enable concurrency for tools that can safely run independently: calls may overlap and share application resources. Batches containing `submit` remain sequential, and synchronous `agent(...)` execution is unchanged. Synchronous tool bodies are not moved to worker threads and can still block the event loop.

The adapter's `parallel_tool_calls` option asks the provider to generate multiple calls; it does not execute them concurrently. Conversely, `max_tool_concurrency` does not make the provider generate more calls.

### Turn budget

ReAct always tells the model how many turns it has. Every LM call contains a `turn_budget` input that details the current turn and how many turns remain before the model must submit. Its value reads, for example, `Turn 3 of 10 (7 more after this one).`. The count follows the effective `max_iters`, including a per-call override. The forced final call shows ``No turns remain. Call `submit` now with your final outputs.``

Showing an agent its remaining budget follows [Budget-Aware Tool-Use Enables Effective Agent Scaling](https://arxiv.org/abs/2511.17006), where a per-step budget status let ReAct agents keep improving with larger tool budgets at lower cost.

### Prompt-caching behavior

Structured history makes each completed turn an append-only message group. On the next request, the system instructions and all earlier user/assistant/tool messages remain a stable prefix; only the newest result and request are appended. Providers with prompt caching can therefore reuse more of the prior request instead of reprocessing one ever-growing, newly formatted trajectory value.

For Anthropic models, request provider-side prompt caching with lm15's history-prefix policy:

```python
lm = dspy.LM(
    "anthropic/claude-sonnet-4-5-20250929",
    prompt_cache=dspy.lm15.CacheConfig(prefix="history"),
)

with dspy.context(lm=lm):
    result = agent(question="What is DSPy?")
```

The history policy asks lm15 to mark the reusable conversation prefix as ReAct appends history. It does not enable DSPy's answer cache or guarantee a provider cache hit. Anthropic only caches prefixes that meet the model's minimum token count, and its default ephemeral cache has a limited lifetime. See [Using Provider-Side Prompt Caching](../tutorials/cache/index.md#using-provider-side-prompt-caching) for native configuration and the optional LiteLLM engine's provider-specific injection options.

Native tool replay gives the cleanest provider-visible structure, but non-native mode still benefits from stable multi-turn history. Internal testing has seen cost reductions of up to 50% on some tasks. Actual savings depend on the provider's cache policy, model, request shape, and the size of tool results; they are not guaranteed for every workload.

## Migration to DSPy 3.5

The structured-history implementation previously named `ReActV2` is now canonical `dspy.ReAct`. Existing constructor calls remain valid; the optional keyword-only `max_tool_concurrency` defaults to `1`.

For users of ReAct in DSPy 3.4 and earlier:

| Previous behavior | DSPy 3.5 migration |
| --- | --- |
| `prediction.trajectory` dictionary | Inspect `prediction.history.messages`; pass `history=` to continue a run. |
| `next_tool_name` and `next_tool_args`, one call per turn | Structured `dspy.ToolCalls`, potentially multiple sequential calls per turn. Treat `max_iters` as a turn budget, not a tool-call budget. |
| `finish` followed by the `extract` predictor | Internal typed `submit` tool; no separate extractor or automatic extractor `reasoning` field. Declare any reasoning you need in your output signature. |
| `truncate_trajectory` overrides | Removed; there is no automatic history truncation on context overflow. Review application-level context management. |
| Best-effort loop completion | A final forced `submit` is attempted when needed; failure raises instead of returning incomplete outputs. |

The input names `history`, `tools`, and `turn_budget`, the output names `history` and `termination_reason`, and the tool name `submit` are reserved. Rename any conflicting fields or tools before upgrading.

Old optimized state must be **recompiled** for the new predictor schema. Do not load old trajectory/extractor predictor state into the new ReAct and assume it remains compatible; rerun your optimizer against the new module.

For users of the experimental name, replace `dspy.ReActV2(...)` with `dspy.ReAct(...)`. In 3.5, `dspy.ReActV2` is a deprecated compatibility subclass with the same behavior; it warns on construction and will be removed in 3.6.

## API summary

**`dspy.ReAct(signature, tools, max_iters=20, *, max_tool_concurrency=1)`**
Runs the structured-history loop and reserves `submit` for final typed outputs.

**`ReAct.forward(**inputs)` / `ReAct.aforward(**inputs)`**
Accepts optional continuation `history=` and an integer `max_iters=` override alongside the signature inputs. Returns typed outputs, `history`, and `termination_reason` when submission succeeds; raises if forced submission fails. Use `agent(...)` or `await agent.acall(...)` for normal module invocation.

**`dspy.ReActV2(signature, tools, max_iters=20)`**
Deprecated compatibility subclass of `ReAct` in 3.5, warning on construction; removed in 3.6.

## Cross-links

- [Tools and MCP](tools.md) — defining, wrapping, validating, and importing the tools these agents call.
- [Adapters: how signatures become prompts](adapters.md) — the formatting boundary for `History`, `ToolCalls`, and native function calling.
- [Modules: composing your own](modules.md) — tracing, state, and composition for DSPy modules.
- [`dspy.ReAct` API reference](../api/modules/ReAct.md)
- [`dspy.ReActV2` API reference](../api/modules/ReActV2.md)
