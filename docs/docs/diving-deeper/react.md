# ReAct and ReActV2

## Intent

ReAct is DSPy's general-purpose tool-using agent loop: the language model reasons about a task, chooses tools, observes their results, and repeats until it can produce the signature's outputs.

DSPy is transitioning between two implementations. `dspy.ReAct` is the current implementation. `dspy.ReActV2` is its experimental, structured-history replacement and will become the implementation behind the canonical `dspy.ReAct` name in DSPy 3.5. The `dspy.ReActV2` name will remain as a deprecated compatibility alias throughout the 3.5 release line and will be removed in DSPy 3.6.

Read this page to prepare for that transition and understand what is changing: how each implementation stores and formats history, calls tools, finishes a run, and interacts with provider prompt caching. For defining tools or importing them from MCP, see [Tools and MCP](tools.md).

## The transition from ReAct to ReActV2

Both modules accept the same task signature, tool list, and iteration limit:

```python
import dspy

def lookup(query: str) -> str:
    """Look up information relevant to a query."""
    return search_index(query)

agent = dspy.ReActV2("question -> answer", tools=[lookup], max_iters=10)
result = agent(question="What is DSPy?")
```

The execution models differ:

| | `dspy.ReAct` | `dspy.ReActV2` |
| --- | --- | --- |
| Status | Current implementation through DSPy 3.4; adopts the ReActV2 implementation in 3.5 | Experimental name; becomes a deprecated alias for `dspy.ReAct` throughout 3.5 and is removed in 3.6 |
| Stored history | Flat `trajectory` dictionary | Structured `dspy.History` events |
| Model-facing history | Entire trajectory formatted into the current input | Prior user, assistant, and tool messages replayed as separate turns |
| Tool selection | `next_tool_name` plus `next_tool_args` | `dspy.ToolCalls` |
| Calls per model turn | One | One or more |
| Completion | `finish`, then a separate extraction LM call | `submit` carries final typed outputs directly |
| Returned diagnostics | `prediction.trajectory` | `prediction.history` and `prediction.termination_reason` |
| Prompt caching | The changing trajectory is resent as one field | Stable prior messages remain a reusable prefix |

For new agent work, use `dspy.ReActV2` now if you can accept an experimental API; it provides an early path to the execution model that will become `dspy.ReAct` in 3.5. Existing `dspy.ReAct` programs can remain on the current implementation until the upgrade, but should prepare for the history and output changes summarized below. Pin the DSPy version while using the experimental `ReActV2` name.

## ReAct: trajectory-based execution

### How history is stored

ReAct builds one insertion-ordered `trajectory` dictionary. Every iteration appends four keys:

```python
{
    "thought_0": "I should look up the release notes.",
    "tool_name_0": "lookup",
    "tool_args_0": {"query": "DSPy release notes"},
    "observation_0": "...",
}
```

The dictionary is returned as `prediction.trajectory`. It is execution state for this invocation, not a conversation object intended to be passed into a later invocation.

### How history is formatted for the model

Before every LM call, `ReAct._format_trajectory` creates a temporary signature from the trajectory keys and asks the active adapter to format it. `ChatAdapter` uses field markers, `JSONAdapter` uses JSON, and `XMLAdapter` uses tags. ReAct then supplies that formatted value as the `trajectory` input to its internal predictor.

The trajectory is therefore stored structurally inside the loop but becomes one formatted input value at the model boundary. On every iteration the complete, now-longer trajectory is formatted again and sent with the original task inputs.

If the request exceeds the context window, ReAct drops the oldest complete thought/tool/args/observation group and retries, up to three times. Override `truncate_trajectory` to implement another retention policy.

### How tools are called

The internal `dspy.Predict` produces `next_thought`, a `next_tool_name` constrained to the registered tool names, and a `next_tool_args` dictionary. ReAct executes exactly one selected `dspy.Tool` per iteration with keyword arguments.

Tool exceptions are caught and stored as the next observation, allowing the model to recover on the following iteration. ReAct also registers a no-argument `finish` tool; selecting it ends the loop. `max_iters` provides a hard stop when the model never selects `finish`.

### What happens after the loop

Selecting `finish`, reaching `max_iters`, or encountering an unrecoverable model/context error ends navigation but does not directly produce the declared outputs. ReAct runs a separate `dspy.ChainOfThought` extractor over the original task inputs and formatted trajectory. The extractor synthesizes and types the signature's output fields. The returned prediction combines those fields with `trajectory`.

This means a normal ReAct run has at least one additional LM call after its tool loop.

### Prompt-caching behavior

ReAct resends the growing trajectory as one newly formatted input on each iteration. The instructions and original task inputs may still form a cacheable prefix, but the trajectory-bearing user content changes each turn rather than being represented as appended assistant/tool messages. Providers cannot reuse the changing portion as effectively, and the separate extraction request has a different signature and prompt shape.

## ReActV2: structured-history execution

### How history is stored

ReActV2 owns a `dspy.History`, whose `messages` list contains one structured event per loop turn. An event can contain:

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

Call IDs are preserved from the provider. ReActV2 generates deterministic IDs when non-native model output does not supply them.

With native function calling disabled, history remains structured inside DSPy, but the adapter renders the assistant fields in its normal text/JSON/XML format and renders tool results as a following user message. This lets ReActV2 work with models that do not expose a native tool API while retaining one internal history representation.

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

The internal predictor returns `dspy.ToolCalls`, so one model turn can request multiple tools. ReActV2 preserves each requested call/result pair by ID. The `parallel_tool_calls` adapter option asks a capable provider to generate multiple independent calls in the same turn; ReActV2 currently executes the returned calls one after another in Python.

Callables supplied to the constructor are converted to `dspy.Tool`. Unknown tool names and execution exceptions become error results in history so the next model turn can respond to them.

ReActV2 reserves `submit` as an internal tool. Its argument schema is generated from the original signature's output fields, so final values travel through the same structured calling path as every other action.

### What happens after each turn and at termination

After executing all calls in a turn, ReActV2 attaches their results to `ToolCalls`, appends the event to history, and invokes the model again with no duplicate copy of the original inputs. If one of the calls successfully invokes `submit`, ReActV2 returns its typed fields immediately with `termination_reason="submit"`; there is no separate extraction module.

If the loop reaches `max_iters`, returns no calls, fails to parse, or exceeds the context window, ReActV2 makes one final LM call requesting `submit`. With native function calling enabled, it sets `tool_choice` to `submit`, so the provider enforces that choice. In non-native mode, the adapter removes `tool_choice` and requests `submit` through the formatted prompt instead, so structured submission is not guaranteed. A successful fallback returns `termination_reason="forced_submit"`. If that attempt fails, the prediction still returns its history and a termination reason describing why the normal loop stopped, but it may not contain the declared output fields.

Unlike ReAct, ReActV2 does not currently truncate old history events on context overflow.

### Prompt-caching behavior

Structured history makes each completed turn an append-only message group. On the next request, the system instructions and all earlier user/assistant/tool messages remain a stable prefix; only the newest result and request are appended. Providers with prompt caching can therefore reuse more of the prior request instead of reprocessing one ever-growing, newly formatted trajectory value.

For Anthropic models, enable provider-side prompt caching on the LM with LiteLLM's cache-control injection points:

```python
lm = dspy.LM(
    "anthropic/claude-sonnet-4-5-20250929",
    cache_control_injection_points=[
        {"location": "message", "role": "system"},
        {"location": "message", "index": -1},
    ],
)

with dspy.context(lm=lm):
    result = agent(question="What is DSPy?")
```

The first injection point caches the stable system instructions. The second places a checkpoint on the trailing turn, allowing Anthropic to reuse the preceding conversation prefix as ReActV2 appends history. Anthropic only caches prefixes that meet the model's minimum token count, and its default ephemeral cache has a limited lifetime. See [Using Provider-Side Prompt Caching](../tutorials/cache/index.md#using-provider-side-prompt-caching) for the general DSPy setup and the [LiteLLM prompt-caching documentation](https://docs.litellm.ai/docs/tutorials/prompt_caching#configuration) for provider details.

Native tool replay gives the cleanest provider-visible structure, but non-native mode still benefits from stable multi-turn history. Internal testing has seen cost reductions of up to 50% on some tasks. Actual savings depend on the provider's cache policy, model, request shape, and the size of tool results; they are not guaranteed for every workload.

## Replacement and migration plan

ReActV2 is a temporary experimental name, not the start of two permanent ReAct APIs. Its structured-history implementation will replace the current `dspy.ReAct` implementation in DSPy 3.5. The canonical public name will remain `dspy.ReAct`.

To avoid forcing users of the experimental name to rename immediately, `dspy.ReActV2` will remain available as a deprecated compatibility alias throughout the DSPy 3.5 release line. Using the alias will warn users to migrate to `dspy.ReAct`; the alias will be removed in DSPy 3.6. In other words, code can move from `dspy.ReActV2(...)` to `dspy.ReAct(...)` any time after upgrading to 3.5, and should do so before upgrading to 3.6.

Existing `dspy.ReAct` programs do not need to change before upgrading to DSPy 3.5. Before that upgrade, review code that depends on `prediction.trajectory`, a separate extractor predictor, one tool call per turn, or custom trajectory truncation. Those behaviors are replaced by `prediction.history`, direct `submit` outputs, multiple calls per turn, and structured message replay.

## API summary

**`dspy.ReAct(signature, tools, max_iters=20)`**
Runs the stable trajectory-based loop and a final extraction pass.

**`ReAct.forward(**inputs)` / `ReAct.aforward(**inputs)`**
Runs the synchronous or asynchronous loop and returns output fields plus `trajectory`.

**`ReAct.truncate_trajectory(trajectory)`**
Drops the oldest complete tool-call group after a context-window error. Override to change the policy.

**`dspy.ReActV2(signature, tools, max_iters=20)`**
Runs the experimental structured-history loop and reserves `submit` for final outputs.

**`ReActV2.forward(history=None, max_iters=None, **inputs)`**
Accepts optional continuation history and a per-call iteration override. Returns `history` and `termination_reason` alongside outputs when submission succeeds.

## Cross-links

- [Tools and MCP](tools.md) — defining, wrapping, validating, and importing the tools these agents call.
- [Adapters: how signatures become prompts](adapters.md) — the formatting boundary for `History`, `ToolCalls`, and native function calling.
- [Modules: composing your own](modules.md) — tracing, state, and composition shared by both implementations.
- [`dspy.ReAct` API reference](../api/modules/ReAct.md)
- [`dspy.ReActV2` API reference](../api/modules/ReActV2.md)
