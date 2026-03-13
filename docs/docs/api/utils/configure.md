# dspy.configure

Set the default language model, adapter, and other settings for DSPy.

```python
dspy.configure(**kwargs)
```

Call `dspy.configure(...)` once near the top of your script or notebook.
Every DSPy module will use these defaults unless you override them with
[`dspy.context`](context.md). The values persist until you call
`dspy.configure(...)` again.

!!! note
    Pass a [`dspy.LM`](../models/LM.md) object as `lm`, not a bare model
    string.

## Settings

| Setting | Default | Description |
| --- | --- | --- |
| `lm` | `None` | Default language model. Pass a [`dspy.LM`](../models/LM.md) instance. |
| `adapter` | `None` | Formats prompts and parses LM responses. When `None`, modules use [`dspy.ChatAdapter`](../adapters/ChatAdapter.md). |
| `callbacks` | `[]` | Observability and logging hooks. See [Observability](../../tutorials/observability/index.md). |
| `track_usage` | `False` | Record token counts for every LM call. |
| `async_max_workers` | `8` | Maximum concurrent workers for async operations. |
| `num_threads` | `8` | Thread count for [`dspy.Parallel`](../modules/Parallel.md). |
| `max_errors` | `10` | Stop parallel execution after this many errors. |
| `disable_history` | `False` | Stop recording LM call history. |
| `max_history_size` | `10000` | Cap on stored history entries. |
| `allow_tool_async_sync_conversion` | `False` | Let async tools run in synchronous code. See [Async](../../tutorials/async/index.md). |
| `provide_traceback` | `False` | Include Python tracebacks in error logs. |
| `warn_on_type_mismatch` | `True` | Warn when a module input type does not match the signature. |

## Examples

### Set the default LM

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-5-mini"))

qa = dspy.Predict("question -> answer")
result = qa(question="What is the capital of France?")
print(result.answer)
```

### Set the LM and adapter

```python
import dspy

dspy.configure(
    lm=dspy.LM("anthropic/claude-sonnet-4-6"),
    adapter=dspy.JSONAdapter(),
)
```

### Enable usage tracking and tune concurrency

```python
import dspy

dspy.configure(
    lm=dspy.LM("gemini/gemini-3-flash-preview"),
    track_usage=True,
    async_max_workers=4,
)
```

## When to use `dspy.configure`

Use `dspy.configure(...)` when one set of defaults should apply to most of
your program—scripts, notebooks, test setup, or application startup.

If you need different settings for one call or one block, use
[`dspy.context`](context.md) instead.

## Thread safety

Only the thread that first calls `dspy.configure(...)` may call it again.
Other threads that try will get a `RuntimeError`. In async code, only the
task that first called `dspy.configure(...)` may continue to call it.

For temporary overrides inside worker threads, async tasks, or
[`dspy.Parallel`](../modules/Parallel.md) blocks, use
[`dspy.context`](context.md).

## See Also

- [`dspy.context`](context.md) — temporary overrides that last for one block.
- [`dspy.LM`](../models/LM.md) — create the language model you pass as `lm`.
- [Language Models](../../learn/programming/language_models.md) — overview of LM configuration.
- [Adapters](../../learn/programming/adapters.md) — how adapters format prompts and parse responses.
