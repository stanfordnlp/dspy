# dspy.context

Override DSPy settings inside one `with` block.

```python
with dspy.context(**kwargs):
    ...  # overrides active here
# original settings restored
```

Use `dspy.context(...)` when you need a different LM, adapter, or flag for
one part of your program without changing the process-wide defaults from
[`dspy.configure`](configure.md). The block inherits every current setting,
overrides only the keys you pass, and restores the originals when it exits.

`dspy.context(...)` accepts the same settings as
[`dspy.configure`](configure.md#settings).

## Examples

### Use a different LM for one block

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-5-mini"))
qa = dspy.Predict("question -> answer")

result = qa(question="What is the capital of France?")
print("default:", result.answer)

with dspy.context(lm=dspy.LM("anthropic/claude-sonnet-4-6")):
    result = qa(question="What is the capital of France?")
    print("temporary:", result.answer)

print("restored:", dspy.settings.lm.model)  # openai/gpt-5-mini
```

### Use a different adapter for one block

```python
import dspy

dspy.configure(lm=dspy.LM("gemini/gemini-3-flash-preview"))
qa = dspy.Predict("question -> answer")

with dspy.context(adapter=dspy.JSONAdapter()):
    result = qa(question="What is the capital of France?")
    print(result.answer)
```

### Enable async tool conversion temporarily

```python
import asyncio
import dspy

async def async_tool(x: int) -> int:
    await asyncio.sleep(0.1)
    return x * 2

tool = dspy.Tool(async_tool)

with dspy.context(allow_tool_async_sync_conversion=True):
    print(tool(x=5))
```

### Nested blocks

Inner blocks override outer ones. Each restores cleanly on exit:

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-5-mini"))

with dspy.context(lm=dspy.LM("anthropic/claude-sonnet-4-6")):
    print(dspy.settings.lm.model)        # anthropic/claude-sonnet-4-6
    with dspy.context(track_usage=True):
        print(dspy.settings.lm.model)    # anthropic/claude-sonnet-4-6 (inherited)
        print(dspy.settings.track_usage) # True
    print(dspy.settings.track_usage)     # False (restored)
print(dspy.settings.lm.model)           # openai/gpt-5-mini (restored)
```

## Thread safety

Unlike [`dspy.configure`](configure.md), you can call `dspy.context(...)` from
**any** thread or async task. This makes it the right tool for overrides inside
[`dspy.Parallel`](../modules/Parallel.md), `asyncio.gather`, or any concurrent
code.

Settings inside a `dspy.context(...)` block do not leak to other threads or
tasks.

## See Also

- [`dspy.configure`](configure.md) — set process-wide defaults.
- [`dspy.LM`](../models/LM.md) — create the language model you pass as `lm`.
- [Language Models](../../learn/programming/language_models.md) — overview of LM configuration.
- [Adapters](../../learn/programming/adapters.md) — how adapters format prompts and parse responses.
