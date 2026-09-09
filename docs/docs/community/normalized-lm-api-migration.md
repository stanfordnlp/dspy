# Moving from DSPy 3.3's experimental LM types to lm15

DSPy is replacing its experimental LM vocabulary with the lm15 objects bundled
inside DSPy. Import these objects from `dspy.lm15`; no separate installation is
needed. They are the original lm15 classes, not DSPy wrappers or subclasses.

## Ordinary DSPy programs

Keep using ordinary calls:

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
outputs = lm("Hello")
outputs = lm(messages=[{"role": "user", "content": "Hello"}])
```

These return lists of strings or dictionaries. `n`, response caching, usage
tracking, callbacks and history remain DSPy's responsibilities. Setting
`experimental=True` no longer changes the return type of an ordinary LM call.
It still controls other experimental DSPy features.

This type-removal step retains the existing LiteLLM execution backend. Bundling
lm15 and accepting its objects do not by themselves enable native provider
routing. Ordinary provider-specific inputs continue through the legacy boundary
rather than being forced into lm15's narrower request vocabulary.

## Explicit typed calls

The built-in `dspy.LM` accepts an explicit request for chat and Responses models:

```python
from dspy.lm15 import Config, Message, Request

request = Request(
    model=lm.model,
    system="Be concise.",
    messages=(Message.user("What is DSPy?"),),
    config=Config(max_tokens=200),
)
response = lm(request)
print(response.text)

# Async equivalent:
# response = await lm.acall(request)
```

The request's model must match the LM. Generation options belong in its config;
LM generation defaults are not added to an explicit request. Client settings
such as API credentials still come from the LM. Pass `cache=False` or
`rollout_id=...` alongside the request to control DSPy's response cache.
`Config.cache` instead controls provider-side prompt caching: it is not DSPy's
response cache.

One lm15 response contains one assistant message, not a list of candidates.
Use ordinary calls for `n` answers. Text-completion models and legacy custom
LMs retain their ordinary call interfaces; they do not automatically accept
explicit lm15 requests.

## What changed for experimental API users

This is a breaking replacement of the **experimental 3.3 API**, not a rename.
The old `dspy.core.types` import raises an error pointing here. Its old names
are no longer exported from `dspy` or `dspy.core`.

| Old experimental API | New API |
| --- | --- |
| `dspy.LMRequest` | `dspy.lm15.Request` |
| `dspy.LMResponse` | `dspy.lm15.Response` |
| `dspy.LMMessage` | `dspy.lm15.Message` |
| `dspy.LMConfig` | `dspy.lm15.Config` |
| `dspy.System(text)` | `Request(system=text, ...)` |
| `dspy.User(text)` | `Message.user(text)` |
| `dspy.Assistant(text)` | `Message.assistant(text)` |
| `dspy.Developer(text)` | `Message.developer(text)` |
| `dspy.ToolCall(id=..., name=..., args=...)` | `ToolCallPart(id=..., name=..., input=...)` |
| `dspy.ToolResult(content, call_id=...)` | `Message.tool(call_id, content)` |
| `LMTextPart`, `LMImagePart`, etc. | `TextPart`, `ImagePart`, etc., from `dspy.lm15` |
| `response.outputs[0].parts` | `response.message.parts` |
| Previous response as a conversation turn | Add `response.message` to the next request's messages |

lm15 objects are frozen dataclasses, not Pydantic models. Required identities,
allowed roles, nonempty messages and media-source validation differ. Do not
perform a blind search-and-replace. Fields such as document citation settings,
file names and arbitrary per-part metadata do not all have direct equivalents.

Old saved Python objects containing the removed classes are not automatically
migrated. Load and export them in their original DSPy environment before moving
that code to the new API. This is separate from ordinary provider-response
cache entries and ordinary saved-program configuration.

## Custom LM authors

Existing custom LMs can continue to subclass `dspy.BaseLM` and implement:

```python
class MyLM(dspy.BaseLM):
    def forward(self, prompt=None, messages=None, **kwargs):
        # Return an OpenAI-shaped provider response.
        ...
```

Implement `aforward` for async calls. An existing `forward_contract = "legacy"`
declaration is harmless. The experimental `forward_contract = "typed_lm"`
implementation contract is removed and rejected explicitly; it is not silently
reinterpreted as a legacy method. The native custom-engine integration is a
separate migration step, not an API promised by this type-removal change.

## Custom adapter authors

Call `lm(messages=messages, **lm_kwargs)` or its `acall` equivalent, then parse
the returned list. Do not depend on the removed private conversion methods in
`Adapter` or `dspy.clients.openai_format`.

DSPy's signature types (`Image`, `Audio`, `File`, `Tool`, `ToolCalls`, `History`,
`Reasoning`, etc.) and LM error classes are not removed by this change.
