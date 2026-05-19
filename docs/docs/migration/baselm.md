# Migrating `dspy.BaseLM` subclasses to the typed contract

DSPy 3 evolves `dspy.BaseLM` in place to the normalized request/response
contract. The v1 signature

```python
class MyLM(dspy.BaseLM):
    def forward(self, prompt=None, messages=None, **kwargs):
        ...  # returns an OpenAI-shaped response object
```

still works during the deprecation window. When you define a v1 subclass,
DSPy emits a one-shot `DeprecationWarning` at the class line. **The v1
signature is removed in DSPy 4.0.**

## What changes

A v2 `BaseLM` subclass takes a single typed `LMRequest` and returns a typed
`LMResponse`:

```python
import dspy

class MyLM(dspy.BaseLM):
    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        ...
        return dspy.LMResponse.from_text("hello", model=request.model)

    @property
    def capabilities(self) -> dspy.LMCapabilities:
        return dspy.LMCapabilities(streaming=True, function_calling=True)
```

The base class handles request normalization, retries, callbacks, caching,
history, and error mapping. Streaming, async, and native reasoning are
opt-in by overriding `forward_stream`, `aforward`, `aforward_stream`, and
publishing the corresponding `LMCapabilities` flag.

## Porting a v1 subclass

A typical v1 implementation looked like this:

```python
class FakeLM(dspy.BaseLM):
    def __init__(self):
        super().__init__(model="fake/test")

    def forward(self, prompt=None, messages=None, **kwargs):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content="hi", tool_calls=None),
                finish_reason="stop",
            )],
            usage={},
            model="fake/test",
            _hidden_params={},
        )
```

The v2 port becomes:

```python
class FakeLM(dspy.BaseLM):
    def __init__(self):
        super().__init__(model="fake/test")

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        return dspy.LMResponse.from_text("hi", model=request.model)
```

`LMResponse.from_text` is the simple path. For tool calls, reasoning,
citations, or multiple output candidates, construct an `LMResponse`
directly with `dspy.lm.LMOutput` and the part types under `dspy.lm.*`.

## Capability declarations

The v1 `supports_*` boolean properties (function calling, reasoning,
response schema) are derived from `lm.capabilities` on v2 subclasses, so
you no longer need to override the property — declare it on
`LMCapabilities` and DSPy adapters will pick it up automatically.

## Streaming and async

Override `forward_stream(request) -> Iterator[LMStreamEvent]` (and
`aforward_stream` for async). The base class exposes
`supports_streaming` / `supports_async` as override-derived properties:
if you override the method, the capability is True.

## What does **not** change

- The user-facing entry point. `dspy.LM("openai/gpt-4o-mini")` still
  works. Setting `dspy.settings.experimental=True` routes `dspy.LM`
  construction through the new `LMRouter` and concrete `BaseLM`
  backends (`OpenAIChatLM`, `OpenAIResponsesLM`, `AnthropicLM`,
  `GenAILM`, `LiteLLMLM`).
- Existing v1 subclasses keep working until DSPy 4.0. The base class
  translates `LMRequest` → `(prompt, messages, **kwargs)` and wraps
  the OpenAI-shaped response into an `LMResponse`.

## Removal target

The legacy `forward(prompt, messages)` signature, the
`_lm_contract_version` sentinel, and the v1 translation shim are removed
in DSPy 4.0. After that release, `BaseLM.forward` is
`def forward(self, request: LMRequest) -> LMResponse` only.
