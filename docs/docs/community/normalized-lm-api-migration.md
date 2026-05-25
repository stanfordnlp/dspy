# Typed LM API migration plan

DSPy is moving toward a typed language-model boundary while keeping `dspy.BaseLM` as the public base class for language models.

**Most DSPy users do not need to change anything in DSPy 3.3**. Existing `lm(...)`, modules, and programs keep their current behavior by default. The typed LM API is opt-in in 3.3 with `dspy.context(experimental=True)`.

TLDR: `dspy.LM.forward` is currently untyped and mixes DSPy-specific behavior with OpenAI/LiteLLM-shaped inputs. We will migrate `BaseLM.forward` and `LM.forward` from:

```python
def forward(self, prompt=None, messages=None, **kwargs):
    ...
```

to:

```python
def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
    ...
```

!!! note "Status"
    This is a migration plan for the DSPy 3.3–3.6/4.0 series. Names and exact release timing may change before implementation lands, but the staged compatibility plan below should guide discussion.

!!! info "Community feedback wanted"
    This plan mostly affects custom LMs and adapters. If you maintain one, please review the proposed one-line `forward_contract` migration and share feedback before the default LM path changes.

## Who is affected?

| Group | What to do now | Future requirement |
| --- | --- | --- |
| Most DSPy users | Nothing required. Optionally try the direct `lm(...)` API with `dspy.context(experimental=True)` and provide feedback. | DSPy programs will keep working before, during, and after this migration without user changes. |
| Existing custom LM authors | Nothing required in 3.3. If you want to be explicit, add `forward_contract = "legacy"`. | Add an explicit `forward_contract`; eventually migrate to `forward_contract = "typed_lm"` before legacy support is removed. |
| New custom LM authors | Use `forward_contract = "typed_lm"` and implement `forward(request: dspy.LMRequest) -> dspy.LMResponse`. | No later migration needed if you start with the typed contract. |
| Custom adapter authors | Call `lm(...)`, not `lm.forward(...)`. | Build `LMRequest` objects and parse `LMResponse` directly. |


## Background

Today, `BaseLM` subclasses implement an untyped forward method, with a few optional parameters:

```python
def forward(self, prompt=None, messages=None, **kwargs):
    ...
```

That hook usually receives OpenAI/LiteLLM-shaped inputs and returns an OpenAI-like provider response. DSPy then post-processes that response into a `list[str | dict]` containing outputs.

Because the current parameters are untyped, it is hard to know inside an LM exactly which inputs you will get and what types they will contain. The new contract is typed and provider-neutral. We have designed `LMRequest` and `LMResponse` to be flexible enough for LMs backed by many different provider APIs:

```python
def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
    ...
```

DSPy has settled on the internal LM type system around `LMRequest`, `LMResponse`, typed messages, parts, config, usage, and stream events. These types should be treated as the stable direction for LM implementations. Concrete LMs translate between these DSPy types and their provider API.

## Why this matters

The typed boundary gives DSPy one clear internal representation for LM calls:

```text
LMRequest -> BaseLM -> LMResponse
```

That gives DSPy and the community:

- cleaner custom LM implementations,
- less OpenAI/LiteLLM-shaped logic inside adapters,
- first-class support for multimodal inputs, tool calls, reasoning, citations, usage, and provider metadata,
- a more expressive direct `lm(...)` UX,
- a clearer path for community packages to ship LMs that feel and are treated like first-class DSPy LMs.

The migration is staged so existing code keeps working while new code can opt into the typed path.

## Guide for DSPy users

Most users should not need to change anything in 3.3.

Default behavior remains legacy:

```python
outputs = lm("hello")
# list[str | dict]
```

To try the typed LM API in 3.3, use the existing experimental switch:

```python
with dspy.context(experimental=True):
    response = lm("hello")
    print(response.text)
```

Typed responses carry structured data:

```python
response.text
response.outputs
response.usage
response.cache_hit
response.provider_data
```

The typed path also makes direct `lm(...)` calls more expressive. Strings, typed messages, media parts, previous responses, and explicit `LMRequest` objects all flow through one call API.

!!! warning "Planned 3.3 API"
    The examples below illustrate the proposed typed LM call API. Helpers such as `dspy.System`, `dspy.User`, `dspy.Assistant`, `dspy.ToolCall`, and `dspy.ToolResult` are part of the planned implementation and may not exist in the current stable namespace yet.

Multimodal request with instructions:

```python
with dspy.context(experimental=True):
    response = lm(
        dspy.System("Be concise."),
        dspy.User("Describe this image.", dspy.Image("https://example.com/dog.png")),
        temperature=0.2,
    )
```

Multi-turn conversation:

```python
with dspy.context(experimental=True):
    response = lm(
        dspy.User("What is DSPy?"),
        dspy.Assistant("DSPy is a framework for programming LM pipelines."),
        dspy.User("Say that in five words."),
    )
```

Tool-call transcript:

```python
with dspy.context(experimental=True):
    response = lm(
        dspy.User("What is the weather in Paris?"),
        dspy.Assistant(dspy.ToolCall(id="call_1", name="get_weather", args={"city": "Paris"})),
        dspy.ToolResult('{"temperature": "22 C"}', call_id="call_1", name="get_weather"),
        dspy.User("Summarize the result."),
    )
```

Passing a previous response back into the conversation:

```python
with dspy.context(experimental=True):
    first = lm("Explain DSPy in one sentence.")
    follow_up = lm(
        dspy.User("Explain DSPy in one sentence."),
        first,
        dspy.User("Now make it even shorter."),
    )
```

## Guide for custom LM authors

Custom LM authors should declare which `forward()` contract their class implements.

Legacy LMs should add:

```python
class MyLegacyLM(dspy.BaseLM):
    forward_contract = "legacy"

    def forward(self, prompt=None, messages=None, **kwargs):
        ...
```

Typed LMs should add:

```python
class MyTypedLM(dspy.BaseLM):
    forward_contract = "typed_lm"

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        ...
```

In DSPy 3.3, classes without an explicit `forward_contract` are treated as legacy for compatibility. In later releases, missing declarations will warn and then may become errors or change defaults.

A minimal typed LM looks like this:

```python
class EchoLM(dspy.BaseLM):
    forward_contract = "typed_lm"

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        return dspy.LMResponse.from_text("hello", model=request.model)
```

## Guide for custom adapter authors

Adapters should call the LM object, not `forward()` directly.

Preferred typed boundary:

```python
request = dspy.LMRequest.from_call(
    model=lm.model,
    messages=messages,
    **lm_kwargs,
)
response = lm(request)
```

Avoid this in adapters:

```python
lm.forward(...)
```

`BaseLM.__call__()` is the compatibility boundary. It owns input normalization, choosing the legacy or typed `forward()` path, adapting legacy outputs into `LMResponse`, and preserving public return behavior unless `experimental=True` is enabled.

During the transition, adapters may still convert `LMResponse` back to legacy parser inputs. The long-term direction is for adapters to parse `LMResponse` directly.

## Version sequence

| Version | Custom `BaseLM.forward` contract | Public `lm(...)` behavior | LiteLLM role |
| --- | --- | --- | --- |
| 3.3 | Missing `forward_contract` is treated as legacy. | Typed returns available only through `experimental=True` or explicit `LMRequest` calls. | Current `dspy.LM` LiteLLM path remains the default. |
| 3.4 | Missing `forward_contract` is treated as legacy and warns. | Still requires `experimental=True` or explicit `LMRequest` while migration continues. | Native typed LMs become preferred where available; LiteLLM is used as a compatibility fallback. |
| 3.5 | Require explicit contract or flip default after final review. | Typed path becomes default with a legacy escape hatch. | Native typed LMs remain preferred; LiteLLM is used as a compatibility fallback but may require manual installation. |
| 3.6 or 4.0 | Remove the legacy `forward(prompt, messages, **kwargs)` implementation contract after final review. | `forward(request: LMRequest) -> LMResponse` is the only supported `BaseLM` implementation contract. | TBD whether the LiteLLM fallback remains. |

The important distinction is that removing the legacy `BaseLM.forward(prompt, messages, **kwargs)` contract does not require removing LiteLLM. LiteLLM can continue as a typed compatibility implementation that accepts `LMRequest` internally and returns `LMResponse`.

Before changing the default, DSPy will give custom LM authors enough time to add one of:

```python
forward_contract = "legacy"
```

or:

```python
forward_contract = "typed_lm"
```
