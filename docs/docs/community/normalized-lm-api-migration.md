# Typed LM API migration plan

DSPy is moving toward a typed language-model boundary while keeping `dspy.BaseLM` as the public base class for language models.

The short version: **most DSPy users do not need to change anything in DSPy 3.3**. Existing `lm(...)`, modules, and programs keep their current behavior by default. The typed LM API is opt-in in 3.3 with `dspy.context(experimental=True)`.

!!! note "Status"
    This is a migration plan for the DSPy 3.3–3.6/4.0 series. Names and exact release timing may change before implementation lands, but the staged compatibility plan below should guide discussion.

!!! info "Community feedback wanted"
    This plan mostly affects custom LM backends and adapters. If you maintain one, please review the proposed one-line `forward_contract` migration and share feedback before the default LM path changes.

## Who is affected?

| Group | DSPy 3.3 action | Later action |
| --- | --- | --- |
| Most DSPy users | No action required. | Optionally try typed direct LM calls with `dspy.context(experimental=True)`. |
| Custom LM authors | Add `forward_contract = "legacy"` if your LM implements `forward(prompt=None, messages=None, **kwargs)`. | Migrate to `forward_contract = "typed_lm"` and `forward(request: dspy.LMRequest) -> dspy.LMResponse` before legacy support is removed. |
| New LM backend authors | Prefer the typed contract from the start. | Package and distribute backends as first-class DSPy-compatible libraries. |
| Custom adapter authors | Do not call `lm.forward(...)` directly. Route LM calls through `lm(...)`. | Move toward building `LMRequest` objects and parsing `LMResponse` directly. |
| DSPy maintainers and contributors | Keep public typed returns guarded by `experimental=True` in 3.3. | Warn in 3.4, make the typed path the default candidate in 3.5, and remove legacy in 3.6 or 4.0 after final review. |

## Background

Today, most `BaseLM` subclasses implement a legacy provider-shaped hook:

```python
def forward(self, prompt=None, messages=None, **kwargs):
    ...
```

That hook usually receives OpenAI/LiteLLM-shaped inputs and returns an OpenAI-like provider response. DSPy then post-processes that response into legacy outputs such as `list[str | dict]`.

The target contract is typed:

```python
def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
    ...
```

DSPy has settled on the internal LM type system around `LMRequest`, `LMResponse`, typed messages, parts, config, usage, and stream events. These types should be treated as the stable direction for LM backends. Concrete LMs translate between these DSPy types and their provider API.

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
- a clearer path for community packages to ship LM backends that feel and are treated like first-class DSPy backends.

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

| Version | Missing `forward_contract` | Public `lm(...)` behavior |
| --- | --- | --- |
| 3.3 | Treat as legacy | Typed returns available only through `experimental=True` or explicit `LMRequest` calls. |
| 3.4 | Treat as legacy and warn | Still guarded while migration continues. |
| 3.5 | Require explicit contract or flip default after final review | Candidate default typed path. |
| 3.6 or 4.0 | Remove the legacy `forward(prompt, messages, **kwargs)` contract after final review | `forward(request: LMRequest) -> LMResponse` is the only supported `BaseLM` implementation contract. |

Before changing the default, DSPy should give custom LM authors enough time to add one of:

```python
forward_contract = "legacy"
```

or:

```python
forward_contract = "typed_lm"
```
