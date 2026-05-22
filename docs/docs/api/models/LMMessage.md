# LM messages

Use these helpers to build normalized messages for direct LM calls.

!!! warning "Experimental LM API"
    This page describes the normalized `BaseLM` API introduced experimentally
    in DSPy 3.3 and enabled with `dspy.configure(experimental=True)`. In the current
    stable path, `dspy.LM(...)` returns the legacy LiteLLM-backed LM. The normalized
    LM path is planned to become the default in DSPy 3.5 and may change before then.

The canonical explicit form is `messages=[...]`:

```python
response = lm(
    messages=[
        dspy.System("Be concise."),
        dspy.User("What is DSPy?"),
    ]
)
```

For hand-written conversations, you can pass messages positionally:

```python
response = lm(
    dspy.System("Be concise."),
    dspy.User("What is DSPy?"),
)
```

If you have already built a list of messages, pass it as the single positional argument:

```python
messages = [
    dspy.System("Be concise."),
    dspy.User("What is DSPy?"),
]

response = lm(messages)
```

That is one conversation, not a batch. Put content parts such as images, audio, files, tool calls, and citations inside the message they belong to:

```python
response = lm(
    dspy.System("Be concise."),
    dspy.User(
        "Describe this image.",
        dspy.Image("https://example.com/dog.png"),
    ),
)
```

<!-- START_API_REF -->
::: dspy.System
    handler: python
    options:
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false

::: dspy.Developer
    handler: python
    options:
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false

::: dspy.User
    handler: python
    options:
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false

::: dspy.Assistant
    handler: python
    options:
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false

::: dspy.ToolCall
    handler: python
    options:
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false

::: dspy.ToolResult
    handler: python
    options:
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false

::: dspy.LMMessage
    handler: python
    options:
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
<!-- END_API_REF -->
