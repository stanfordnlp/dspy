# dspy.TinkerLM

<!-- START_API_REF -->
::: dspy.TinkerLM
    handler: python
    options:
        members:
            - __call__
            - acall
            - copy
            - dump_state
            - inspect_history
            - load_state
            - update_history
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
<!-- END_API_REF -->

## Using Tinker models

[Tinker](https://thinkingmachines.ai/tinker) hosts open-weight base models and
the LoRA checkpoints you train on them. `dspy.TinkerLM` samples from either
through the Tinker SDK. The tinker_cookbook renderer for the base model turns
DSPy's messages and tools into a token prompt, and parses the sampled tokens
back into text, thinking and tool calls.

Install the extra (Python 3.11 or later) and set `TINKER_API_KEY`, or pass
`api_key=`:

```shell
pip install "dspy[tinker]"
```

The extra installs `tinker` and `tinker-cookbook`. The cookbook depends on
PyTorch, Transformers and Hugging Face Datasets, so expect a large install even
though sampling runs on Tinker's servers. On Python 3.10 the extra installs
nothing.

```python
import dspy

lm = dspy.TinkerLM("Qwen/Qwen3-8B", temperature=0.7, max_tokens=2048)
dspy.configure(lm=lm)
print(dspy.Predict("question -> answer")(question="What is the capital of France?").answer)
```

For a checkpoint, pass its `tinker://` sampler path. The base model is read
from the checkpoint:

```python
tuned = dspy.TinkerLM(model_path="tinker://<run-id>/sampler_weights/final")
```

`renderer=` selects a tinker_cookbook renderer by name; the default is the one
recommended for the base model. Thinking is chosen by the renderer, so use, for
example, `renderer="qwen3_disable_thinking"` to turn off Qwen3's thinking. When
the renderer produces thinking, it is returned as a reasoning part, not in the
output fields. You can also pass an existing `tinker.SamplingClient` as
`sampling_client=`, for example one returned by
`training_client.save_weights_and_get_sampling_client()`. Such a client is not
saved with the program, and unless you also pass its `model_path=`, DSPy's
response cache is off by default, because the cache key could only name the
base model.

### Generation options

`temperature`, `max_tokens`, `top_p`, `top_k` and `seed` go to Tinker's
`SamplingParams`, and DSPy samples one sequence per candidate. The sampler stops
at the end of the assistant turn. A `stop` you set is applied to the parsed
reply, and the call's history records this adaptation. Options Tinker cannot
honour raise `dspy.LMUnsupportedFeatureError` instead of being ignored:
`response_format`, `reasoning_effort`, `logprobs`, penalties, provider-side
prompt caching, and provider-specific options. Images, audio and other media
are refused too.

### Tools

Tools are rendered in the model's own tool format, and its tool calls are parsed
into native tool calls, so `dspy.ReAct` and adapters with
`use_native_function_calling=True` work as with any function-calling model. A
tool call the renderer cannot parse is kept in the reply text. Sampling cannot
force a tool call: a `tool_choice` naming one tool renders only that tool, and
the history records this adaptation.

### Async

`await lm.acall(...)` and `await program.acall(...)` await the SDK's sampling
future. One sampling client serves synchronous and asynchronous calls from any
event loop. It is created on the first call, not at construction.

### Saving and loading

`dump_state()` and `program.save(...)` keep the model, checkpoint path, renderer
and generation options, never the API key. `TinkerLM` is a custom LM class, so
loading it needs `allow_unsafe_lm_state=True` on trusted files:

```python
program.save("program.json")
restored = dspy.Predict("question -> answer")
restored.load("program.json", allow_unsafe_lm_state=True)
```

DSPy's response cache keys on `lm.model`, which names the checkpoint path and
any renderer override, so cached answers do not cross checkpoints. The Tinker SDK
retries transient connection failures inside a sample; DSPy's `num_retries`
applies on top of that, so pass `num_retries=0` to leave retries to the SDK.

The engines behind `TinkerLM`, `dspy.clients.tinker.TinkerEngine` and
`AsyncTinkerEngine`, follow the [custom engine contract](../../tutorials/custom_lm_engines/index.md)
and also work with `dspy.LM(..., engine=..., async_engine=...)`.
