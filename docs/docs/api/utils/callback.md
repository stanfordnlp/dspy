# Callback (BaseCallback)

DSPy exposes a callback interface to observe execution across modules, LMs, adapters, tools, and evaluation.

## Interface

- on_module_start(call_id, instance, inputs)
- on_module_end(call_id, outputs, exception)
- on_lm_start(call_id, instance, inputs)
- on_lm_end(call_id, outputs, exception)
- on_lm_raw_response(call_id, instance, response)
  - Fired immediately after the provider returns, before parsing/processing.
- on_adapter_format_start(call_id, instance, inputs)
- on_adapter_format_end(call_id, outputs, exception)
- on_adapter_parse_start(call_id, instance, inputs)
  - inputs["completion"] contains the raw completion string before parsing.
- on_adapter_parse_end(call_id, outputs, exception)
- on_tool_start(call_id, instance, inputs)
- on_tool_end(call_id, outputs, exception)
- on_evaluate_start(call_id, instance, inputs)
- on_evaluate_end(call_id, outputs, exception)
- on_retry_start(call_id, instance, attempt, reason=None, parent_call_id=None)
  - Emitted by modules that retry (BestOfN, Refine, ProgramOfThought).
- on_retry_end(call_id, outputs, exception)

All handlers include a call_id to correlate events.

## Usage

```python
import dspy
from dspy.utils.callback import BaseCallback

class MyCallback(BaseCallback):
    def on_retry_start(self, call_id, instance, attempt, reason=None, parent_call_id=None):
        ...

    def on_lm_raw_response(self, call_id, instance, response):
        ...

    def on_adapter_parse_start(self, call_id, instance, inputs):
        raw = inputs.get("completion", "")
        ...

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), callbacks=[MyCallback()])
```