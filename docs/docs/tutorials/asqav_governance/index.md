# Adding AI Governance to DSPy Programs with Asqav

This tutorial shows how to add governance and audit signing to DSPy programs using [Asqav](https://github.com/jagmarques/asqav-sdk). Every module call, LM interaction, and tool invocation gets a cryptographic governance signature, giving you a tamper-proof record of what your AI system did and when.

Asqav uses DSPy's `BaseCallback` interface, so it plugs into any existing program with no code changes beyond registering the callback.

## Prerequisites

- Python 3.10+
- An Asqav account and API key (see [Asqav docs](https://github.com/jagmarques/asqav-sdk))
- A working DSPy program

## Installation

```bash
pip install dspy asqav
```

## How It Works

DSPy's `BaseCallback` class provides start/end hooks for every major component:

| Hook pair | Fires when |
|:--|:--|
| `on_module_start` / `on_module_end` | A `dspy.Module` subclass runs `forward()` |
| `on_lm_start` / `on_lm_end` | A `dspy.LM` instance is called |
| `on_tool_start` / `on_tool_end` | A `dspy.Tool` is invoked |
| `on_adapter_format_start` / `on_adapter_format_end` | An adapter formats the input prompt |
| `on_adapter_parse_start` / `on_adapter_parse_end` | An adapter parses LM output |

Asqav's callback subclasses `BaseCallback` and calls `agent.sign()` inside each hook, creating a signed governance event. Signing failures are logged but never raise, so governance issues will not break your AI pipeline.

## Step 1: Define the Asqav Callback

```python
import logging
from typing import Any

import asqav
from dspy.utils.callback import BaseCallback

logger = logging.getLogger(__name__)


class AsqavCallback(BaseCallback):
    """DSPy callback that signs governance events via Asqav."""

    def __init__(self, agent_name: str = "dspy-agent"):
        asqav_agent = asqav.Agent.create(agent_name)
        self._agent = asqav_agent
        self._signatures = []

    def _sign(self, action_type: str, context: dict | None = None):
        """Sign an action. Failures are logged but never raised."""
        try:
            sig = self._agent.sign(action_type, context)
            self._signatures.append(sig)
            return sig
        except Exception as exc:
            logger.warning("asqav signing failed (fail-open): %s", exc)
            return None

    # -- Module hooks ----------------------------------------------------------

    def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        self._sign("module:start", {
            "call_id": call_id,
            "module": type(instance).__name__,
            "input_keys": list(inputs.keys()),
        })

    def on_module_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None):
        self._sign("module:end", {
            "call_id": call_id,
            "has_error": exception is not None,
        })

    # -- LM hooks --------------------------------------------------------------

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        model = getattr(instance, "model", "unknown")
        self._sign("lm:start", {
            "call_id": call_id,
            "model": str(model),
        })

    def on_lm_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None):
        self._sign("lm:end", {
            "call_id": call_id,
            "has_error": exception is not None,
        })

    # -- Tool hooks ------------------------------------------------------------

    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        tool_name = getattr(instance, "name", type(instance).__name__)
        self._sign("tool:start", {
            "call_id": call_id,
            "tool": str(tool_name),
        })

    def on_tool_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None):
        self._sign("tool:end", {
            "call_id": call_id,
            "has_error": exception is not None,
        })
```

## Step 2: Register the Callback

Pass the callback to `dspy.configure` so it applies to all module, LM, and tool calls:

```python
import dspy

# Initialize Asqav with your API key
asqav.init(api_key="your-asqav-api-key")

# Configure DSPy with the governance callback
lm = dspy.LM("openai/gpt-4o-mini")
callback = AsqavCallback(agent_name="my-dspy-agent")

dspy.configure(lm=lm, callbacks=[callback])
```

## Step 3: Run Your Program

No changes are needed to your existing DSPy code. The callback hooks fire automatically:

```python
cot = dspy.ChainOfThought("question -> answer")
result = cot(question="What is retrieval-augmented generation?")
print(result.answer)
```

Every module call and LM interaction now has a signed governance record. You can inspect the collected signatures:

```python
for sig in callback._signatures:
    print(f"{sig.action_type}: {sig.signature_id}")
```

## Step 4: Using with ReAct Agents

The callback also covers tool calls, which is useful for agent-based programs:

```python
def search(query: str) -> str:
    """Search for information."""
    return "DSPy is a framework for programming language models."

agent = dspy.ReAct("question -> answer", tools=[search])
result = agent(question="What is DSPy?")

# Module, LM, and tool events are all signed
print(f"Total governance signatures: {len(callback._signatures)}")
```

## Step 5: Scope Callbacks to Specific Components

You can also attach the callback to a single LM or module instead of setting it globally:

```python
# Only this LM instance gets governance signing
governed_lm = dspy.LM("openai/gpt-4o-mini", callbacks=[AsqavCallback(agent_name="governed-lm")])

# This LM has no governance callback
ungoverned_lm = dspy.LM("openai/gpt-4o-mini")
```

## Next Steps

- Browse the [Asqav SDK documentation](https://github.com/jagmarques/asqav-sdk) for details on agents, sessions, and signature verification
- See DSPy's [Debugging and Observability tutorial](../observability/index.md) for more on the `BaseCallback` interface
- Combine Asqav with [MLflow tracing](../observability/index.md#tracing) for both governance and observability
