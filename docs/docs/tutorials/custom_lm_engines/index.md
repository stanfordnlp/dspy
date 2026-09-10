# Custom LM Engines

In the DSPy 3.4 development API, you can supply your own execution engine to `dspy.LM`. The minimum synchronous interface is:

```python
class MyEngine:
    def complete(self, request: dspy.lm15.Request) -> dspy.lm15.Response:
        ...
```

DSPy formats the program's inputs, converts them to an lm15 request, calls your engine, and parses the answer into a `Prediction`. You do not need to subclass `BaseLM` or return an OpenAI SDK object.

This tutorial wraps the [Pi CLI](https://pi.dev) as a custom engine. Pi keeps its normal system prompt and tools, so a DSPy program can ask it to inspect a repository. The entire agent run, including tool calls, becomes one DSPy LM response.

## Prerequisites

- A DSPy build with the custom `engine=` interface.
- Pi installed, available as `pi` on your PATH, and authenticated.
- A model available to your Pi account. The example uses `openai-codex` and `gpt-6-astra`; change those two CLI arguments if needed.
- A local Git repository to inspect.

!!! warning "Pi has filesystem and shell access"
    This example keeps Pi's normal tools, settings, and discovered resources. They can execute commands and modify files. Asking it not to modify anything is an instruction, not a sandbox. Run only against a trusted repository and use an OS-level sandbox if isolation matters.

## Define the engine

Pi's print mode runs its tool loop and writes the final assistant text to stdout. Wrap that text in an lm15 response; no event-stream parsing is needed for this demo.

Append DSPy's system instructions with `--append-system-prompt`. This preserves Pi's normal system prompt while telling it how to format the answer for DSPy.

```python
import subprocess
import dspy
from dspy.lm15 import Message, Response, Usage


class PiEngine:
    def complete(self, request):
        result = subprocess.run(
            ["pi", "--print", "--no-session",
             "--provider", "openai-codex", "--model", "gpt-6-astra",
             "--append-system-prompt", request.system or ""],
            input=request.messages[-1].text,
            text=True, capture_output=True, check=True, timeout=120,
        )
        return Response(
            id=None, model=request.model,
            message=Message.assistant(result.stdout.strip()),
            finish_reason="stop", usage=Usage(),
        )
```

`Usage()` means **usage was not reported to DSPy**, not that the run consumed zero tokens. This minimal wrapper does not aggregate Pi's token usage or costs across its tool loop.

## Run a DSPy program

Run these cells in a notebook whose working directory is your repository root. The last expression displays the prediction without `print()`:

```python
dspy.configure(
    lm=dspy.LM("pi", engine=PiEngine(), cache=False, num_retries=0),
    adapter=dspy.ChatAdapter(use_json_adapter_fallback=False),
)

program = dspy.Predict("question -> answer")
program(
    question="Find the largest tracked file by byte size in this repository. "
             "Use a tool to check. Report its path and size; do not modify anything."
)
```

Here `"pi"` is a DSPy model label for history; the engine's CLI arguments select the actual provider and model.

An answer for the DSPy repository can look like:

```text
Prediction(
    answer='Largest tracked file: docs/docs/tutorials/observability/mlflow_trace_ui_navigation.gif\nSize: 8,704,325 bytes.'
)
```

The answer depends on your repository. Pi still uses tools, but print mode does not expose tool events; this is not incremental streaming.

## Demo boundaries

- **Minimal input handling:** forwards only the last user text message and a plain-text system prompt. Earlier messages, generation options, and DSPy-declared tools are not forwarded or validated. Use this demo without demonstrations, conversation history, or media. Pi's own tools remain enabled.
- **Synchronous only:** no async counterpart or streaming implementation.
- **No saved Pi session:** `--no-session` avoids transcript persistence, but Pi may still update its own settings or credentials.
- **No automatic DSPy replay:** caching, LM retries, and adapter fallback are disabled for the demonstrated program. Pi's own retry and compaction settings still apply.
- **No detailed telemetry:** usage, costs, and tool events are not exposed by print mode. This demo labels a successful process exit as `stop`; it cannot distinguish a token-limit finish. Use `--mode json` for authoritative finish reasons and accounting across every assistant turn.
- **Normal Pi output:** extensions can affect stdout. For a controlled production protocol, use JSON/RPC mode and explicitly select trusted extensions.
- **Buffered subprocess output:** stdout and stderr are collected in memory. The timeout limits the direct Pi process, but this demo does not manage an entire descendant process tree.
- **System instructions in arguments:** the appended prompt is visible to local process inspection and subject to OS argument-length limits. Avoid sensitive system instructions in this minimal example.

These are deliberate shortcuts for demonstrating the custom-engine interface, not a production subprocess backend. For a more faithful integration, preserve complete messages and provider metadata, implement cancellation and bounded output handling, and explicitly map or reject each generation option.
