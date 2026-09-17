# Custom LM Engines

You can supply your own execution engine to `dspy.LM`. Before you do, check which of two situations you are in:

- **An HTTP provider lm15 can already speak to** — an OpenAI-compatible service, a company gateway, a host DSPy's bundled lm15 does not list yet. Do not write an engine: [declare the provider](#declaring-a-provider-instead-of-writing-an-engine) and `dspy.LM("<provider>/<model>")` routes natively, with `api_key`, `api_base`, `timeout`, saving and loading all behaving as for a built-in provider.
- **Not an HTTP provider at all** — a CLI, an in-process model, an agent harness. Write an engine. It owns its whole connection, so the [rules below](#what-a-custom-engine-owns) apply.

The minimum synchronous engine interface is:

```python
class MyEngine:
    def complete(self, request: dspy.lm15.Request) -> dspy.lm15.Response:
        ...
```

DSPy formats the program's inputs, converts them to an lm15 request, calls your engine, and parses the answer into a `Prediction`. You do not need to subclass `BaseLM` or return an OpenAI SDK object.

!!! warning "DSPy 3.5 cutoff"
    Custom `BaseLM.forward()`/`aforward()` integrations, `LegacyEngine`/`AsyncLegacyEngine`, and `complete_legacy()` shortcuts are deprecated in 3.4 and scheduled for removal in 3.5. Implement the request/response engine contract shown here; a legacy wrapper does not extend the migration deadline. OpenAI-style `lm(messages=[...])` calls are also being removed. `lm("hello")` remains a list-returning convenience; adapters use `lm(Request(...))` and consume `Response` directly. See the [migration guide](../../community/normalized-lm-api-migration.md).

Expected backend failures should raise specific errors from `dspy.lm15`, such as
`AuthError` or `RateLimitError`. DSPy translates them into its public `LMError`
family and owns retries. Unexpected exceptions retain their original cause and
are not guessed to be retryable from their message text. See
[errors and retry ownership](../../community/normalized-lm-api-migration.md#errors-and-retry-ownership).

## Declaring a provider instead of writing an engine

lm15 routes model strings through a registry of providers it has verified against the wire. A provider it does not list can be declared for the process with the same three facts a registry entry is made of — an access policy (name, key variable, address), a wire dialect, and a compat policy describing the server's spellings:

```python
import dspy
from dspy.lm15 import AccessPolicy, EndpointSupport, ModelSupport, OpenAIChatCompat, ProviderDefinition

dspy.lm15.register_provider(
    ProviderDefinition.chat(
        AccessPolicy(
            provider="fireworks",
            supports=EndpointSupport(complete=True, stream=True, models=True),
            auth_modes=("bearer",),
            env_keys=("FIREWORKS_API_KEY",),
            base_url="https://api.fireworks.ai/inference/v1",
        ),
        compat=OpenAIChatCompat(max_tokens_field="max_tokens", thinking_format="reasoning_effort"),
        aliases=("fireworks-ai",),          # accept LiteLLM's spelling, fireworks_ai/, as a model prefix
    ),
    metadata_namespaces=("fireworks_ai",),  # this endpoint IS Fireworks: its snapshot entries may describe and price the models
)

lm = dspy.LM("fireworks/accounts/fireworks/models/deepseek-v4p1-flash")   # or fireworks_ai/...
```

Do this at import time of your application, before the LMs that use it are constructed. **Each `dspy.LM` binds the registrations present when it is constructed and keeps them for its whole life** — selection, capabilities, pricing, and both its sync and async engines read that one binding. `register_provider(..., replace=True)` therefore changes only LMs constructed afterwards; it never moves an existing LM, and it can never leave one LM's sync calls on one definition and its async calls on another.

After that the provider behaves like a built-in one:

- `dspy.LM(..., api_key=..., api_base=..., timeout=...)` are honored; with no key given, `FIREWORKS_API_KEY` is read, and a missing key names that variable. A `FIREWORKS_API_BASE` variable in the environment does not redirect a declared provider (it does for registry providers, to preserve LiteLLM-era gateways): the declaration states its address itself.
- `engine="auto"` plans each request against the declared compat. A client setting only LiteLLM carries (`extra_headers`, `organization`, …) selects LiteLLM before any I/O — **at the declared address, with the declared credential**, through LiteLLM's generic OpenAI-compatible door. The LM's aliases are never handed to LiteLLM as a provider prefix, because LiteLLM may know that name as a different service. A refusal the declared compat produces (a reasoning dial on a wire with no reasoning field, say) is final: the declaration is the authority on what that server cannot do, and sending the request through LiteLLM anyway would drop the setting silently. `engine="lm15"` refuses both cases instead.
- `dump_state()`/`load_state()` and program `save()`/`load()` need nothing extra — the state is the model string. A registration is an environment fact, like the key variable: it must be in place when the program is loaded.

**Aliases are spellings, nothing more.** Which models a declared provider's endpoint supports, and what they cost, are separate statements:

- `metadata_namespaces=("fireworks_ai",)` says DSPy's model-metadata snapshot entries under that LiteLLM namespace describe (capabilities) and price this provider's models. Give it only when the endpoint really is that service. Without it nothing is inherited, and cost is reported as unknown.
- `supports=ModelSupport(function_calling=True, ...)` states model support for the whole provider; `models={"private-v1": ModelSupport(...)}` per model id. A snapshot entry, when one is found, is the more specific fact and wins over a statement. What neither states is treated as unsupported — the rule built-in providers already live with — and a statement never prices anything.
- The compat object still bounds everything: a wire with `thinking_format="none"` has no reasoning whatever the statement says.

What a declaration is not: a receipt. lm15's own registry entries are pinned from live captures; a declaration is your word, and the route says so (`Resolution.declared`). Registering the same registration twice is a no-op; a different one under the same id needs `replace=True`; a spelling lm15 or LiteLLM already uses is refused.

## What a custom engine owns

A custom engine is borrowed by `dspy.LM` and owns its connection. Three rules follow, and DSPy enforces each rather than guessing:

**Connection settings are refused.** `dspy.LM(engine=MyEngine(), api_key=...)` raises `ValueError`, and so do `api_base`, `base_url`, `timeout`, `headers`, `extra_headers` and the other client settings — on construction, on `copy()`, and on every call (`lm("hi", api_key=...)`), before any cache lookup. There is no channel from the LM to the engine for them, and dropping them silently would let a call run with the engine's key while the LM said another. Give them to the engine's constructor.

**The engine pair is one unit.** `async_engine=` is only accepted with a custom `engine=`, and each side must be its kind: the sync engine's `complete` is a plain function returning a `Response`, the async engine's is a coroutine function (`async def complete`). A sync method on the async side, or the reverse, is refused at construction rather than on the first call. `lm.copy(engine=...)` replaces both: the copy has no async engine unless you pass a new `async_engine=` in the same call. A copy that does not mention `engine` keeps the pair.

**Saving needs the engine's own state.** An engine is saved when it implements `dump_state() -> dict` (JSON-serializable, no secrets) and the classmethod `load_state(state) -> engine`:

```python
class MyEngine:
    def __init__(self, model="gpt-6-astra"):
        self.model = model

    def complete(self, request): ...

    def dump_state(self):
        return {"model": self.model}

    @classmethod
    def load_state(cls, state):
        return cls(**state)
```

`lm.dump_state()` then records `{"engine": {"class": "your.module:MyEngine", "state": {...}}}` (and `async_engine` likewise). Loading imports that class from the file, which is the same trust decision as a custom LM class, so it is gated the same way: `program.load(path, allow_unsafe_lm_state=True)` or `dspy.LM.load_state(state, allow_custom_lm_class=True)`. The class must be importable by that path in the process that loads the state. Define it at module level, not inside a function — `dump_state()` refuses a class it cannot import back — and for state that must outlive the session, in an importable module: a class defined in a script or notebook is recorded as `__main__:MyEngine`, which loads only in a process whose `__main__` defines it again. `dump_state()` also refuses engine state JSON cannot carry. An engine without the two methods cannot be saved either; `dump_state()` says so and the way out is to save the program without that LM and set it again after loading. Keep secrets out of engine state, as DSPy keeps `api_key` out of LM state.

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

Here `"pi"` is a DSPy model label for history; the engine's CLI arguments select the actual provider and model. `PiEngine` holds no state, so to make it saveable add `dump_state` returning `{}` and `load_state` returning `cls()`.

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
