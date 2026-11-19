# Evaluating a DSPy program with Harbor

`dspy.experimental.harbor.HarborEvaluation` wraps a child module in a complete
Harbor trial. Harbor provisions the environment, runs its verifier, saves trial
artifacts, and tears down the environment. DSPy optimizes the child, not the
verifier or the task infrastructure. No changes to `Example` or optimizers are
required.

This experimental integration targets Harbor 0.24's public `BaseAgent`,
`TrialConfig`, and `Trial.create()` / `Trial.run()` interfaces. It requires
Python 3.12+ and a separately installed Harbor:

```shell
pip install 'harbor>=0.24,<0.25'
```

Harbor's environment provider must also be configured (for example, a working
Docker daemon for Docker tasks). Installing the package does not provision a
sandbox or download a benchmark's external assets.

## Bind inputs and submit the child's answer

Each benchmark has its own tools and submission contract. Supply an async
callback that receives the **current candidate program**, Harbor's instruction,
the live `BaseEnvironment`, and a list of `dspy.Tool` objects. The callback calls
the child and submits its output. It must use the supplied program rather than
capture the original child in a closure, or optimization would evaluate the wrong
candidate.

This example assumes a task that accepts an answer at `/app/answer.txt`:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import dspy
from harbor.models.trial.config import TaskConfig, TrialConfig
from dspy.experimental.harbor import HarborEvaluation


async def run_answer(program, instruction, environment, tools):
    prediction = await program.acall(question=instruction)
    with TemporaryDirectory() as directory:
        submission = Path(directory) / "answer.txt"
        submission.write_text(prediction.answer)
        await environment.upload_file(submission, "/app/answer.txt")
    return prediction


program = dspy.ChainOfThought("question -> answer")
evaluation = HarborEvaluation(program, run=run_answer, reward_key="reward")
examples = [
    dspy.Example(
        task=TrialConfig(
            task=TaskConfig(path=Path("path/to/harbor-task")),
            trials_dir=Path("harbor-trials"),
        ),
    ).with_inputs("task")
]

# Configure your DSPy LM before running evaluation.
result = evaluation(**examples[0].inputs())
print(result.score, result.trial_result.trial_uri)


def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    return prediction.score


evaluator = dspy.Evaluate(devset=examples, metric=metric, num_threads=1)
evaluator(evaluation)

# With separate training and validation examples:
# compiled = dspy.GEPA(metric=metric, ...).compile(
#     evaluation, trainset=train_examples, valset=val_examples
# )
# optimized_program = compiled.program
```

For an environment-dependent composed child, construct tool callables in the
callback using `environment.exec`, `upload_file`, or the other public environment
methods, and pass them to the child's outer interface. The child need not pass
those Python resources into an LM signature. Do not store trial environments on
the module or mutate shared tools while examples run concurrently.

The callback uses async child execution so Harbor can enforce timeouts and
cancellation. A synchronous blocking call inside this callback blocks Harbor's
event loop; offloading it to a thread does not make the underlying work cancellable.
Prefer a child with native `aforward` / `acall` for interactive tasks.

Trials and callbacks run on one process-wide background event loop, shared by
all wrapper copies and successive optimization rounds. Harbor providers retain
async locks and clients across trials, so starting an independent event loop in
every evaluation worker is unsafe. The bridge preserves each caller's DSPy
context and trace while running concurrent trials on this shared loop. Create
loop-bound resources inside the callback; do not capture clients or async locks
created on another loop. Avoid mixing these trials with native Harbor trials
running on other loops in the same process. Cancelling `acall` waits for Harbor
cleanup before returning; the synchronous API blocks until its trial completes.

## MCP servers and skills

Harbor merges task and runtime MCP configurations and provisions runtime skills.
The bridge uses the resulting public agent fields, `mcp_servers` and `skills_dir`,
to build the callback's `tools` list:

- **MCP:** stdio, SSE, and Streamable HTTP (including Harbor's `http` alias).
  Commands and connections run **inside the task environment**, so sandbox paths,
  localhost, Compose sidecar DNS, and sandbox environment variables retain their
  meaning. Sessions persist across calls within one agent phase. Tools are named
  `mcp__<server>__<tool>`; non-alphanumeric characters other than `_` and `-` become
  underscores. Collisions raise instead of silently replacing tools.
- **Results:** DSPy's existing `Tool.from_mcp_tool(..., result_mode="structured")`
  preserves structured content, falls back to text/non-text content, and raises
  on MCP tool errors. Input schemas come from the server.
- **Skills:** `harbor_read_skill(name=...)` returns a discovered skill's complete
  `SKILL.md` and its sandbox path. Its argument schema lists the available skill
  directory names. Companion files and scripts remain in the sandbox; the bridge
  does not execute them or inject skill text into every predictor automatically.

For a child whose outer interface accepts runtime tools, the callback can call:

```python
prediction = await program.acall(question=instruction, tools=tools)
# Submit prediction using this task's contract, then return it.
```

This does **not** add a `tools` argument to stock `ReAct.forward`. The composed
child owns its tool-selection loop and how it uses the available schemas. Keep
its optimizable predictors on the child; do not construct a new independent
agent in the callback on every evaluation or mutate shared initialization-time
tools across concurrent examples.

MCP requires `mcp` on the DSPy host and **Python 3.10+ named `python3`, with
`mcp>=1.26,<2`, in the Linux task image**. Install these dependencies when building
the image. The bridge does not install packages or bypass task network policies.
Skills alone require only the environment's shell, `find`, and `cat`.

The MCP client helper uses only public `upload_file` and `exec` environment
methods. It communicates over a private sandbox Unix socket, not an exposed
network port. Sessions, helper processes, and temporary files are released on
normal completion, callback failure, and cancellation, before Harbor verifies
the submission. Shutdown has a 20-second grace period; failures raise and leave
diagnostics in the sandbox. Harbor still owns environment teardown.
Tool discovery and session startup count against the agent's run timeout.
Multi-step tasks get fresh MCP sessions per agent phase, not conversation resume.
Cancellation closes the bridge's clients; it does not guarantee cancellation or
rollback of an operation already dispatched to an HTTP server.

This is an MCP **tool** bridge, not a complete MCP client UI: resources, prompts,
sampling, elicitation, interactive OAuth, and dynamic tool-list notifications
are not exposed. It uses Harbor's server configuration as given; Harbor 0.24's
`MCPServerConfig` has no per-server headers, environment, or working-directory
fields. HTTP servers must already be running, typically as task services.

## Execution and results

The wrapper copies the supplied `TrialConfig`, replaces its agent with an
in-process DSPy custom agent, and assigns a unique trial name. Agent execution
settings such as environment variables, timeout limits, and log filters are
preserved. The original agent's name, import path, model, and constructor kwargs
are replaced; configure the child's LM through DSPy instead. Task, environment,
verifier, output directory, and other trial settings stay under Harbor's control.
The agent bridge resolves the current candidate through an execution-local
context; it is not a standalone CLI agent or a remote-worker protocol.

Results include:

- `score`: the requested named reward, defaulting to `reward`.
- `rewards`: all named verifier rewards.
- `predictions`: child predictions, in agent-phase order.
- `trial_result`: Harbor's complete result, including artifact locations.

Verification completes before the wrapper returns. Metrics can therefore read
the score repeatedly without retaining a live environment or rerunning tests.
Task-specific feedback can be read from Harbor's saved verifier artifacts.

Harbor-reported errors raise, as do missing verification results or reward keys.
A valid zero reward remains zero. DSPy's evaluator may apply its configured
failure-score policy to raised errors; use its error settings when diagnosing
infrastructure failures.

The bridge implements ordinary agent `run`, not Harbor's trajectory-resume,
handoff, or simulated-user protocols. Do not enable those capabilities for it.
Resume/load settings are forwarded so Harbor can reject unsupported capabilities,
rather than silently running a different protocol. Install-only and disabled
verification trials cannot produce an evaluation score; the wrapper raises after
Harbor completes their lifecycle.
Keep judge/reference data outside the child and outside its predictor traces.
Store only copyable configuration on callbacks/modules; create and release any
additional host-side resources within the callback.

## What this wrapper does not map

Running a `Trial` delegates its lifecycle to Harbor; it does not implement every
Harbor agent, job, or deployment interface:

| Harbor surface | Boundary of this integration |
| --- | --- |
| Native session resume, trajectory loading, handoff | Unsupported capabilities; a fresh callback is not resumed agent state. |
| Simulated-user bridge protocols | Not implemented; a simulator can be composed explicitly inside the callback/child, but that is not Harbor's native bridge. |
| ATIF trajectories and `AgentContext` usage/model accounting | Not populated from DSPy traces or LM usage. DSPy traces remain available for optimization. |
| Native agent configuration and installation | The original agent implementation and constructor kwargs are replaced. Image dependencies and benchmark-specific setup remain the task/callback's responsibility. |
| Job/queue scheduling, retries, shared concurrency pools | This calls `Trial` directly, not `Job` or `TrialQueue`. `agent.n_concurrent` does not throttle DSPy evaluations; use DSPy's evaluation concurrency. |
| Remote workers / standalone Harbor CLI agent | The bridge resolves an in-process candidate, not a serialized remote program. |
| Live lifecycle hooks / log subscriptions | No wrapper-level subscription API; completed Harbor results and artifacts are available. |
| Install-only, verification disabled, regrade without agent execution | Harbor can execute these modes, but this wrapper cannot return a child evaluation score and raises. |
| Windows task containers | Not advertised or supported. |

Provider-specific features remain subject to Harbor's provider support. In
particular, agent environment variables scope sandbox commands; they do not
configure the host-side DSPy LM. Docker coverage is not evidence for every cloud
provider, network policy, dataset, or benchmark verifier.

## Configuration coverage

The opt-in Docker suite uses disposable tasks, real containers, and Harbor's
built-in script verifier. A composed DSPy child calls an environment-backed tool
and submits a deterministic answer through the callback; no paid LM is needed.

```shell
# Python 3.12+, Harbor 0.24, pytest, pytest-asyncio, Docker Engine,
# Docker Compose and Buildx must be installed; the daemon must be accessible.
HARBOR_DOCKER_TESTS=1 python -m pytest tests/experimental -v
```

Covered configurations include shared verification, named JSON rewards, separate
verification with declared artifact transfer, a dedicated verifier Dockerfile,
and multi-step tasks with both mean and final reward aggregation. The suite also
checks agent-only environment variables do not reach the verifier, predictor
traces survive, and caller-owned configurations remain unchanged.
The GEPA regression runs actual `compile()` calls with two evaluation threads for
both single-step and multi-step Docker tasks. A deterministic LM and instruction
proposer isolate the integration contract: Harbor rewards select the changed
child, the original stays unchanged, and the extracted child works after saving
and loading. These tests prove optimizer wiring, not real-LM benchmark gains.
Additional Docker cases cover stateful stdio/SSE/Streamable HTTP tools, Compose
sidecar connectivity, task/runtime server overrides, runtime skills, structured
results, MCP errors, and cleanup before verification and environment teardown.
Failure cases include partial initialization, startup timeout, callback failure,
cancelled tool calls, and Harbor's agent timeout. A task-bundled skills case uses
an image without Python or MCP dependencies.

For example, a separate verifier receives declared artifacts, not the complete
agent filesystem. The source path is restored in the verifier; `destination`
controls the artifact's saved path on the host:

```toml
artifacts = [{source = "/app/answer.txt", destination = "answer.txt"}]

[verifier]
environment_mode = "separate"
```

For multi-step tasks, Harbor invokes the callback for each step in the same agent
environment. The wrapper returns all child predictions and Harbor's aggregate
reward. A fresh `run` does not imply a fresh environment or resumed conversation:

```toml
multi_step_reward_strategy = "mean" # or "final"

[[steps]]
name = "first"

[[steps]]
name = "second"
```

These tests establish Docker task execution, not cloud-provider compatibility,
registry downloads, benchmark asset availability, or benchmark score parity.
