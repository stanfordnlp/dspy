# dspy.LocalInterpreter

`LocalInterpreter` is the default for code-executing DSPy modules. It runs the current CPython executable in an
MXC-confined process, using installed Python packages rather than Pyodide builds. State and imports persist until
`shutdown()`. Host tools cross a JSON protocol and retain host authority.

```python
import dspy

rlm = dspy.RLM(
    "question: str -> answer: int",
    interpreter_factory=dspy.LocalInterpreter,
)
```

## Installation and isolation

The `dspy-mxc-runtime` dependency supplies a pinned MXC executor and, on Linux, static bubblewrap. Runtime startup
does not download software, change host security policy, or fall back to unconfined Python. Linux needs glibc >=2.34
and permitted unprivileged namespaces; restrictive containers/AppArmor policies can prevent startup. macOS uses
Seatbelt; the upstream release has an ARM64 executor but no Intel Mac executor. Windows uses native process
containment and rejects fallbacks requiring host ACL mutation; ordinary pip installation cannot enable missing OS
facilities. A diagnostic-only wheel allows DSPy itself to install on unsupported hosts, but executing code fails.

!!! warning "Platform qualification is a release gate"
    This default-switch draft depends on an unpublished companion package. Linux x86-64 is locally tested;
    Windows is not qualified. Do not describe the draft as a universal `pip install dspy` release.

The worker can read the Python environment and upstream OS baseline paths, and write its private work and temp
directories. Do not keep secrets in the Python installation. Host environment credentials are scrubbed and direct
IP networking is denied. Install native libraries before starting DSPy; editable packages outside the allowed
Python prefixes, GPUs and arbitrary system dependencies are not automatically accessible. Session files disappear
at shutdown. This changes `LocalInterpreter`'s former unrestricted host-access behavior.

!!! warning "Process confinement is not a VM or a resource quota"
    MXC shares the host kernel. macOS retains broad reads of system paths including `/Library`, `/System` and
    `/private/etc`. Detached macOS children can survive shutdown or timeout and keep consuming resources or holding
    deleted files open. Finish and reap child processes; use a separately qualified boundary if whole-job
    termination is required. No aggregate memory, process-count or disk quota is promised.

To select the existing Deno/Pyodide backend explicitly, install `dspy[deno]` and pass
`interpreter_factory=dspy.PythonInterpreter`, or use `dspy.configure(interpreter_factory=dspy.PythonInterpreter)`.
An explicit constructor factory wins over the active setting. No automatic fallback changes the isolation model.

Inputs, host-tool arguments/results, and structured outputs must be JSON-compatible. The worker uses the current
Python executable.
`execution_timeout` includes time spent in host tools and terminates the worker promptly when the deadline expires.
Python cannot forcibly stop a running host callable, so that callable may finish in a detached daemon thread; its
result is discarded and the interpreter session remains terminal.

The host accepts guest protocol messages up to 16 Mi characters and at most 32 outstanding host-tool calls per
execution. Invalid tool requests or exceeded limits terminate the session. Completed calls do not count toward the
concurrency limit. The response queue is bounded; these transport protections do not limit aggregate guest memory,
process count, filesystem usage, or output written before it reaches the host.

Parallel guest code is supported when all threads finish before the current `execute()` call returns, such as a
context-managed `ThreadPoolExecutor`. Leaving a guest thread running makes the session terminal because that thread
could otherwise mutate state, write output, or invoke tools during a later execution. Linux PID namespaces and
Windows MXC jobs provide descendant cleanup; macOS process-group cleanup has the detached-child limitation above.

`LocalInterpreter.execution_instructions` is stable class metadata. `RLM` adds it to the action prompt without
starting a worker.

<!-- START_API_REF -->
::: dspy.LocalInterpreter
    handler: python
    options:
        members:
            - __call__
            - execute
            - shutdown
            - start
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
<!-- END_API_REF -->
