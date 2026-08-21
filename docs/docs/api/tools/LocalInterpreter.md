# dspy.LocalInterpreter

`LocalInterpreter` runs Python in one persistent local CPython worker. It is useful when generated code needs
ordinary Python compatibility plus separation from the DSPy process's memory, stdout, and lifecycle. State and
imports persist until `shutdown()`, host tools cross a JSON protocol, and `execution_timeout` can terminate a stuck
worker.

```python
import dspy

rlm = dspy.RLM(
    "question: str -> answer: int",
    interpreter_factory=dspy.LocalInterpreter,
)
```

!!! warning "A subprocess is not a security sandbox"
    Generated code retains the host user's filesystem, environment, credentials, subprocess, and network authority.
    Use the default [`PythonInterpreter`](PythonInterpreter.md) or a remote sandbox for untrusted code. The subprocess
    boundary protects ordinary host memory and stdout from accidental mutation; it does not contain hostile code.

Inputs, host-tool arguments/results, and structured outputs must be JSON-compatible. The worker uses the current
Python executable.
`execution_timeout` includes time spent in host tools and terminates the worker promptly when the deadline expires.
Python cannot forcibly stop a running host callable, so that callable may finish in a detached daemon thread; its
result is discarded and the interpreter session remains terminal.

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
