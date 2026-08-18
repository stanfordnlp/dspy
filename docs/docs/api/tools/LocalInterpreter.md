# dspy.LocalInterpreter

`LocalInterpreter` executes Python directly in the DSPy host process. It starts immediately, preserves Python state
between calls, can import the host environment's installed packages, and requires no Deno runtime.

!!! danger "Trusted code only"

    `LocalInterpreter` is **not a sandbox**. Executed code has the same filesystem, environment, credential, network,
    and process access as the DSPy application. Do not use it for untrusted or model-generated code unless that code
    is already trusted to run with full host authority. Use [`PythonInterpreter`](PythonInterpreter.md) for local
    sandboxed execution.

```python
import dspy

with dspy.LocalInterpreter() as interpreter:
    interpreter.execute("import math\nvalue = math.sqrt(1764)")
    assert interpreter.execute("value") == 42
```

`LocalInterpreter.execution_instructions` describes its persistent host-Python environment. Passing
`interpreter_factory=dspy.LocalInterpreter` to `RLM` therefore adds that description to the action prompt, but it
does not create a security boundary around generated actions. Executions are serialized across `LocalInterpreter`
instances because Python's standard-output redirection is process-global.

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
