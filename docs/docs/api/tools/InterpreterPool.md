# dspy.InterpreterPool

`InterpreterPool` is an `interpreter_factory` that keeps interpreters started ahead of need. Each call still returns
a fresh interpreter that has run no code, but usually one that has already started, so a `dspy.Flex` forward or a
`dspy.RLM` call doesn't wait for a sandbox to boot. With the default `PythonInterpreter`, a pooled interpreter has also
loaded dspy, which is the first thing a `Flex` forward imports.

```python
import dspy

pool = dspy.InterpreterPool(size=2)
program = dspy.Flex("question -> answer", interpreter_factory=pool)
```

The pool starts `size` interpreters in the background and starts a replacement whenever it hands one out. When none
is ready, a call makes a new interpreter on the spot, as the plain factory would.

Pick `size` by how calls arrive. One interpreter is enough when each call spends longer in LM calls than a sandbox
takes to start (well under a second). Calls that run back to back with little LM time, or in parallel threads such
as `dspy.Evaluate(num_threads=...)`, need one for each call in flight.

!!! warning "Idle interpreters hold memory"
    Every pooled interpreter is a running sandbox process. A `PythonInterpreter` with dspy loaded holds about 400 MB,
    so size the pool to your machine. Idle interpreters are shut down by `close()`, when the pool is used as a
    context manager, and at exit.

`factory` must return interchangeable interpreters, because the pool calls it before the interpreter is needed. A
factory whose interpreters depend on when it is called, for example on the current input, should not be pooled.

<!-- START_API_REF -->
::: dspy.InterpreterPool
    handler: python
    options:
        members:
            - __call__
            - close
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
<!-- END_API_REF -->
