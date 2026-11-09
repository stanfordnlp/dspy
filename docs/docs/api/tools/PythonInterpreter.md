# dspy.PythonInterpreter

## Deno Installation

`PythonInterpreter` uses Deno and Pyodide to run Python in a local WASM sandbox. The recommended installation
keeps Deno in the same Python environment as DSPy:

```bash
pip install "dspy[deno]"
```

The `deno` extra installs the official Deno Python distribution (`>=2.4.5,<3.0.0`). DSPy prefers that managed
binary when it is installed, so Python dependency locking also locks the Deno runtime. The extra provides binaries
for macOS x86-64/arm64, glibc Linux x86-64/arm64, and Windows x86-64 and adds approximately 40–50 MiB to the
environment.

On other platforms, install a compatible Deno 2.x release (`>=2.0.0,<3.0.0`) using the
[Deno installation instructions](https://docs.deno.com/runtime/getting_started/installation/). DSPy falls back to
the `deno` executable on `PATH`. An explicit `deno_command` passed to `PythonInterpreter` takes precedence over
both options.

DSPy disables ambient `deno.json`, lockfile, `package.json`, and local `node_modules` discovery for its default
runner. A `package.json` in the current directory or an ancestor therefore cannot redirect the sandbox's pinned
Pyodide dependency. No `DENO_NO_PACKAGE_JSON` environment variable is required.

## Startup

Pyodide normally compiles each module of its standard library every time it starts. The first time a
`PythonInterpreter` starts on a machine, DSPy byte-compiles Pyodide's standard library once into
`$DSPY_CACHEDIR/sandbox` (`~/.dspy_cache/sandbox` by default), and every later interpreter starts from it, in about
a quarter of the time. If that cache can't be written, interpreters start from Pyodide's own standard library.

Where writing files is unwelcome, turn the cache off with `PythonInterpreter(sandbox_cache=False)`, or for every
interpreter, including those Flex and RLM make, with the environment variable `DSPY_SANDBOX_CACHE=0`. DSPy then
writes nothing to `$DSPY_CACHEDIR/sandbox`, and Pyodide starts from its own standard library. Sandboxes that import
dspy still need its bundle (below): each process builds its own in a temporary directory, removed when it exits.

`provide_dspy()` lets sandboxed code `import dspy`, with the host's dspy source and Pyodide builds of its
dependencies. `dspy.Flex` and `dspy.RLM` call it on the interpreters they create. Such an interpreter starts from a
snapshot of one that has already imported dspy, which DSPy records once per dspy source, so it starts faster than a
plain one. To start interpreters before they are needed, see [`InterpreterPool`](InterpreterPool.md).

## Snapshots

`snapshot()` returns the sandbox's Python state: every variable, function, and object its code made, as compressed
bytes. `PythonInterpreter(snapshot=...)` starts a new sandbox from them, in another process if you like, instead of
from a fresh one:

```python
with dspy.PythonInterpreter() as interpreter:
    interpreter("rows = [1, 2, 3]")
    state = interpreter.snapshot()

with dspy.PythonInterpreter(snapshot=state) as interpreter:
    interpreter("print(sum(rows))")  # "6"
```

It is a Pyodide memory snapshot, so it holds everything in the sandbox's memory, environment variables passed in
included: store it as you would a secret. Files the code wrote are not in it. It restores only in an interpreter that
starts as the one that made it did (the same dspy and Pyodide, `sandbox_cache`, and `provide_dspy()`), and a sandbox
whose Python holds JavaScript objects (from `js`) cannot be snapshotted; both raise `CodeInterpreterError`.

## Execution Instructions

`PythonInterpreter.execution_instructions` describes stable constraints of its Pyodide execution environment. It
is class metadata, so code-generating modules can inspect it without starting Deno or allocating an interpreter.
`RLM` includes these instructions in its action prompt, which adapters render in the model's system prompt.

Custom interpreter factories may expose their own `execution_instructions` string. This metadata is optional; a
factory without it remains valid and uses RLM's generic action prompt.

<!-- START_API_REF -->
::: dspy.PythonInterpreter
    handler: python
    options:
        members:
            - __call__
            - execute
            - provide_dspy
            - shutdown
            - snapshot
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
