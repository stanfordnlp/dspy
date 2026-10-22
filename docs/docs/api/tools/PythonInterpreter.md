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
