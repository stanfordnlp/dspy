# MXC-confined CPython as DSPy's default

This draft **changes the default**, rather than introducing another public
interpreter. `LocalInterpreter` owns the existing persistent CPython worker,
JSON transport and MXC policy. `PythonInterpreter` remains the explicit
Deno/Pyodide alternative. There is no unconfined fallback.

**Release gate:** the companion `dspy-mxc-runtime==0.1.0a1` is not published.
Do not merge/release the required dependency until its supported wheels are
qualified, published and resolved into `uv.lock`. The package build sources
are in [`packages/mxc-runtime`](../../packages/mxc-runtime). Windows native
execution and a complete advertised-platform installation matrix remain gates.

## One worker, one policy owner

```text
RLM / Flex / ProgramOfThought / CodeAct
          |
explicit factory > active setting > LocalInterpreter
          |
existing JSON worker protocol <----> host tools / predictors / LM
          |                            (host authority)
MXC native executor
          |
current CPython + installed libraries + private work/tmp
```

DSPy stages its existing worker read-only, writes a policy in a private session
directory, clears ambient environment credentials and launches the wheel-owned
executor. Python prefixes are readable; arbitrary checkout/home paths are not
added just because the environment is a venv. Linux also masks `/etc`. Startup
must produce the existing ready message within ten seconds. There are no
runtime downloads, package installers, ACL mutations or elevated setup steps.

MXC owns OS enforcement: Linux bubblewrap/namespaces, macOS Seatbelt and Windows
process containment. DSPy supplies command quoting, the environment/path/network
policy, lifecycle and diagnostics. The runtime package owns binary selection,
checksums, notices, wheel tags and updates, not a second policy engine. No VM
image, snapshot, guest package manager, Node layer or source translator is added.

Flex still uses its shared facade: generated classes execute in Python while
predictors and nested agents call the host. This does not expose every DSPy API
through the facade. Native image sessions use installed `dspy.Image`, Pillow
and OpenCV; values cross JSON as explicit data, not pickle. Image imports remain
lazy. Editable dependencies outside Python prefixes, GPU access and arbitrary
system libraries are not automatically supported.

## Default selection and rollback

Only `None` means “use the configured default.” Explicit Deno now wins even
when a setting selects another backend. Resolution stays lazy so contexts and
nested code-executing modules use the active setting.

```python
import dspy

agent = dspy.RLM("question -> answer")  # MXC-confined LocalInterpreter
deno_agent = dspy.RLM("question -> answer", interpreter_factory=dspy.PythonInterpreter)

# Application-wide rollback, with dspy[deno] installed:
dspy.configure(interpreter_factory=dspy.PythonInterpreter)
```

`LocalInterpreter` no longer grants unrestricted host access. Install required
native libraries before execution and use host tools for deliberately granted
host IO. The companion diagnostic wheel lets DSPy import on unsupported hosts;
attempted native execution fails clearly. It does not automatically choose Deno.

## Explicit limitations

* This is shared-kernel process confinement, not VM isolation or a multi-tenant
  service boundary. Upstream baseline system paths remain readable.
* macOS retains broad reads including `/Library`, `/System` and `/private/etc`.
  Detached children can survive timeout/shutdown, continue allowed IO or hold
  deleted files open. This is an accepted documented limitation; DSPy does not
  add process scanning or a private Seatbelt policy to pretend to fix it.
* Python installation roots are readable and must not contain secrets.
* Host tools run outside the sandbox. A running Python callable can outlive a
  guest deadline; its late result is discarded, not its side effects cancelled.
* Protocol limits (16 Mi characters/message, four queued messages, 32 concurrent
  host calls, 8 Ki-character diagnostic tail) are not aggregate memory, process,
  disk or pre-capture output quotas. Execution deadlines remain caller-configured.
* Linux needs glibc >=2.34 and permitted user namespaces. Bundling static bwrap
  does not bypass AppArmor/container restrictions or bundle CPython itself.
* No Intel macOS executor is present in Microsoft's pinned release. Linux ARM64
  needs a qualified static bwrap build. Windows requires an available native
  containment tier: the policy rejects ACL-mutating fallback. Windows job
  kill-on-close behavior comes from upstream source, not a completed DSPy test;
  a hard kill between process creation and job assignment is a remaining race.

## Reproduce with locally built wheels

Prepare the runtime using its package README. Use an installed DSPy wheel, not
an editable link to code outside the guest's readable Python prefixes.

```sh
uv build --wheel --out-dir /tmp/dspy-candidate
uv pip install --python .venv/bin/python --find-links packages/mxc-runtime/dist \
  /tmp/dspy-candidate/dspy-*.whl
.venv/bin/python -m pytest -q tests/primitives/test_local_interpreter.py \
  tests/primitives/test_interpreter_factory_config.py

# Extra native-image integration dependencies and cmpnd tracing are required.
.venv/bin/python -m pytest -q scripts/mxc_interpreter/test_interpreter.py
.venv/bin/python -m scripts.mxc_interpreter.benchmark
```

The maintained tests exercise persistence, errors, typed outputs, async and
concurrent host tools, images, Flex, recursive RLM, live-listener denial,
synthetic files/environment, independent sessions and terminal cleanup. The
Linux detached-child test keeps an inode open after directory deletion and has
a survival control. These are bounded functional checks, not security-audit or
resource-exhaustion claims. `benchmark.py` is a five-session cached smoke
measurement, not a statistical portability or model-quality claim.

Earlier prototype evidence remains in Git history: native image/Azure tasks,
three seeded Flex/GEPA runs and platform probes. Those results do not qualify
the changed launcher/package by themselves. The current PR must report exact
new wheels, executed checks and unsupported platforms separately.
