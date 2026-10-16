# MXC-backed CPython: prototype and default-migration plan

**Draft, Linux-only, opt-in. Not a production security boundary.** This change
does not change DSPy's default interpreter, exports, dependencies, or supported
platforms. It proposes Microsoft [MXC](https://github.com/microsoft/mxc) as the
launcher for DSPy's existing persistent CPython worker, rather than another
Python implementation or a new tool protocol.

## What this draft establishes

```text
RLM / Flex / ProgramOfThought
           |
     CodeInterpreter
           |
 LocalInterpreter protocol <----> registered host tools / LM
           | JSON pipes                 (outside sandbox)
     MXC native launcher
           |
   confined CPython worker
   + installed Python packages
   + private session filesystem
```

`MXCInterpreter` subclasses `LocalInterpreter`, overriding launch and session
cleanup. The existing worker source is staged read-only for
each session; its protocol, output capture, callbacks, tool dispatch, deadlines,
and process-group cleanup are reused. Execution deadlines now also cover blocked
command writes, rather than starting to protect the caller only while receiving.
There is no `runner.js`, Node, Pyodide,
source translation, or second evaluator on this path.

Flex still installs its existing shared DSPy facade: generated classes run as
ordinary Python, while predictors and sub-agents call the host. This does not
make every DSPy API available in the guest. Provider credentials stay on the
host; direct guest networking is disabled. Tools are explicitly granted host
capabilities, not operations confined by MXC.

## Image stack

This integration stacks RLM image support on the MXC prototype. Native sessions
use the installed `dspy.Image` class, not a `DSPyImage` stand-in. Generated code
uses `dspy.Image(pil_image_or_encoded_bytes)` and `image.to_pil()`. The Deno path
retains a lightweight representation behind that same editing interface; it does
not load the full DSPy package or claim full Pydantic-class parity.

Images cross JSON pipes as `{"url": ...}` data. Typed RLM outputs reconstruct
real host images; nested predictors restore directly Image-annotated inputs.
Bridged predictor results remain JSON data: wrap an image field with
`dspy.Image(result.edited)` before further guest-side editing. This is not a
general native-object serializer or arbitrary nested custom-type restoration.
Remote URLs are not downloaded by `to_pil()`; download on the host explicitly.

The native suite checks actual class identity, asymmetric Pillow crops/OpenCV
edits, single/batched vision requests, typed output and recursive RLM images.
A live Azure GPT-4o task cropped a 120×80 red/blue image to its blue 60×80 right
half, queried vision and returned the correct typed image and color in four
REPL iterations (five LM calls), without execution errors. Its first version
failed because the prompt advertised `DSPyImage.from_pil` after aliasing that
name to the real class. Removing that conflicting API fixed the same task.

Image imports are lazy: the ordinary no-LM Flex benchmark remains about 53 ms
on this Linux host, rather than the approximately 663 ms measured with an eager
DSPy import. Image sessions still pay the real package's import cost once.

## Run the prototype

Use a disposable Linux x86-64 development environment with working unprivileged
namespaces, `bwrap >= 0.5.0` on PATH, and this DSPy checkout installed. Do not relax
host security settings to make it run. macOS and Windows are rejected, and missing
dependencies fail rather than falling back to unsandboxed execution.

```sh
mkdir -p /tmp/dspy-mxc-runtime
curl -fL https://github.com/microsoft/mxc/releases/download/v1.0.0/mxc-release-binaries.zip \
  -o /tmp/dspy-mxc-runtime/release.zip
echo '0bb483b3592d0f663f2c5a7a2ae17db89cc04b82238684b0a1a9384b0fe828ed  /tmp/dspy-mxc-runtime/release.zip' | sha256sum -c -
unzip -p /tmp/dspy-mxc-runtime/release.zip x64/lxc-exec > /tmp/dspy-mxc-runtime/lxc-exec
echo '31b747a6a31b20fabd9e19ec30dfff3acf844c25ae4507a41ccac76e11916b2a  /tmp/dspy-mxc-runtime/lxc-exec' | sha256sum -c -
chmod 755 /tmp/dspy-mxc-runtime/lxc-exec
export MXC_EXECUTABLE=/tmp/dspy-mxc-runtime/lxc-exec

# Additional test-only packages; no DSPy dependency changes in this draft.
uv pip install --python .venv/bin/python cmpnd numpy pillow opencv-python-headless pytest pytest-asyncio
# Native images need an installed DSPy wheel, not an editable link outside the
# runtime prefixes. The prototype launcher itself still lives in this checkout.
uv build --wheel --out-dir /tmp/dspy-mxc-wheels
uv pip install --python .venv/bin/python --no-deps --reinstall /tmp/dspy-mxc-wheels/*.whl
.venv/bin/python -m pytest scripts/mxc_interpreter/test_interpreter.py \
  tests/primitives/test_local_interpreter.py tests/primitives/test_interpreter_factory_config.py -q
.venv/bin/python -m scripts.mxc_interpreter.benchmark
```

The official release ZIP is about 536 MB, but the extracted executor is
12,461,672 bytes and links to libc/libgcc_s. The ZIP is not the runtime payload.
Neither binaries nor upstream source are vendored here. Runtime downloads and
package installation never happen during interpreter execution.

```python
import os
import dspy
from scripts.mxc_interpreter.interpreter import MXCInterpreter

factory = MXCInterpreter.configured(
    runtime=os.environ["MXC_EXECUTABLE"], execution_timeout=30,
)
with factory() as interpreter:
    interpreter.execute("value = 17")
    assert interpreter.execute("value + 26") == 43

# Configure an LM as usual before invoking the agent.
with dspy.context(interpreter_factory=factory):
    agent = dspy.RLM("question -> answer")
```

Packages come from the current Python environment. Its prefixes are readable
inside the guest and must not contain secrets. Editable packages outside those
prefixes, system libraries, GPUs and arbitrary host paths are not automatically
supported. Session files persist between execute calls, not after shutdown.

## Accepted macOS lifecycle limitation

For a future macOS backend, timeout and shutdown do **not** guarantee termination
of every descendant. A subprocess that creates a new Unix session (for example,
with `subprocess.Popen(..., start_new_session=True)` or `os.setsid()`) can survive
the worker and MXC launcher. It can continue consuming resources, writing to
allowed paths, or holding files open after those paths are deleted. Do not detach
background processes; finish and reap subprocesses before returning from execution.

This is a lifecycle limitation, not a demonstrated filesystem/network escape.
Our probe verified survival, but did not retest the survivor's access restrictions.
The related [MXC report #1459](https://github.com/microsoft/mxc/issues/1459) also
observed survival through the Node SDK and verified that its surviving process
still received permission errors for forbidden file access. Neither result is a
general guarantee against resource exhaustion or all forms of sandbox escape.

We accept this documented limitation rather than make complete detached-process
cleanup a prerequisite for macOS support. Applications requiring guaranteed
termination of every job descendant need a separately verified execution boundary
that provides that guarantee. DSPy will not add process scanning or a custom
Seatbelt policy as a purported fix. This decision does not waive the remaining
installation, isolation or platform-integration checks: the current prototype
still rejects macOS and Windows.

## Evidence and its limits

The opt-in suite has 20 cases covering native Python inheritance, dataclasses,
context managers, item access/assignment, NumPy/Pillow/OpenCV, persistent state
and files, recoverable errors, concurrent/async tools, typed SUBMIT, Flex modules
and predictors, recursive RLM, ReActV2 delegation, async RLM, unexpected worker
exit, blocked host tools and host interruption. Image checks cover the stacked
typed image integration above. Test LMs are scripted/DummyLM, with cmpnd callbacks
enabled; the Azure image task and Azure/GEPA results are separate live checks.

Isolation checks use synthetic secrets, symlinks, native `ctypes` IO, a reachable
host listener control, concurrent sessions, and detached children. Child cleanup
has a survival control and an open inode retained after workspace deletion, so
deletion cannot hide a surviving process. These are functional probes, not an
adversarial security audit. The suite is outside `tests/`. The added
`.github/workflows/mxc.yml` provisions the checksum-pinned official executor and
distro bubblewrap, then tests an installed DSPy wheel outside the checkout on
Ubuntu 22.04/Python 3.11. Its wheel-install/test portion also passed locally in a
clean uv-managed Python 3.12 environment (52 tests including LocalInterpreter).
This local run is not evidence of a completed GitHub Actions run.

Two earlier clean Python 3.11 installations, including a fully offline replay
of 69 hash-pinned wheels, validated the **OS + pip** installation route. The
Python 3.12 run additionally exposed a dangling uv runtime-alias path; the launcher
now mounts that alias read-only without exposing its parent cache. There is
still no published, supported MXC Python runtime wheel or pip-only installation.
The CI job pins MXC and its test payloads, but resolves DSPy's transitive Python
dependencies and current distro packages: it is a compatibility check, not a
bit-for-bit reproducible build or immutable OS image.

Cached-runtime medians on Linux 6.1, Python 3.11.6, bwrap 0.8.0, MXC 1.0.0:

| Runtime | Startup + first expression | Warm expression | Simple Flex forward |
| --- | ---: | ---: | ---: |
| Local CPython, **unsandboxed control** | 36.0 ms | 0.46 ms | 42.3 ms |
| MXC/CPython | 48.6 ms | 0.48 ms | 55.9 ms |
| Deno/Pyodide | 2035.8 ms | 3.17 ms | 1979.3 ms |

`benchmark.py` runs five sessions and 50 warm expressions per runtime. Flex sums
squares without an LM call. Timings exclude installation and importing DSPy;
they do not measure optimization quality or guarantee a speedup on other hosts.
An earlier comparison on the experimental Monty/image stack measured MXC at
61 ms and SRT at 341 ms startup. That comparison favored MXC for network-disabled
execution, but was not a profiler attribution. SRT's hostname-filtered guest
networking may suit different requirements. This draft maintains only MXC.

### Live Flex/GEPA compatibility runs

Seed 20260926 selected Iris, Oolong and BFCL parallel non-native from a six-task
dspy-bench pool. Each run used 16 training, 8 validation and 8 held-out examples,
a 112-metric-call budget, four evaluation threads, Azure-served GPT-4o for task
calls and GPT-5.6 Luna for reflection, with cmpnd callbacks enabled. Oolong used
the first 180 streamed rows with 1K–8K contexts before splitting.

| Task | Candidate sources | Optimization forwards | Execution errors | Held-out baseline MXC / Deno | Held-out winner MXC / Deno |
| --- | ---: | ---: | ---: | ---: | ---: |
| Iris | 11 | 112 | 0 | 1.000 / 1.000 | 1.000 / 1.000 |
| Oolong | 10 | 112 | 0 | 0.820 / 0.820 | 0.719 / 0.599 |
| BFCL, after serialization fix | 14 | 112 | 0 | 1.000 / 1.000 | 1.000 / 1.000 |

The first BFCL attempt had 100 failures in 100 optimization forwards: Flex passed
`Tool` objects to the native JSON-only variable transport. All 64 distinct failed
source/input pairs succeeded on Deno. Normalizing inputs in Flex fixed that
backend dependency; a full new optimization then completed without execution
errors. The retained native image-input regression exercises the same boundary.

Generated code included modules, nested predictors/RLMs, imports, branching,
loops, comparisons, tuples and item access. These small runs demonstrate useful
compatibility coverage, not universal quality parity. The Oolong winner embeds
an RLM: backend-specific execution instructions change its LM requests, so this
is not a deterministic replay of identical inner programs. Both winner scores
are below the baseline on this held-out subset; do not claim optimization always
improves quality or attribute the score difference to execution speed.

## Changes required before changing the default

These are proposed follow-up stages, not capabilities delivered by this draft.

### 1. Finish the native backend and lifecycle contract

- [x] Demonstrate a persistent native worker with the existing transport.
- [ ] Fold verified confinement into `LocalInterpreter` rather than adding a
  second public native backend. This changes its host-access contract and needs
  the packaging and platform gates below; never silently fall back to an
  unsandboxed worker when confinement is unavailable.
- [ ] Finalize runtime discovery, version checks, diagnostics, finite default
  deadlines, interruption, failed startup, unexpected worker exit, repeated
  shutdown, and platform-specific descendant cleanup, subject to the accepted
  macOS limitation above. Keep protocol ownership in
  `local_interpreter.py` / `local_interpreter_worker.py`, not in each launcher.
- [ ] Test callbacks, cancellation and concurrent tools under the real launcher.
  Document that a host callable can outlive an execution timeout; guest process
  termination cannot stop a Python thread executing on the host.

### 2. Establish a defensible sandbox policy

- [x] Mask MXC's default host `/etc` mount with an empty directory; verify native
  package imports and absence of host configuration through the real launcher.
- [ ] Resolve the remaining broad baseline mounts: system binaries/libraries
  remain readable. This prototype is **not an exact runtime-only allowlist**.
  A separate v1.0.0 probe found denying `/bin` canonicalizes to `/usr/bin` while
  MXC's separately mounted `/bin` stays readable. Resolve or explicitly exclude
  those semantics before promising path-denial guarantees.
- [ ] Specify private workspace, temporary files, read-only runtime, explicit
  user file mounts, symlink handling, environment/credential isolation, and
  network denial. Test native syscalls, child processes, shared temp paths and
  cross-session access. Do not implement an ad hoc policy engine in DSPy.
- [ ] Set and verify memory, process-count, disk and output limits. Bound RPC
  message sizes and outstanding tool calls; reject malformed messages and
  unregistered capabilities. No resource-exhaustion audit has been run here.
- [x] Bound received frames to 16 Mi characters, buffer at most four protocol
  messages, validate tool requests before dispatch, and cap outstanding host
  tool calls at 32. Observe completed futures and propagate cancellation without
  blocking on a full response queue. These bounds do not provide guest disk or
  aggregate resource quotas. A SIGSTOP/full-input-pipe regression now verifies
  that blocked command writes respect the execution deadline too.
- [ ] Test startup failure races. MXC's own 1-second timer orphaned bwrap during
  a separate slow-start probe with thousands of path masks. This prototype sets
  that timer to zero and uses DSPy's startup/execution deadlines and process-group
  cleanup. Normal timeout cleanup passes; the workaround is not a general proof.
- [ ] Review the threat model independently. Process isolation shares the host
  kernel; do not present it as VM isolation or a multi-tenant service boundary.

### 3. Deliver installation and platform support

- [ ] Arrange an upstream or separately maintained, version-pinned Python runtime
  distribution for MXC and its platform dependencies. Prefer documented native
  entry points; do not depend on Node wrappers or fork MXC's policy engine.
- [ ] Build and test wheels, provenance/checksums, licenses, supported ABI/CPU
  combinations, dependency updates and security-response ownership. A smoke test
  with the actual `bubblewrap-bin==0.11.0` executable worked, but its provenance
  and companion libcap packaging still require review. **A complete pip-only
  distribution has not been built or verified.**
- [ ] Prove fresh-environment `pip install dspy` on every advertised platform
  before making this a required/default runtime. An opt-in extra can precede it.
  No silent runtime download or unsandboxed fallback on unsupported hosts.
- [ ] Verify Linux namespaces/AppArmor/container restrictions, then native macOS
  and Windows confinement and their documented cleanup contracts. Windows
  provisioning/elevation and host capabilities cannot be solved by putting an
  executable in a wheel.

Native macOS verification established the accepted lifecycle limitation above:
MXC v1.0.0 arm64 on macOS 27.0.1 / CPython 3.14.7 ran the same persistent worker,
host-tool roundtrip and synthetic filesystem/network isolation probes, but a
detached child survived launcher-group SIGKILL, graceful worker shutdown, and
MXC's 2-second timeout. Its heartbeat inode kept growing after unlink; the
attached-child timeout control stopped. The Microsoft-signed executor passed
codesign and notarization verification. This was a worker/launcher probe, not
verification of a macOS DSPy backend; the Linux-only guard remains until that
integration and its remaining checks are implemented. Windows remains untested.

### 4. Close compatibility gaps without an adapter redesign

- [ ] Differentially test `CodeInterpreter.execute`, errors, printed output,
  final expressions, tools and SUBMIT against the supported Deno behavior.
  Native transport requires JSON-compatible values; Pyodide has additional
  conversions/string fallbacks. Specify conversions deliberately, without pickle
  or executing guest-provided deserialization code in the host.
- [ ] Keep ordinary generated `dspy.Module` code and the existing
  `facade.py` / `_facade_shim.py` bridge. Validate host predictor behavior,
  custom types, sub-agent budgets, settings and nested backend propagation.
  Do not add a Flex translator or promise the complete DSPy API in the facade.
- [x] Normalize Flex inputs through Pydantic at the Flex boundary, instead of
  relying on Deno to serialize them. Live BFCL evaluation exposed rejection of
  `list[dspy.Tool]` by the native JSON transport; the same issue affected
  `dspy.Image`. The existing image-to-host-tool regression now covers native
  CPython as well as Deno. No interpreter-specific DSPy serializer was added.
- [ ] Support documented preinstalled native-library environments, including
  dependency failures and read-only packages. Keep `SandboxSerializable`'s
  Pyodide-specific setup on the existing `PythonInterpreter` path; it is not a
  universal native package installer. Define native data transfer independently.
- [x] Integrate the RLM image work using native Pillow/OpenCV plus explicit
  serialization across the host boundary. Test image inputs, outputs, tools,
  transforms and nested RLM; do not infer typed image support from import success.
- [x] Run three seeded Azure-backed Flex/GEPA optimizations and replay failures
  on Deno; fix the discovered typed-input regression and rerun BFCL. See above.
- [ ] Broaden quality coverage and compare end-to-end time on equivalent tasks,
  controlling inner RLM programs/LM responses when attributing backend differences.

### 5. Roll out the default explicitly

- [ ] First ship the supported backend opt-in, with dedicated real-runtime CI
  for isolation, lifecycle and compatibility. Update API docs and tutorials.
- [ ] Change the default in `code_interpreter.resolve_interpreter_factory`,
  preserving precedence: explicit per-module factory > scoped/global setting >
  default. RLM, Flex and ProgramOfThought currently use `PythonInterpreter` as
  both a literal default and an "unset" sentinel. Separate those meanings so an
  explicit Deno selection still works after the switch; preserve caller-owned
  interpreter lifetimes and test nested calls on the resolved backend.
- [ ] Keep `PythonInterpreter` available as the explicit Deno/Pyodide alternative
  during migration. Document dependency, filesystem, serialization and isolation
  differences, saved-program compatibility, rollback and deprecation policy.
- [ ] Flip only after installation, isolation, platform and real-workload gates
  pass. This draft intentionally does not commit to a release date or change
  DSPy's default today. The stacked image work changes the existing Pyodide
  package-preload path too; the MXC launcher does not use that JavaScript runner.
