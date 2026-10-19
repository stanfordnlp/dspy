# dspy-mxc-runtime 0.1.0a1 (unpublished preparation)

A small, DSPy-independent companion: wheels own pinned Microsoft MXC v1.0.1
executors, with static bubblewrap 0.13.1 on Linux. No dependencies on DSPy,
runtime installer, policy engine, updater, PATH search, or fallback. This does
not select DSPy's default interpreter or implement its security policy.

```python
from dspy_mxc_runtime import executables
executor, bwrap = executables()  # tuple[pathlib.Path, pathlib.Path | None]
```

Paths are absolute and relative to the installed package, not its original
build directory. `bwrap` is only present on Linux. Diagnostic wheels, missing
files, wrong OS/architecture, and unsupported Linux libc raise `RuntimeError`.
The function does not start executables, contact the network, or change host
policy. It does not certify backend availability, kernel permissions, sandbox
isolation, quotas, macOS entitlements, Windows services, or installed OS DLLs.
Unprivileged namespaces must already be permitted for Linux bubblewrap use.
Native MXC itself exposes optional provisioning/download commands; this package
never invokes them. Callers must not treat arbitrary MXC options as offline-safe.

## Payload inventory and compatibility

The 516 MiB official ZIP was downloaded and its SHA256 checked against
`manifest.json`; all five executor member hashes are recorded there. It has
118 entries, including symbols, proxies, DLLs and SPDX manifests, but no license
text. Only the chosen executor and its upstream SPDX manifest are copied;
MIT `LICENSE.md` from tag v1.0.1 is separately hash-checked. No binary rewriting
is performed, including on signed Mac/Windows executables.

| Target | Wheel platform (all `py3-none-…`) | Inspection / status |
| --- | --- | --- |
| Linux x64 | `manylinux_2_34_x86_64` | Locally built and CLI-smoke-tested on glibc 2.36 |
| Linux arm64 | `manylinux_2_34_aarch64` | Executor inspected; build blocked until a static bwrap bundle/hash is qualified |
| Windows x64 | `win_amd64` | Installation/relocation pass; exact sandbox startup fails closed on Windows Server 2025 CI |
| Windows arm64 | `win_arm64` | Bytes/hash inspected; imports and native execution still require Windows qualification |
| macOS arm64 | `macosx_11_0_arm64` | macOS 14 CI and macOS 27 installed-wheel checks passed; older OS baselines unqualified |
| Diagnostic | `any` | No executors; explicit failure on runtime use |

Linux executors both require **glibc >= 2.34** (highest versioned GLIBC symbol),
`libgcc_s.so.1`, `libc.so.6` and their architecture's dynamic loader. No musllinux
wheel is advertised. Static bwrap does not remove the executor's libc floor.
These tags express observed loader/ABI requirements, not a completed manylinux
certification or validation on the oldest supported distribution.

Mac load commands specify macOS 11.0 minimum, SDK 15.5, and only system
libSystem, libsandbox, CoreFoundation and libiconv dependencies. Native Mac
verification passed strict codesign, an Apple-anchored Microsoft Team UBF8T346G9
requirement, and notarization verification on the exact bundled executor. This
does not establish the minimum usable OS or qualification on every supported OS.
No Intel Mac executor exists. Windows x64 imports system/API-set DLLs including
kernel32, bcryptprimitives, advapi32, ntdll, ws2_32, ole32, shell32, combase,
userenv, crypt32, wintrust, tdh, bcrypt, oleaut32 and WinRT API sets. Its PE
security directory is present. Signature trust, minimum usable Windows build,
and auxiliary backend/service needs remain native-platform release gates.
Packaging a sole executor does not promise every upstream backend is usable.

## Local builds (no publication)

From this directory, with Python >=3.10 and `uv` installed:

```sh
# Can download only here, explicitly at asset-preparation time.
# --archive and --mxc-license allow fully offline asset preparation.
python tools/build_bundle.py --engine docker --output assets/runtime-build-source.tar.gz
# On Linux with working unprivileged user/mount namespaces and bwrap instead:
# python tools/build_bundle.py --engine namespace --output assets/runtime-build-source.tar.gz
python prepare_assets.py --target linux-x86_64 \
  --archive /path/to/mxc-release-binaries.zip \
  --mxc-license /path/to/LICENSE.md \
  --bwrap-bundle assets/runtime-build-source.tar.gz
uv build --wheel --out-dir dist
python tools/check_wheel.py dist/dspy_mxc_runtime-0.1.0a1-py3-none-manylinux_2_34_x86_64.whl

# Removes stale binaries and build output before making an executor-free wheel.
python prepare_assets.py --target diagnostic
uv build --wheel --out-dir dist
python tools/check_wheel.py dist/dspy_mxc_runtime-0.1.0a1-py3-none-any.whl

# Unit tests (pytest is a development dependency only)
PYTHONPATH=. ../../.venv/bin/python -m pytest -q tests
```

Use the other target names in `manifest.json` for cross-packaging, then test on
that native OS. Builds never fetch assets implicitly: unprepared wheel builds
fail. The explicit preparation command may download the ZIP and license; Python
build isolation may separately fetch build tools. Prepared source distributions
can be built offline given those tools. Clean builds/relocation checks need no
host privilege changes. Do not prepare two targets concurrently in one checkout.
All binary assets, generated notices/sources, wheels and build directories are
ignored in Git. Unprepared source checkouts intentionally have no payload.

### Static bubblewrap and corresponding source

`tools/build_bundle.py` replaces the private qualification-bundle prerequisite.
It downloads a SHA256-pinned Alpine 3.24.2 x86_64 minirootfs, bubblewrap and libcap
sources pinned in `manifest.json`, and musl source for its copyright notice.
Docker imports that checked rootfs (no mutable image tag); the namespace engine
uses a fresh root with the existing host bwrap. Both execute the same
`tools/build-bwrap.sh` in `/build`, never against the consumer host. No daemon
startup, host policy changes, setuid installation or runtime downloads occur.

The script builds bubblewrap from source with Meson/Ninja and `-static`, disables
SELinux, and rejects ELF interpreter/NEEDED entries. `tools/apk-packages.txt` pins
the direct build dependencies, including GCC 15.2.0-r5, binutils 2.45.1-r1,
musl-dev 1.2.6-r2 and libcap-static 2.78-r0. Libcap is the Alpine static library,
not locally compiled; its matching upstream 2.78 source and notice accompany it.
The **entire** installed package inventory, notices, source archives and resulting
bwrap must match the existing qualification hashes in `manifest.json`, including
`df25afa3338d316da86ec134544ca0b61882a719dab67b152d56590d9208b961`.
This exact binary was reproduced using the namespace engine locally and the
Docker engine in native Linux CI. The bundle includes notices, corresponding
bubblewrap/libcap source and the inventory; asset preparation embeds these plus
the tracked build recipes, dependency pins and manifest. It only reads named,
hash-checked members, never arbitrary archive extraction or unverified assets.

Alpine repositories are not snapshot-archived here: retired APK versions or a
changed transitive inventory deliberately fail closed. This is a checked recipe,
not a guarantee that mutable mirrors retain its inputs indefinitely. MXC remains
a pinned upstream binary, not reproduced from source. Linux arm64 is still
blocked pending native build/inspection/qualification and a reviewed hash.

The workflow builds the companion on Ubuntu 22.04, macOS ARM64 and Windows x64,
installs it explicitly via local `--find-links`, and runs the installed DSPy
`LocalInterpreter` primitive/native suites outside the checkout. A mandatory
backend smoke test fails on unavailable backends. Windows never provisions,
elevates or modifies ACLs: a red job is useful qualification evidence, **not** a
Windows support claim. Native Linux CI passed 54 interpreter/integration tests;
macOS 14 ARM64 passed 52 (four Linux-specific cases skipped). A separate macOS
27 ARM64 runner passed 37 interpreter tests, including inherited-pipe
deadline/teardown regressions. Windows Server 2025 passed package installation,
relocation and 12 package tests, but rejected sandbox startup because no permitted
non-DACL backend could service the request.

MXC's pinned [OS support table](https://github.com/microsoft/mxc/blob/v1.0.1/docs/backends/process-container/os-version-support.md)
lists serviced Windows 11 client builds (including 24H2 build 26100.9278 and
25H2 build 26200.9278). This is an upstream claim, not a successful DSPy test on
those clients. The shipping build excludes the BFS fallback. Do not enable
registry feature overrides or run host-preparation tools to hide this gate;
qualify the exact unchanged policy on a supported, unelevated Windows 11 host.

### Redistribution obligations and remaining gates

The Python wrapper is MIT; this does not relicense bundled components. MXC's
MIT notice is included. Bubblewrap's LGPL-2.0-or-later COPYING, complete upstream
source tarball and build instructions are included to support modification and
rebuilding. Preserve these and provide corresponding source with any binary
redistribution, including modifications/build scripts if changed. The static
libcap component offers BSD-3-Clause OR GPL-2.0-only; the BSD alternative and
musl's permissive notices must be preserved. Source for libcap is included too.
These are recorded obligations, **not legal-clearance claims**. The upstream MXC
SPDX inventory declares `NOASSERTION` for all 548 packages; its presence is not
complete third-party license clearance. Review transitive notices/source
obligations before publication, and review version/security maintenance ownership.

Before a release/default switch: qualify native Linux arm64, Windows and Mac
backends; verify platform signatures/trust and auxiliary files; validate the
manylinux floor on real baseline hosts; reproduce/audit bwrap and its toolchain;
finish license/security review; test pip selection against an index containing
both native and universal wheels; and test DSPy's separate resolver/policy end
to end. Nothing here authorizes publishing or claims a merge-ready default.
