"""Locate wheel-owned executables. No downloads, host changes, or fallbacks."""
import json
import os
import platform
from pathlib import Path

__version__ = "0.1.0a1"


def executables() -> tuple[Path, Path | None]:
    """Return (MXC executor, Linux bubblewrap), or raise RuntimeError."""
    root = Path(__file__).resolve().parent
    try:
        payload = json.loads((root / "payload.json").read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("MXC runtime has no prepared native payload; reinstall a compatible wheel") from exc
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    if (payload["system"], payload["machine"]) != (platform.system(), machine):
        raise RuntimeError(
            f"MXC runtime has no compatible native payload for {platform.system()}/{machine} "
            f"(installed target: {payload['target']}); no download or fallback is available"
        )
    if platform.system() == "Linux":
        libc, version = platform.libc_ver()
        try:
            compatible = libc == "glibc" and tuple(map(int, version.split(".")[:2])) >= (2, 34)
        except ValueError:
            compatible = False
        if not compatible:
            raise RuntimeError("MXC v1.0.1 requires Linux glibc >= 2.34; musl is not supported")
    executor = root / "bin" / payload["executor"]
    bwrap = root / "bin/bwrap" if platform.system() == "Linux" else None
    for path in (executor, bwrap):
        if path is not None and (not path.is_file() or (os.name != "nt" and not os.access(path, os.X_OK))):
            raise RuntimeError(f"MXC runtime payload is missing or not executable: {path}")
    return executor, bwrap
