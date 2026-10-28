import json
from pathlib import Path

import dspy_mxc_runtime as runtime
import pytest


@pytest.fixture
def payload(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "__file__", str(tmp_path / "__init__.py"))
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime.platform, "libc_ver", lambda: ("glibc", "2.34"))
    data = {"target": "linux-x86_64", "system": "Linux", "machine": "x86_64", "executor": "lxc-exec"}
    (tmp_path / "payload.json").write_text(json.dumps(data))
    (tmp_path / "bin").mkdir()
    for name in ("lxc-exec", "bwrap"):
        p = tmp_path / "bin" / name
        p.touch()
        p.chmod(0o755)
    return tmp_path, data


def test_returns_relocated_paths(payload):
    root, _ = payload
    assert runtime.executables() == (root / "bin/lxc-exec", root / "bin/bwrap")
    assert all(isinstance(p, Path) for p in runtime.executables())


@pytest.mark.parametrize("system,machine", [(None, None), ("Darwin", "aarch64")])
def test_no_compatible_payload(payload, system, machine):
    root, data = payload
    data.update(system=system, machine=machine)
    (root / "payload.json").write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match=r"no compatible native payload.*no download or fallback"):
        runtime.executables()


@pytest.mark.parametrize("libc,version", [("glibc", "2.33"), ("musl", "1.2"), ("", "")])
def test_libc_floor(payload, monkeypatch, libc, version):
    monkeypatch.setattr(runtime.platform, "libc_ver", lambda: (libc, version))
    with pytest.raises(RuntimeError, match=r"glibc >= 2\.34"):
        runtime.executables()


def test_missing_bwrap_never_falls_back(payload):
    root, _ = payload
    (root / "bin/bwrap").unlink()
    with pytest.raises(RuntimeError, match="missing or not executable"):
        runtime.executables()


def test_missing_payload(payload):
    root, _ = payload
    (root / "payload.json").unlink()
    with pytest.raises(RuntimeError, match="no prepared native payload"):
        runtime.executables()


@pytest.mark.parametrize("system,machine", [("Windows", "AMD64"), ("Darwin", "arm64")])
def test_nonlinux_has_no_bwrap(payload, monkeypatch, system, machine):
    root, data = payload
    monkeypatch.setattr(runtime.platform, "system", lambda: system)
    monkeypatch.setattr(runtime.platform, "machine", lambda: machine)
    data.update(system=system, machine="x86_64" if system == "Windows" else "aarch64")
    (root / "payload.json").write_text(json.dumps(data))
    assert runtime.executables() == (root / "bin/lxc-exec", None)
