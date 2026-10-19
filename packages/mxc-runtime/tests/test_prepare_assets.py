"""Exercise the build-time CLI without touching the checkout's prepared payload."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def checkout(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name in ("prepare_assets.py", "manifest.json"):
        shutil.copyfile(root / name, tmp_path / name)
    package = tmp_path / "dspy_mxc_runtime"
    for name in ("bin", "sources", "licenses"):
        (package / name).mkdir(parents=True)
        (package / name / "stale").write_bytes(b"unverified")
    (package / "payload.json").write_text('{"target": "stale"}')
    return tmp_path, package


def run(root, *args):
    return subprocess.run([sys.executable, str(root / "prepare_assets.py"), *args],
                          capture_output=True, text=True)


def test_diagnostic_removes_stale_native_payload(checkout):
    root, package = checkout
    result = run(root, "--target", "diagnostic")
    assert result.returncode == 0, result.stderr
    assert json.loads((package / "payload.json").read_text())["target"] == "diagnostic"
    assert not any((package / name).exists() for name in ("bin", "sources", "licenses"))


def test_existing_unverified_archive_is_rejected(checkout):
    root, package = checkout
    archive = root / "unverified.zip"
    archive.write_bytes(b"not the pinned release")
    result = run(root, "--target", "windows-x86_64", "--archive", str(archive))
    assert result.returncode != 0
    assert "MXC ZIP SHA256 mismatch" in result.stderr
    assert not (package / "payload.json").exists()
    assert not (package / "bin").exists()
