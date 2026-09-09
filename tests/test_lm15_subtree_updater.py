"""Exercise subtree operations using local repositories, without network access."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/update_vendored_lm15.py"


def run(cwd, *args, check=True):
    return subprocess.run(args, cwd=cwd, check=check, capture_output=True, text=True, timeout=90)


def git(cwd, *args):
    return run(cwd, "git", *args).stdout.strip()


def initialize(path):
    path.mkdir()
    git(path, "init", "-b", "main")
    git(path, "config", "user.name", "Subtree test")
    git(path, "config", "user.email", "subtree-test@example.invalid")


def commit(path):
    git(path, "add", ".")
    git(path, "commit", "-m", "Test snapshot")
    return git(path, "rev-parse", "HEAD")


def test_pinned_import_update_and_noop(tmp_path, monkeypatch):
    # Tests create commits in disposable repositories only. Disable inherited
    # signing/hooks and identity overrides from developer or CI configuration.
    for key in list(os.environ):
        if key.startswith(("GIT_", "PRE_COMMIT_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Subtree test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "subtree-test@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Subtree test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "subtree-test@example.invalid")

    source, target = tmp_path / "source", tmp_path / "target"
    initialize(source)
    initialize(target)
    (source / "lm15").mkdir()
    (source / "lm15/__init__.py").write_text('SNAPSHOT = "old"\n')
    (source / "lm15/_version.py").write_text('__version__ = "1.0.0a1"\n')
    (source / "LICENSE").write_text("MIT test license\n")
    (source / "CONTRACT_PIN").write_text("1" * 40 + "\n")
    old = commit(source)
    (source / "lm15/__init__.py").write_text('SNAPSHOT = "new"\n')
    (source / "CONTRACT_PIN").write_text("2" * 40 + "\n")
    new = commit(source)

    (target / "scripts").mkdir()
    shutil.copyfile(SCRIPT, target / "scripts/update_vendored_lm15.py")
    commit(target)

    def update(ref):
        return run(target, sys.executable, "scripts/update_vendored_lm15.py", ref, "--source", str(source))

    # Import an older pin while the source's default branch points elsewhere.
    update(old)
    assert git(target, "rev-parse", "HEAD:dspy/_vendor/lm15") == git(source, "rev-parse", f"{old}:lm15")
    assert (target / "dspy/_vendor/lm15/__init__.py").read_text() == 'SNAPSHOT = "old"\n'
    update(new)
    assert git(target, "rev-parse", "HEAD:dspy/_vendor/lm15") == git(source, "rev-parse", f"{new}:lm15")
    record = dict(line.split("=", 1) for line in (target / "dspy/_vendor/lm15-provenance.txt").read_text().splitlines())
    assert record["commit"] == new
    assert record["contract"] == "2" * 40
    assert record["version"] == "1.0.0a1"
    assert (target / "dspy/_vendor/lm15-LICENSE").read_text() == "MIT test license\n"
    before = git(target, "rev-parse", "HEAD")
    assert "already at this source commit" in update(new).stdout
    assert git(target, "rev-parse", "HEAD") == before
    assert git(target, "status", "--porcelain") == ""
