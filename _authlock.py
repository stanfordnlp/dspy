"""
lm15._authlock — cross-process credential-file locking and atomic writes.

Internal module. Two primitives with strict semantics:

- :func:`hold_file_lock` — an advisory, cross-process, exclusive lock scoped
  to a credential file's real path. Locks live in an lm15-owned directory
  (``$XDG_CACHE_HOME/lm15/locks`` by default), NOT next to the guarded file,
  because credential files such as ``~/.claude/.credentials.json`` belong to
  other tools whose directories lm15 must not populate.
- :func:`write_private_json_atomic` — write-to-temp + fsync + ``os.replace``
  so a crash mid-write can never leave a truncated or half-written
  credential file. The temp file is created with mode 0600 before any secret
  byte is written.

Stated limitations (these are trade-offs, not oversights):

- The lock is advisory and cooperative: it serializes lm15 processes against
  each other. Foreign writers (the Claude Code CLI, the Codex CLI) do not
  take this lock. Callers mitigate by re-reading the file inside the lock
  before refreshing (double-checked refresh), so a foreign refresh that
  landed while we waited is used instead of clobbered.
- ``flock`` semantics are unreliable on some network filesystems (NFS).
- On Windows the implementation falls back to ``msvcrt.locking`` byte locks;
  same advisory semantics, best-effort.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .errors import LockTimeoutError

_DEFAULT_LOCK_TIMEOUT_S = 60.0
_LOCK_POLL_INTERVAL_S = 0.05


class CredentialLockTimeout(LockTimeoutError, TimeoutError):
    """Could not acquire the credential-file lock within the deadline.

    An :class:`lm15.errors.LockTimeoutError` (ErrorCode ``lock_timeout``,
    retryable; spec/auth.md AUTH-6) and the builtin ``TimeoutError`` at
    once: inside the lm15 family for ``except LM15Error`` handlers, and a
    ``TimeoutError`` for code written before the code existed.
    """


def _lock_dir() -> Path:
    override = os.environ.get("LM15_LOCK_DIR")
    if override:
        return Path(override).expanduser()
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home).expanduser() if cache_home else Path("~/.cache").expanduser()
    return base / "lm15" / "locks"


def lock_path_for(path: Path) -> Path:
    """Deterministic lock-file path for a guarded credential file.

    Keyed by the guarded file's absolute (not resolved-through-symlink
    ``strict``) path so all lm15 processes agree on the same lock file even
    before the credential file exists.
    """
    canonical = os.path.realpath(os.path.expanduser(str(path)))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
    return _lock_dir() / f"{digest}.lock"


if os.name == "posix":
    import fcntl

    def _try_lock(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)

else:  # pragma: no cover - exercised only on Windows
    import msvcrt

    def _try_lock(fd: int) -> bool:
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(fd: int) -> None:
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass


@contextmanager
def hold_file_lock(
    path: str | os.PathLike[str],
    *,
    timeout_s: float = _DEFAULT_LOCK_TIMEOUT_S,
) -> Iterator[None]:
    """Hold the exclusive advisory lock for ``path``.

    Blocks up to ``timeout_s`` (polling, so it works on POSIX and Windows),
    then raises :class:`CredentialLockTimeout`. Not re-entrant: a process
    that already holds the lock must not re-enter; internal callers use the
    ``*_unlocked`` write variants for that reason.
    """
    lock_file = lock_path_for(Path(path))
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        deadline = time.monotonic() + timeout_s
        while not _try_lock(fd):
            if time.monotonic() >= deadline:
                raise CredentialLockTimeout(
                    f"Could not lock credential file {path} within {timeout_s:.0f}s "
                    f"(lock file: {lock_file}). Another process may be refreshing "
                    "the same credential; retry, or remove a stale lock only if "
                    "you are certain no other process holds it.",
                    path=str(path),
                    lock_path=str(lock_file),
                )
            time.sleep(_LOCK_POLL_INTERVAL_S)
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


def write_private_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Atomically replace ``path`` with ``data`` as private (0600) JSON.

    Durability order: write temp (created 0600 by ``mkstemp``) → flush →
    fsync(temp) → ``os.replace`` → fsync(parent dir, POSIX). A reader
    observes either the complete old file or the complete new file, never a
    partial write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(data, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    try:
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover - permission oddities are best-effort
        pass
    if os.name == "posix":
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:  # pragma: no cover - fsync of dir is best-effort
            pass
