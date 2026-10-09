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

import errno
import errno
import hashlib
import json
import os
import stat
import tempfile
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .errors import LockTimeoutError, NotConfiguredError

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
    if cache_home:
        return Path(cache_home).expanduser() / "lm15" / "locks"
    try:
        base = Path("~/.cache").expanduser()
    except RuntimeError as exc:  # no home directory: a WASI guest, some containers
        raise _no_home_error() from exc
    return base / "lm15" / "locks"


def _no_home_error() -> NotConfiguredError:
    return NotConfiguredError(
        "No home directory to keep the credential lock in, so lm15 cannot "
        "serialize credential refreshes against other processes. Reading "
        "credentials still works; refreshing them from here does not.",
        credential_hint="Set LM15_LOCK_DIR (or XDG_CACHE_HOME), or pass an explicit credential instead of a stored login",
    )


def _strip_windows_verbatim(path: str) -> str:
    path = path.replace("/", "\\")
    if path[:8].lower() == "\\\\?\\unc\\":
        return "\\\\" + path[8:]
    if path.startswith("\\\\?\\"):
        rest = path[4:]
        if len(rest) < 3 or not rest[0].isascii() or not rest[0].isalpha() or rest[1:3] != ":\\":
            raise ValueError("Unsupported Windows credential path namespace")
        return rest
    return path


def _windows_identity_key(canonical: str) -> str:
    """Pure key spelling, also testable on POSIX; no lexical path resolution.

    Windows migration: old and new SDK processes MUST NOT overlap. Removing
    verbatim prefixes and lowercasing changes previously published lock names.
    Like normcase, lowercase deliberately overlocks case-sensitive directories.
    Use Unicode string lowercase (not casefold or locale-sensitive lowercase)
    in all SDKs, including Rust's str::to_lowercase.
    """
    return _strip_windows_verbatim(canonical).lower()


def _real_path_allow_missing(target: str) -> str:
    """Resolve components in filesystem order; only ENOENT is recoverable.

    Unlike abspath/normpath, never collapse symlink/.. before reading the
    link. A missing component does not stop the walk: later .. can return
    to an existing ancestor. Forty link expansions bound loops in every SDK.
    """
    windows = os.name == "nt"

    def parts(value: str, base: str) -> tuple[str, deque[str]]:
        if "\0" in value:
            raise ValueError("NUL in credential path")
        # Refuse unpaired surrogates rather than hash a lossy encoding.
        value.encode("utf-8")
        if windows:
            value = _strip_windows_verbatim(value)
            drive, tail = os.path.splitdrive(value)
            unc = drive.startswith("\\\\")
            if value.startswith("\\\\.\\") or (drive and not unc and not tail.startswith("\\")):
                raise ValueError("Unsupported Windows credential path namespace or drive-relative path")
            if unc:
                roots = drive[2:].split("\\")
                if len(roots) != 2 or any(not p or not p.isascii() or p in (".", "..") or "?" in p or ":" in p or p.endswith((".", " ")) for p in roots):
                    raise ValueError("Unsupported Windows credential path root")
            elif drive and not (len(drive) == 2 and drive[1] == ":" and drive[0].isascii() and drive[0].isalpha()):
                raise ValueError("Unsupported Windows credential path root")
            if tail.startswith("\\") or unc:
                base = (drive or os.path.splitdrive(base)[0]) + "\\"
            base = _strip_windows_verbatim(os.path.realpath(base, strict=True))
            names = deque(p for p in tail.split("\\") if p)
            for name in names:
                if name in (".", ".."):
                    continue
                stem = name.split(".", 1)[0].upper()
                if (name.endswith((".", " ")) or ":" in name
                        or stem in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
                        or (len(stem) == 4 and stem[:3] in {"COM", "LPT"}
                            and stem[3] in "123456789¹²³")):
                    raise ValueError("Unsupported Windows credential path component")
        else:
            if value.startswith("/"):
                base = "/"
            names = deque(p for p in value.split("/") if p)
        return base, names

    if target.startswith("~") and target != "~" and not target.startswith(("~/", "~\\") if windows else ("~/",)):
        raise ValueError("Named-user home expansion is unsupported for credential locks")
    target = os.path.expanduser(target)
    if target == "~" or target.startswith("~/") or (windows and target.startswith("~\\")):
        # No home directory (Windows has no passwd fallback, so this is where
        # it shows there): the same fact about the host as _lock_dir's, and
        # the same typed error with the same remedy.
        raise _no_home_error()
    cwd = os.path.realpath(os.getcwd(), strict=True)
    resolved, pending = parts(target, _strip_windows_verbatim(cwd) if windows else cwd)
    links = 0
    while pending:
        name = pending.popleft()
        if name == ".":
            continue
        if name == "..":
            resolved = os.path.dirname(resolved)
            continue
        candidate = os.path.join(resolved, name)
        try:
            info = os.lstat(candidate)
        except FileNotFoundError:
            # Windows compares with filesystem upcase tables, not Unicode
            # lowercase (e.g. sigma/final sigma). With no on-disk spelling
            # to canonicalize, refuse non-ASCII rather than risk underlocking.
            if windows and not name.isascii():
                raise ValueError("Missing non-ASCII Windows credential path component is unsupported") from None
            resolved = candidate
            continue
        if stat.S_ISLNK(info.st_mode) or (windows and getattr(info, "st_reparse_tag", 0) == 0xA0000003):
            links += 1
            if links > 40:
                raise OSError(errno.ELOOP, "Too many credential path symlinks", target)
            resolved, linked = parts(os.readlink(candidate), resolved)
            linked.extend(pending)
            pending = linked
        else:
            if pending and not stat.S_ISDIR(info.st_mode):
                raise NotADirectoryError(errno.ENOTDIR, "Credential path ancestor is not a directory", candidate)
            resolved = _strip_windows_verbatim(os.path.realpath(candidate, strict=True)) if windows else candidate
    return resolved


def lock_path_for(path: str | os.PathLike[str]) -> Path:
    """AUTH-4 lock identity, including missing leaves and dangling symlinks.

    Ordinary POSIX realpath hashes are unchanged. Resolution errors fail
    closed; a guessed identity must never allow concurrent credential refresh.
    Windows old/new processes must not overlap (see _windows_identity_key).
    """
    canonical = _real_path_allow_missing(os.fspath(path))
    if os.name == "nt":
        canonical = _windows_identity_key(canonical)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
    return _lock_dir() / f"{digest}.lock"


# Chosen by which primitive the platform actually has, not by `os.name`: WASI reports
# "posix" and ships no fcntl, so a name test picks an implementation that cannot import.
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows, and POSIX-ish builds without fcntl
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - every non-Windows platform
    msvcrt = None


if fcntl is not None:

    def _try_lock(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                return False
            raise NotConfiguredError("The filesystem could not acquire a credential lock; use a local locking-capable filesystem or an explicit credential") from exc

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)

elif msvcrt is not None:  # pragma: no cover - exercised only on Windows

    def _try_lock(fd: int) -> bool:
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                return False
            raise NotConfiguredError("The filesystem could not acquire a credential lock; use a local locking-capable filesystem or an explicit credential") from exc

    def _unlock(fd: int) -> None:
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

else:  # pragma: no cover - platforms with neither primitive, such as WASI

    # NotConfiguredError, as in lm15-ts (`stores.ts`): a fact about this host's setup,
    # not about a provider (CapabilityError) and not a timeout (LockTimeoutError).
    def _try_lock(fd: int) -> bool:
        raise NotConfiguredError(
            "This platform provides no advisory file locking (neither fcntl nor msvcrt), "
            "so lm15 cannot serialize credential refreshes against other processes. "
            "Reading credentials still works; refreshing them from here does not.",
            credential_hint="Pass an explicit credential instead of a stored login",
        )

    def _unlock(fd: int) -> None:
        return None


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
    lock_file = lock_path_for(path)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        deadline = time.monotonic() + timeout_s
        while not _try_lock(fd):
            if time.monotonic() >= deadline:
                raise CredentialLockTimeout(
                    f"Could not lock credential file {path} within {timeout_s:.0f}s "
                    f"(lock file: {lock_file}). Another process may be refreshing "
                    "the same credential; retry after it finishes or stop the holder. "
                    "Never delete a lock file while processes may be using it.",
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
