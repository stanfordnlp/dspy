"""
lm15.login.store — the managed credential store (spec/auth.md AUTH-25).

One document per scope.  The document is the same JSON object the legacy
xAI login already writes (``{"xai": {"type": "oauth", ...}}``, Pi's
``auth.json`` convention) extended with one non-secret metadata block,
``_lm15``::

    {
      "xai":         {"type": "oauth", "access": "...", "refresh": "...", "expires": 1700000000000},
      "claude-code": {"type": "oauth", ...},
      "openrouter":  {"type": "api_key", "key": "..."},
      "_lm15": {
        "version": 1,
        "slots": {
          "xai": {"generation": "3", "connection_id": "cn_...", "revision": "7", ...},
          "gemini": {"generation": "2", "connection_id": null, "logged_out": true}
        }
      }
    }

Why one file and not a second one (decision stated in the 2026-09-22
implementation report): the xAI subscription entry must remain the single
copy the legacy loader, the router's ``oauth-unless-explicit`` chain and the
managed manager all read and renew.  A second file would have needed either
a token copy (two owners of a rotating refresh token — forbidden by R1) or
a plain ``LMRouter()`` that cannot see a ``connect()`` login.  Provider
entries keep their provider-specific private shape; ``_lm15`` holds only
what AUTH-19/20/25 need: identity generations, credential revisions,
renewal-in-flight markers, logout suppression markers and display metadata.

Two stores implement the same document contract:

- :class:`FileStore` — private (0600), atomic writes, cross-process advisory
  lock on the canonical path (``lm15._authlock``).  Default path is AUTH-8's
  ``$LM15_CREDENTIALS_PATH`` / ``~/.config/lm15/credentials.json``.
- :class:`MemoryStore` — process-lifetime, for tests and short-lived tools.

Both expose one compound write path, :meth:`Store.mutate`, a serialized
read-modify-write of the whole document.  Every commit the manager makes
(login, refresh, logout, replacement) is one ``mutate`` call, so credential
material and slot metadata change together or not at all.

An unreadable or unrecognised document is a typed
:class:`~lm15.errors.AuthOperationError` (``storage_unavailable`` /
``unsupported_store_version``), never an empty store and never overwritten.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable

from .._authlock import hold_file_lock, write_private_json_atomic
from ..errors import AuthOperationError

__all__ = ["FileStore", "MemoryStore", "Store", "Transaction", "META_KEY", "STORE_VERSION", "default_store_path"]

META_KEY = "_lm15"
STORE_VERSION = 1

Document = dict[str, Any]
Mutation = Callable[[Document], "Document | None"]


def default_store_path() -> Path:
    """``$LM15_CREDENTIALS_PATH``, else ``$XDG_CONFIG_HOME/lm15/credentials.json``
    (spec/auth.md AUTH-8).  Read at call time, never at import."""
    override = os.environ.get("LM15_CREDENTIALS_PATH")
    if override:
        return Path(override).expanduser()
    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home).expanduser() if config_home else Path("~/.config").expanduser()
    return base / "lm15" / "credentials.json"


def _storage_error(message: str, *, reason: str = "storage_unavailable", stage: str = "persistence") -> AuthOperationError:
    return AuthOperationError(
        message, reason=reason, stage=stage, commit_state="not_committed", recovery="repair_storage",
        operation="store",
    )


def validate_document(data: Any, *, where: str) -> Document:
    """Reject anything that is not the document shape above.  Raises the
    typed storage error; never returns a guess."""
    if not isinstance(data, dict):
        raise _storage_error(f"Credential store at {where} is not a JSON object; not touching it.")
    meta = data.get(META_KEY)
    if meta is not None:
        if not isinstance(meta, dict):
            raise _storage_error(f"Credential store at {where} has a malformed {META_KEY!r} block; not touching it.")
        version = meta.get("version")
        if version != STORE_VERSION:
            raise _storage_error(
                f"Credential store at {where} is managed-store version {version!r}; this lm15 reads version "
                f"{STORE_VERSION}. Upgrade lm15 or point LM15_CREDENTIALS_PATH at another file.",
                reason="unsupported_store_version",
            )
        slots = meta.get("slots", {})
        if not isinstance(slots, dict) or any(not isinstance(v, dict) for v in slots.values()):
            raise _storage_error(f"Credential store at {where} has malformed slot metadata; not touching it.")
    for key, value in data.items():
        if key != META_KEY and not isinstance(value, dict):
            raise _storage_error(f"Credential store at {where}: entry {key!r} is not an object; not touching it.")
    return data


class Transaction:
    """The store, locked.  ``read`` gives a private copy of the document as
    it is now; ``write`` replaces it atomically.  A transaction may write
    more than once (AUTH-20.4: a renewal marker before the exchange, the
    result after it) — every write is durable on return."""

    def read(self) -> Document:
        raise NotImplementedError

    def write(self, document: Document) -> None:
        raise NotImplementedError


class Store:
    """The document contract both stores implement.  Subclass, do not use."""

    description: str = "store"

    def read(self) -> Document:
        """A private copy of the whole document.  No lock; for status and
        selection.  Correctness-critical reads happen inside a transaction."""
        raise NotImplementedError

    def transaction(self):
        """``with store.transaction() as txn:`` — exclusive, cross-process
        where the backend supports it (the file store's canonical-path lock)."""
        raise NotImplementedError

    def mutate(self, fn: Mutation) -> Document:
        """Serialized read-modify-write.  ``fn`` gets a private copy of the
        current document and returns the new one, or ``None`` to leave the
        store untouched.  Returns the post-write document (a copy)."""
        with self.transaction() as txn:
            current = txn.read()
            replacement = fn(copy.deepcopy(current))
            if replacement is None:
                return current
            txn.write(replacement)
            return copy.deepcopy(replacement)

    def reserve(self) -> None:
        """AUTH-17: prove the store can be written before any external
        authorization starts.  Raises the typed storage error; never
        substitutes another store."""
        raise NotImplementedError

    def __repr__(self) -> str:  # never contents
        return f"{type(self).__name__}({self.description})"


class MemoryStore(Store):
    """Process-lifetime document; ``Auth.memory()``."""

    def __init__(self) -> None:
        self._data: Document = {}
        self._lock = threading.RLock()
        self.description = "memory"

    def read(self) -> Document:
        with self._lock:
            return copy.deepcopy(self._data)

    def transaction(self):
        store = self

        class _Txn(Transaction):
            def __enter__(self):
                store._lock.acquire()
                return self

            def __exit__(self, *_exc):
                store._lock.release()

            def read(self) -> Document:
                return copy.deepcopy(store._data)

            def write(self, document: Document) -> None:
                store._data = validate_document(copy.deepcopy(document), where="memory")

        return _Txn()

    def reserve(self) -> None:
        return None


class FileStore(Store):
    """AUTH-8 private file; AUTH-4 locking and atomic writes."""

    def __init__(self, path: str | os.PathLike[str] | None = None, *, lock_timeout_s: float = 30.0) -> None:
        # Anchor the absolute path now (AUTH-14): a later chdir must not move
        # the store.  Nothing is read or created here.
        chosen = Path(path).expanduser() if path is not None else default_store_path()
        self.path = chosen if chosen.is_absolute() else Path(os.getcwd()) / chosen
        self.lock_timeout_s = lock_timeout_s
        self.description = str(self.path)

    def _load(self) -> Document:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise _storage_error(f"Could not read credential store at {self.path}: {exc.strerror or exc}") from exc
        try:
            data = json.loads(text, object_pairs_hook=_reject_duplicate_members)
        except ValueError as exc:
            raise _storage_error(f"Credential store at {self.path} is not valid JSON; not touching it.") from exc
        return validate_document(data, where=str(self.path))

    def read(self) -> Document:
        return self._load()

    def transaction(self):
        store = self

        class _Txn(Transaction):
            def __enter__(self):
                self._guard = hold_file_lock(store.path, timeout_s=store.lock_timeout_s)
                self._guard.__enter__()
                return self

            def __exit__(self, *exc):
                return self._guard.__exit__(*exc)

            def read(self) -> Document:
                return store._load()

            def write(self, document: Document) -> None:
                validate_document(document, where=str(store.path))
                try:
                    write_private_json_atomic(store.path, document)
                except OSError as exc:
                    raise _storage_error(
                        f"Could not write credential store at {store.path}: {exc.strerror or exc}"
                    ) from exc

        return _Txn()

    def reserve(self) -> None:
        """Take the lock once and touch nothing: proves the directory and the
        lock directory are writable before a browser is opened."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise _storage_error(
                f"Cannot create {self.path.parent} for the credential store: {exc.strerror or exc}",
                stage="reservation",
            ) from exc
        if self.path.exists() and not os.access(self.path, os.W_OK):
            raise _storage_error(f"Credential store at {self.path} is not writable.", stage="reservation")
        with hold_file_lock(self.path, timeout_s=self.lock_timeout_s):
            self._load()


def _reject_duplicate_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """AUTH-25: a duplicate member name is an ambiguity a post-parse schema
    cannot see; refuse before collapsing into a dict."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate member {key!r}")
        result[key] = value
    return result
