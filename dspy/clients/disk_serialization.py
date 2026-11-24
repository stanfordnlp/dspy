"""Restricted pickle deserialization for disk cache.

Provides a RestrictedDisk subclass that restricts both keys and values,
and confines file-backed entries to the cache directory on POSIX. Windows
path checks are best effort against concurrent directory replacement.

Trust model:
- openai.types.* classes are trusted by their actual module and qualified name
- user safe_types are trusted by exact (module, qualname) pairs
- arrays use non-pickle .npy serialization; legacy pickled arrays are misses
"""

from __future__ import annotations

import contextlib
import io
import os
import pickle
import re
import sys
from typing import Any

from diskcache import Disk
from diskcache.core import MODE_BINARY, MODE_PICKLE, MODE_TEXT, UNKNOWN

_TRUSTED_MODULE_PREFIXES = ("openai.types.",)

_NPY_MAGIC = b"\x93NUMPY"
_HEX = re.compile(r"[0-9a-f]+\Z")


class DeserializationError(Exception):
    """Raised when a cached value cannot be deserialized."""


def _array_bytes(value):
    numpy = sys.modules["numpy"]
    if type(value) is not numpy.ndarray:
        raise TypeError("NumPy array subclasses cannot be safely serialized in the disk cache")
    buffer = io.BytesIO()
    numpy.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


class _ArrayPickler(pickle.Pickler):
    def persistent_id(self, obj):
        numpy = sys.modules.get("numpy")
        if numpy is not None and isinstance(obj, numpy.ndarray):
            return _NPY_MAGIC, _array_bytes(obj)
        return None


class _RestrictedUnpickler(pickle.Unpickler):
    _allowed: frozenset[tuple[str, str]] = frozenset()

    def persistent_load(self, pid):
        if (
            type(pid) is tuple
            and len(pid) == 2
            and type(pid[0]) is bytes
            and pid[0] == _NPY_MAGIC
            and type(pid[1]) is bytes
            and pid[1].startswith(_NPY_MAGIC)
        ):
            import numpy as np

            result = np.load(io.BytesIO(pid[1]), allow_pickle=False)
            if type(result) is np.ndarray:
                return result
        raise DeserializationError("Invalid cache array reference")

    def find_class(self, module: str, name: str) -> type:
        if any(module.startswith(p) for p in _TRUSTED_MODULE_PREFIXES) or (module, name) in self._allowed:
            resolved = super().find_class(module, name)
            if isinstance(resolved, type) and (resolved.__module__, resolved.__qualname__) == (module, name):
                return resolved
        raise DeserializationError(
            f"Type {module}.{name} is not in the safe_types allowlist. "
            f"Register it via dspy.configure_cache(safe_types=[...])."
        )


def _restricted_load(f: Any, allowed: frozenset[tuple[str, str]]) -> Any:
    unpickler = _RestrictedUnpickler(f)
    unpickler._allowed = allowed
    try:
        return unpickler.load()
    except DeserializationError:
        raise
    except Exception as e:
        raise DeserializationError(f"Corrupt cache entry: {e}") from e


class _RestrictedDisk(Disk):
    """Disk subclass with restricted keys, values, and file-backed entries."""

    _allowed: frozenset[tuple[str, str]]

    def get(self, key, raw):
        if not raw:
            return _restricted_load(io.BytesIO(key), self._allowed)
        return super().get(key, raw)

    def store(self, value, read, key=UNKNOWN):
        numpy = sys.modules.get("numpy")
        if not read and numpy is not None and isinstance(value, numpy.ndarray):
            size, _, filename, data = super().store(_array_bytes(value), False, key=key)
            return size, MODE_PICKLE, filename, data
        if not read and type(value) not in (bytes, str, int, float):
            buffer = io.BytesIO()
            _ArrayPickler(buffer, protocol=self.pickle_protocol).dump(value)
            size, _, filename, data = super().store(buffer.getvalue(), False, key=key)
            return size, MODE_PICKLE, filename, data
        return super().store(value, read, key=key)

    @staticmethod
    def _file_parts(filename):
        if not isinstance(filename, str):
            raise DeserializationError("Invalid cache filename")
        parts = filename.split(os.sep)
        if (
            len(parts) != 3
            or len(parts[0]) != 2
            or len(parts[1]) != 2
            or len(parts[2]) != 32
            or not parts[2].endswith(".val")
            or any(not _HEX.fullmatch(part) for part in (*parts[:2], parts[2][:-4]))
        ):
            raise DeserializationError("Invalid cache filename")
        return parts

    @contextlib.contextmanager
    def _parent_dir(self, filename, create=False):
        first, second, leaf = self._file_parts(filename)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        with contextlib.ExitStack() as stack:
            root_fd = os.open(self._directory, os.O_RDONLY | os.O_DIRECTORY)
            stack.callback(os.close, root_fd)
            if create:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(first, dir_fd=root_fd)
            first_fd = os.open(first, flags, dir_fd=root_fd)
            stack.callback(os.close, first_fd)
            if create:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(second, dir_fd=first_fd)
            parent_fd = os.open(second, flags, dir_fd=first_fd)
            stack.callback(os.close, parent_fd)
            yield root_fd, first_fd, parent_fd, first, second, leaf

    def _windows_path(self, filename):
        # Windows lacks the dir_fd/no-follow operations used on POSIX. A
        # concurrent junction or symlink swap after this check remains possible.
        parts = self._file_parts(filename)
        root = os.path.realpath(self._directory)
        path = os.path.realpath(os.path.join(root, *parts))
        if os.path.commonpath((root, path)) != root:
            raise DeserializationError("Cache filename escapes cache directory")
        return path

    def _open_file(self, filename):
        if os.name == "nt":
            return open(self._windows_path(filename), "rb")
        with self._parent_dir(filename) as (_, _, parent_fd, _, _, leaf):
            return os.fdopen(os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd), "rb")

    def _write(self, full_path, iterator, mode, encoding=None):
        filename = os.path.relpath(full_path, self._directory)
        if os.name == "nt":
            return super()._write(self._windows_path(filename), iterator, mode, encoding)
        for attempt in range(10):
            try:
                with self._parent_dir(filename, create=True) as (_, _, parent_fd, _, _, leaf):
                    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                    fd = os.open(leaf, flags, 0o666, dir_fd=parent_fd)
            except OSError:
                if attempt == 9:
                    raise
                continue
            with os.fdopen(fd, mode, encoding=encoding) as writer:
                size = 0
                for chunk in iterator:
                    size += len(chunk)
                    writer.write(chunk)
                return size

    def fetch(self, mode, filename, value, read):
        if mode in (MODE_BINARY, MODE_TEXT, MODE_PICKLE):
            try:
                if filename is not None:
                    f = self._open_file(filename)
                    if mode == MODE_BINARY and read:
                        return f
                elif mode == MODE_PICKLE:
                    f = io.BytesIO(value)
                else:
                    return super().fetch(mode, filename, value, read)
                with f:
                    if mode == MODE_BINARY:
                        return f.read()
                    if mode == MODE_TEXT:
                        with io.TextIOWrapper(f, encoding="utf-8") as reader:
                            return reader.read()
                    if f.read(len(_NPY_MAGIC)) == _NPY_MAGIC:
                        import numpy as np

                        f.seek(0)
                        return np.load(f, allow_pickle=False)
                    f.seek(0)
                    return _restricted_load(f, self._allowed)
            except (OSError, ValueError, UnicodeError) as e:
                raise DeserializationError("Invalid cache entry") from e
        return super().fetch(mode, filename, value, read)

    def remove(self, filename):
        try:
            if os.name == "nt":
                self._windows_path(filename)
                return super().remove(filename)
            with self._parent_dir(filename) as (root_fd, first_fd, parent_fd, first, second, leaf):
                with contextlib.suppress(OSError):
                    os.unlink(leaf, dir_fd=parent_fd)
                with contextlib.suppress(OSError):
                    os.rmdir(second, dir_fd=first_fd)
                with contextlib.suppress(OSError):
                    os.rmdir(first, dir_fd=root_fd)
        except (OSError, ValueError, DeserializationError):
            pass


def restricted_disk(allowed: frozenset[tuple[str, str]]) -> type[_RestrictedDisk]:
    """Return a Disk subclass bound to the given allowlist.

    diskcache expects ``disk=`` to be a class it instantiates itself, so
    we return a subclass with ``_allowed`` baked in as a class attribute.
    """
    return type("RestrictedDisk", (_RestrictedDisk,), {"_allowed": allowed})
