import dataclasses
import importlib
import logging
import os
import pickle
import sqlite3
import threading
import time
from typing import Any

import numpy as np
import orjson
import pydantic

logger = logging.getLogger(__name__)

# diskcache mode constants
_DC_MODE_RAW = 1
_DC_MODE_BINARY = 2
_DC_MODE_TEXT = 3
_DC_MODE_PICKLE = 4
_ENCODED_TYPE_KEY = "__dspy_cache_type__"
_ENCODED_MODULE_KEY = "__dspy_cache_module__"
_ENCODED_QUALNAME_KEY = "__dspy_cache_qualname__"
_ENCODED_DATA_KEY = "__dspy_cache_data__"
_ENCODED_DTYPE_KEY = "__dspy_cache_dtype__"
_PYDANTIC_TYPE = "pydantic"
_DATACLASS_TYPE = "dataclass"
_NDARRAY_TYPE = "ndarray"

def _serialize(value: Any) -> bytes:
    return orjson.dumps({"_data": _encode_value(value)})


def _encode_value(value: Any) -> Any:
    if isinstance(value, pydantic.BaseModel):
        return {
            _ENCODED_TYPE_KEY: _PYDANTIC_TYPE,
            _ENCODED_MODULE_KEY: type(value).__module__,
            _ENCODED_QUALNAME_KEY: type(value).__qualname__,
            _ENCODED_DATA_KEY: {k: _encode_value(v) for k, v in _iter_pydantic_items(value).items()},
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            _ENCODED_TYPE_KEY: _DATACLASS_TYPE,
            _ENCODED_MODULE_KEY: type(value).__module__,
            _ENCODED_QUALNAME_KEY: type(value).__qualname__,
            _ENCODED_DATA_KEY: {
                field.name: _encode_value(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, np.ndarray):
        return {
            _ENCODED_TYPE_KEY: _NDARRAY_TYPE,
            _ENCODED_DTYPE_KEY: str(value.dtype),
            _ENCODED_DATA_KEY: value.tolist(),
        }
    if isinstance(value, dict):
        return {k: _encode_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_value(item) for item in value]
    return value


def _resolve_class(module_name: str, qualname: str) -> type:
    module = importlib.import_module(module_name)
    obj = module
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


def _deserialize(blob: bytes) -> Any:
    envelope = orjson.loads(blob)
    return _decode_value(envelope["_data"])


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        encoded_type = value.get(_ENCODED_TYPE_KEY)
        if encoded_type == _PYDANTIC_TYPE:
            cls = _resolve_class(value[_ENCODED_MODULE_KEY], value[_ENCODED_QUALNAME_KEY])
            return cls(**_decode_value(value[_ENCODED_DATA_KEY]))
        if encoded_type == _DATACLASS_TYPE:
            cls = _resolve_class(value[_ENCODED_MODULE_KEY], value[_ENCODED_QUALNAME_KEY])
            return cls(**_decode_value(value[_ENCODED_DATA_KEY]))
        if encoded_type == _NDARRAY_TYPE:
            return np.asarray(_decode_value(value[_ENCODED_DATA_KEY]), dtype=np.dtype(value[_ENCODED_DTYPE_KEY]))
        return {k: _decode_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def _iter_pydantic_items(value: pydantic.BaseModel) -> dict[str, Any]:
    data = {key: item for key, item in value.__dict__.items() if not key.startswith("_")}
    extra = getattr(value, "__pydantic_extra__", None)
    if extra:
        for key, item in extra.items():
            data.setdefault(key, item)
    return data


class SQLiteCache:
    """Dict-like SQLite-backed cache using orjson serialization.

    Uses WAL mode for concurrent read performance.
    Enforces a size limit via LRU eviction.
    """

    def __init__(self, directory: str, size_limit: int | None = None):
        os.makedirs(directory, exist_ok=True)
        self._db_path = os.path.join(directory, "dspy_cache.db")
        self._size_limit = size_limit
        self._lock = threading.Lock()
        self._pid = os.getpid()
        self._conn = self._connect()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA mmap_size=67108864")
        conn.execute("PRAGMA cache_size=8192")
        conn.execute("PRAGMA auto_vacuum=FULL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  key TEXT PRIMARY KEY,"
            "  value BLOB NOT NULL,"
            "  size INTEGER NOT NULL,"
            "  last_access REAL NOT NULL"
            ")"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_last_access ON cache(last_access)")
        conn.commit()
        return conn

    def _reset_after_fork(self) -> None:
        pid = os.getpid()
        if pid == self._pid:
            return
        # Locks and SQLite connections are process-local. Recreate both in the child.
        self._lock = threading.Lock()
        new_conn = self._connect()
        self._pid = pid
        self._conn = new_conn

    def _get_conn(self) -> sqlite3.Connection:
        """Return the SQLite connection, recreating process-local state after fork."""
        self._reset_after_fork()
        return self._conn

    def __contains__(self, key: str) -> bool:
        self._reset_after_fork()
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT 1 FROM cache WHERE key = ?", (key,)).fetchone()
        return row is not None

    def __getitem__(self, key: str) -> Any:
        self._reset_after_fork()
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
            if row is None:
                raise KeyError(key)
            conn.execute("UPDATE cache SET last_access = ? WHERE key = ?", (time.time(), key))
            conn.commit()
        return _deserialize(row[0])

    def __setitem__(self, key: str, value: Any) -> None:
        blob = _serialize(value)
        size = len(blob)
        now = time.time()
        self._reset_after_fork()
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, size, last_access) VALUES (?, ?, ?, ?)",
                (key, blob, size, now),
            )
            conn.commit()
            self._maybe_evict(conn)

    def is_empty(self) -> bool:
        self._reset_after_fork()
        with self._lock:
            conn = self._get_conn()
            return conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 0

    def _maybe_evict(self, conn: sqlite3.Connection) -> None:
        """Evict least-recently-accessed entries until total size is within the limit.

        Must be called while self._lock is held.
        """
        if self._size_limit is None or self._size_limit <= 0:
            return
        total_size = conn.execute("SELECT COALESCE(SUM(size), 0) FROM cache").fetchone()[0]
        if total_size <= self._size_limit:
            return
        overage = total_size - self._size_limit
        evicted = 0
        rows = conn.execute("SELECT key, size FROM cache ORDER BY last_access ASC").fetchall()
        keys_to_delete = []
        for row_key, row_size in rows:
            keys_to_delete.append(row_key)
            evicted += row_size
            if evicted >= overage:
                break
        if keys_to_delete:
            placeholders = ",".join("?" for _ in keys_to_delete)
            conn.execute(f"DELETE FROM cache WHERE key IN ({placeholders})", keys_to_delete)
            conn.commit()

    def bulk_set(self, entries: list[tuple[str, bytes, int, float]]) -> None:
        """Insert multiple pre-serialized entries in a single transaction.

        Each entry is (key, serialized_blob, blob_size, last_access_time).
        """
        if not entries:
            return
        self._reset_after_fork()
        with self._lock:
            conn = self._get_conn()
            conn.executemany(
                "INSERT OR REPLACE INTO cache (key, value, size, last_access) VALUES (?, ?, ?, ?)",
                entries,
            )
            conn.commit()
            self._maybe_evict(conn)

    def close(self) -> None:
        self._reset_after_fork()
        self._conn.close()
def has_legacy_diskcache(directory: str) -> bool:
    """Check if directory contains an old diskcache FanoutCache (16-shard) layout."""
    return os.path.isfile(os.path.join(directory, "000", "cache.db"))


def migrate_diskcache(directory: str, target: SQLiteCache) -> tuple[int, int]:
    """Migrate entries from a legacy diskcache FanoutCache (16 shards) into *target*.

    Values in the old cache are pickle-serialized. Each value is unpickled and
    re-serialized with orjson before being inserted into the new SQLiteCache.
    The original ``access_time`` is preserved so LRU eviction order is maintained.

    Returns ``(migrated_count, error_count)``.
    """
    migrated = 0
    errors = 0

    for shard_id in range(16):
        shard_dir = os.path.join(directory, f"{shard_id:03d}")
        shard_db = os.path.join(shard_dir, "cache.db")

        if not os.path.isfile(shard_db):
            continue

        conn = sqlite3.connect(shard_db, timeout=10)
        try:
            rows = conn.execute(
                "SELECT key, value, mode, access_time, filename FROM Cache WHERE raw = 1"
            ).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("Skipping diskcache shard %03d: %s", shard_id, e)
            conn.close()
            continue
        conn.close()

        entries: list[tuple[str, bytes, int, float]] = []
        for key, value_blob, mode, access_time, filename in rows:
            try:
                if mode == _DC_MODE_PICKLE:
                    if value_blob is not None:
                        obj = pickle.loads(value_blob)
                    else:
                        # Large pickle values stored as .val files in the shard directory
                        filepath = os.path.join(shard_dir, filename)
                        with open(filepath, "rb") as f:
                            obj = pickle.loads(f.read())
                elif mode == _DC_MODE_TEXT:
                    filepath = os.path.join(shard_dir, filename)
                    with open(filepath, encoding="UTF-8") as f:
                        obj = f.read()
                elif mode == _DC_MODE_RAW:
                    obj = bytes(value_blob) if value_blob is not None else b""
                elif mode == _DC_MODE_BINARY:
                    filepath = os.path.join(shard_dir, filename)
                    with open(filepath, "rb") as f:
                        obj = f.read()
                else:
                    logger.debug("Skipping entry with unknown diskcache mode %d", mode)
                    errors += 1
                    continue

                blob = _serialize(obj)
                entries.append((key, blob, len(blob), access_time if access_time is not None else time.time()))
            except (
                pickle.UnpicklingError,
                ModuleNotFoundError,
                AttributeError,
                ImportError,
                TypeError,
                OSError,
            ) as e:
                errors += 1
                logger.debug("Failed to migrate cache entry %.16s: %s", key, e)

        if entries:
            target.bulk_set(entries)
            migrated += len(entries)

    return migrated, errors
