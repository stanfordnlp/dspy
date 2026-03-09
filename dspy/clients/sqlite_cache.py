import dataclasses
import importlib
import logging
import os
import pickle
import sqlite3
import threading
import time
from typing import Any

import orjson
import pydantic

logger = logging.getLogger(__name__)

# diskcache mode constants
_DC_MODE_RAW = 1
_DC_MODE_BINARY = 2
_DC_MODE_TEXT = 3
_DC_MODE_PICKLE = 4




def _serialize(value: Any) -> bytes:
    if isinstance(value, pydantic.BaseModel):
        envelope = {
            "_pydantic": f"{type(value).__module__}.{type(value).__qualname__}",
            "_data": value.model_dump(mode="json"),
        }
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        envelope = {
            "_dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "_data": dataclasses.asdict(value),
        }
    else:
        envelope = {"_data": value}
    return orjson.dumps(envelope)


def _resolve_class(qualname: str) -> type:
    parts = qualname.rsplit(".", 1)
    module_name, attr_path = parts[0], parts[1]
    # Try importing progressively shorter module paths to handle nested classes
    # e.g. "mymod.Outer.Inner" -> try "mymod.Outer" (fails), then "mymod" + getattr "Outer.Inner"
    while True:
        try:
            module = importlib.import_module(module_name)
            break
        except ModuleNotFoundError:
            dot = module_name.rfind(".")
            if dot == -1:
                raise
            attr_path = module_name[dot + 1 :] + "." + attr_path
            module_name = module_name[:dot]
    obj = module
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def _deserialize(blob: bytes) -> Any:
    envelope = orjson.loads(blob)
    if "_pydantic" in envelope:
        cls = _resolve_class(envelope["_pydantic"])
        return cls(**envelope["_data"])
    if "_dataclass" in envelope:
        cls = _resolve_class(envelope["_dataclass"])
        return cls(**envelope["_data"])
    return envelope["_data"]


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

    def _get_conn(self) -> sqlite3.Connection:
        """Return the SQLite connection, creating a new one after fork."""
        pid = os.getpid()
        if pid != self._pid:
            self._pid = pid
            self._conn = self._connect()
        return self._conn

    def __contains__(self, key: str) -> bool:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT 1 FROM cache WHERE key = ?", (key,)).fetchone()
        return row is not None

    def __getitem__(self, key: str) -> Any:
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
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, size, last_access) VALUES (?, ?, ?, ?)",
                (key, blob, size, now),
            )
            conn.commit()
            self._maybe_evict(conn)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict) and len(other) == 0:
            with self._lock:
                conn = self._get_conn()
                return conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 0
        return NotImplemented

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
        with self._lock:
            conn = self._get_conn()
            conn.executemany(
                "INSERT OR REPLACE INTO cache (key, value, size, last_access) VALUES (?, ?, ?, ?)",
                entries,
            )
            conn.commit()
            self._maybe_evict(conn)

    def close(self) -> None:
        self._conn.close()


def _extract_json_data(obj: Any) -> Any:
    """Recursively extract JSON-safe data from an object.

    Pydantic models unpickled from older versions may have broken serializers
    (``model_dump`` raises ``TypeError: 'MockValSer' ...``).  This helper
    walks the object tree via ``__dict__`` and ``__pydantic_extra__`` instead,
    producing a plain dict / list / scalar structure that ``orjson`` can handle.
    """
    if isinstance(obj, pydantic.BaseModel):
        data = {}
        for k, v in obj.__dict__.items():
            if not k.startswith("_"):
                data[k] = _extract_json_data(v)
        # Some pydantic models (e.g. litellm's OpenAIObject subclasses) store
        # all data in __pydantic_extra__ rather than __dict__
        extra = getattr(obj, "__pydantic_extra__", None)
        if extra:
            for k, v in extra.items():
                if k not in data:
                    data[k] = _extract_json_data(v)
        return data
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _extract_json_data(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    if isinstance(obj, dict):
        return {k: _extract_json_data(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_extract_json_data(item) for item in obj]
    return obj


def _serialize_for_migration(value: Any) -> bytes:
    """Serialize a value for migration from legacy pickle caches.

    Unlike ``_serialize``, this does not call ``model_dump`` (which can fail on
    pydantic objects that were unpickled across pydantic versions).  Instead it
    recursively extracts data via ``__dict__`` and produces the same envelope
    format that ``_deserialize`` expects.
    """
    if isinstance(value, pydantic.BaseModel):
        envelope = {
            "_pydantic": f"{type(value).__module__}.{type(value).__qualname__}",
            "_data": _extract_json_data(value),
        }
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        envelope = {
            "_dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "_data": _extract_json_data(value),
        }
    else:
        envelope = {"_data": _extract_json_data(value)}
    return orjson.dumps(envelope)


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

                blob = _serialize_for_migration(obj)
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
