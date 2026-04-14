"""Legacy diskcache migration utilities.

Migrates entries from old pickle-based FanoutCache (16 shards) into
a new diskcache FanoutCache with OrjsonDisk (orjson serialization).
"""

import logging
import os
import pickle
import sqlite3
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_NUM_SHARDS = 16

# Legacy diskcache mode constants (used during migration from FanoutCache shards)
_DC_MODE_RAW = 1
_DC_MODE_BINARY = 2
_DC_MODE_TEXT = 3
_DC_MODE_PICKLE = 4


@dataclass
class LegacyReadReport:
    entries: list[tuple[str, Any]]
    row_count: int
    read_failures: int


def _rebuild_incomplete_models(obj: Any) -> None:
    """Call ``model_rebuild()`` on any incomplete pydantic model classes in *obj*.

    Unpickling pydantic models across Python versions, or when the openai/litellm
    SDK defers model building (``DEFER_PYDANTIC_BUILD``), can leave classes with a
    ``MockValSer`` placeholder instead of a real ``SchemaSerializer``.  Calling
    ``model_rebuild()`` forces the class to build its schema; this is a no-op if
    the class is already complete.

    See https://github.com/pydantic/pydantic/issues/7713
    """
    import pydantic

    seen: set[type] = set()

    def _walk(value: Any) -> None:
        if isinstance(value, pydantic.BaseModel):
            cls = type(value)
            if cls not in seen:
                seen.add(cls)
                if not cls.__pydantic_complete__:
                    cls.model_rebuild()
            for field_value in dict(value).values():
                _walk(field_value)
        elif isinstance(value, dict):
            for v in value.values():
                _walk(v)
        elif isinstance(value, list):
            for item in value:
                _walk(item)

    _walk(obj)


def _deserialize_legacy_entry(mode: int, value_blob: bytes | None, shard_dir: str, filename: str | None) -> Any:
    """Deserialize a single legacy diskcache entry based on its storage mode.

    Returns the deserialized Python object, or raises on failure.
    """
    if mode == _DC_MODE_PICKLE:
        if value_blob is not None:
            obj = pickle.loads(value_blob)
        else:
            filepath = os.path.join(shard_dir, filename)
            with open(filepath, "rb") as f:
                obj = pickle.loads(f.read())
        _rebuild_incomplete_models(obj)
        return obj
    if mode == _DC_MODE_TEXT:
        filepath = os.path.join(shard_dir, filename)
        with open(filepath, encoding="UTF-8") as f:
            return f.read()
    if mode == _DC_MODE_RAW:
        return bytes(value_blob) if value_blob is not None else b""
    if mode == _DC_MODE_BINARY:
        filepath = os.path.join(shard_dir, filename)
        with open(filepath, "rb") as f:
            return f.read()
    raise ValueError(f"Unknown diskcache mode {mode}")


def inspect_legacy_entries(directory: str) -> LegacyReadReport:
    """Read all legacy diskcache entries from 16 shards and report read failures.

    Must be called BEFORE FanoutCache is created in the same directory,
    since FanoutCache will overwrite the shard DB files.

    Entries that fail to deserialize (corrupt pickle, missing modules, etc.)
    are counted in ``read_failures`` so callers can decide whether to abort
    migration and preserve the source cache.
    """
    entries = []
    row_count = 0
    read_failures = 0
    for shard_id in range(_NUM_SHARDS):
        shard_dir = os.path.join(directory, f"{shard_id:03d}")
        shard_db = os.path.join(shard_dir, "cache.db")
        if not os.path.isfile(shard_db):
            continue
        conn = sqlite3.connect(shard_db, timeout=10)
        try:
            # raw=1 selects entries with raw-bytes keys, which is the default for
            # string keys in diskcache (all DSPy cache keys are SHA-256 hex strings).
            rows = conn.execute(
                "SELECT key, value, mode, access_time, filename FROM Cache WHERE raw = 1"
            ).fetchall()
        except sqlite3.OperationalError as e:
            read_failures += 1
            logger.warning("Failed to read diskcache shard %03d during migration: %s", shard_id, e)
            continue
        finally:
            conn.close()
        for key, value_blob, mode, _access_time, filename in rows:
            row_count += 1
            try:
                obj = _deserialize_legacy_entry(mode, value_blob, shard_dir, filename)
                entries.append((key, obj))
            except ValueError as e:
                read_failures += 1
                logger.debug("Failed to read legacy cache entry %.16s: %s", key, e)
            except (
                pickle.UnpicklingError,
                ModuleNotFoundError,
                AttributeError,
                ImportError,
                TypeError,
                OSError,
            ) as e:
                read_failures += 1
                logger.debug("Failed to read legacy cache entry %.16s: %s", key, e)
    return LegacyReadReport(entries=entries, row_count=row_count, read_failures=read_failures)


def read_legacy_entries(directory: str) -> list[tuple[str, Any]]:
    """Read all legacy diskcache entries from 16 shards, returning (key, value) pairs."""
    return inspect_legacy_entries(directory).entries


def remove_legacy_shard_dbs(directory: str) -> None:
    """Delete legacy shard DB files so FanoutCache starts with clean tables.

    Without this, FanoutCache reuses existing DBs and stale pickle rows
    corrupt its internal count/size bookkeeping.

    Logs a warning and continues if a file cannot be removed (e.g. read-only filesystem).
    """
    for shard_id in range(_NUM_SHARDS):
        shard_dir = os.path.join(directory, f"{shard_id:03d}")
        for suffix in ("cache.db", "cache.db-wal", "cache.db-shm"):
            path = os.path.join(shard_dir, suffix)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError as e:
                    logger.warning("Could not remove legacy shard file %s: %s", path, e)


def migrate_diskcache(directory: str, target) -> tuple[int, int]:
    """Migrate entries from a legacy diskcache FanoutCache (16 shards) into *target*.

    Values in the old cache are pickle-serialized. Each value is unpickled and
    re-stored into the new cache using its serialization format (orjson via OrjsonDisk).
    The target should be a diskcache.FanoutCache (or any dict-like cache).

    Returns ``(migrated_count, error_count)``.
    """
    entries = read_legacy_entries(directory)
    migrated = 0
    errors = 0
    for key, value in entries:
        try:
            target[key] = value
            migrated += 1
        except (TypeError, OSError) as e:
            errors += 1
            logger.debug("Failed to write migrated entry %.16s: %s", key, e)
    return migrated, errors
