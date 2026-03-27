"""Legacy diskcache migration utilities.

Migrates entries from old pickle-based FanoutCache (16 shards) into
a new diskcache FanoutCache with OrjsonDisk (orjson serialization).
"""

import logging
import os
import pickle
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

# Legacy diskcache mode constants (used during migration from FanoutCache shards)
_DC_MODE_RAW = 1
_DC_MODE_BINARY = 2
_DC_MODE_TEXT = 3
_DC_MODE_PICKLE = 4


def _deserialize_legacy_entry(mode: int, value_blob: bytes | None, shard_dir: str, filename: str | None) -> Any:
    """Deserialize a single legacy diskcache entry based on its storage mode.

    Returns the deserialized Python object, or raises on failure.
    """
    if mode == _DC_MODE_PICKLE:
        if value_blob is not None:
            return pickle.loads(value_blob)
        filepath = os.path.join(shard_dir, filename)
        with open(filepath, "rb") as f:
            return pickle.loads(f.read())
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


def read_legacy_entries(directory: str) -> list[tuple[str, Any]]:
    """Read all legacy diskcache entries from 16 shards, returning (key, value) pairs.

    Must be called BEFORE FanoutCache is created in the same directory,
    since FanoutCache will overwrite the shard DB files.

    Entries that fail to deserialize (corrupt pickle, missing modules, etc.)
    are silently skipped.
    """
    entries = []
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
        for key, value_blob, mode, access_time, filename in rows:
            try:
                obj = _deserialize_legacy_entry(mode, value_blob, shard_dir, filename)
                entries.append((key, obj))
            except ValueError:
                logger.debug("Skipping entry with unknown diskcache mode %d", mode)
            except (
                pickle.UnpicklingError,
                ModuleNotFoundError,
                AttributeError,
                ImportError,
                TypeError,
                OSError,
            ) as e:
                logger.debug("Failed to read legacy cache entry %.16s: %s", key, e)
    return entries


def remove_legacy_shard_dbs(directory: str) -> None:
    """Delete legacy shard DB files so FanoutCache starts with clean tables.

    Without this, FanoutCache reuses existing DBs and stale pickle rows
    corrupt its internal count/size bookkeeping.
    """
    for shard_id in range(16):
        shard_dir = os.path.join(directory, f"{shard_id:03d}")
        for suffix in ("cache.db", "cache.db-wal", "cache.db-shm"):
            path = os.path.join(shard_dir, suffix)
            if os.path.isfile(path):
                os.remove(path)


def migrate_diskcache(directory: str, target) -> tuple[int, int]:
    """Migrate entries from a legacy diskcache FanoutCache (16 shards) into *target*.

    Values in the old cache are pickle-serialized. Each value is unpickled and
    re-stored into the new cache using its serialization format (orjson via OrjsonDisk).
    The target should be a diskcache.FanoutCache (or any dict-like cache).

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

        for key, value_blob, mode, access_time, filename in rows:
            try:
                obj = _deserialize_legacy_entry(mode, value_blob, shard_dir, filename)
                target[key] = obj
                migrated += 1
            except ValueError:
                logger.debug("Skipping entry with unknown diskcache mode %d", mode)
                errors += 1
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

    return migrated, errors
