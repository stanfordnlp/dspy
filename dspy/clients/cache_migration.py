"""Legacy pickle-to-orjson cache migration utilities."""

import logging
import os
import pickle
import shutil
import sqlite3
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import diskcache
import orjson
import pydantic

from dspy.clients.disk_serialization import OrjsonDisk

logger = logging.getLogger(__name__)

_NUM_SHARDS = 16

# Legacy diskcache mode constants for staged migration from pickle-backed shards.
_DC_MODE_RAW = 1
_DC_MODE_BINARY = 2
_DC_MODE_TEXT = 3
_DC_MODE_PICKLE = 4


class CacheMigrationError(RuntimeError):
    """Raised when an explicit disk cache migration cannot complete safely."""


class LegacyCacheReadError(RuntimeError):
    """Raised when a legacy cache entry cannot be read safely."""


def _rebuild_incomplete_models(obj: Any) -> None:
    """Call ``model_rebuild()`` on any incomplete pydantic model classes in *obj*."""
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
            for child in value.values():
                _walk(child)
        elif isinstance(value, list):
            for child in value:
                _walk(child)

    _walk(obj)


def _deserialize_legacy_entry(mode: int, value_blob: bytes | None, shard_dir: str, filename: str | None) -> Any:
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


def _iter_legacy_entries(directory: str) -> Iterator[tuple[str, Any]]:
    """Yield decoded legacy cache entries from *directory*."""
    for shard_id in range(_NUM_SHARDS):
        shard_dir = os.path.join(directory, f"{shard_id:03d}")
        shard_db = os.path.join(shard_dir, "cache.db")
        if not os.path.isfile(shard_db):
            continue
        conn = sqlite3.connect(shard_db, timeout=10)
        try:
            cursor = conn.execute("SELECT key, value, mode, filename FROM Cache WHERE raw = 1")
            for key, value_blob, mode, filename in cursor:
                try:
                    yield key, _deserialize_legacy_entry(mode, value_blob, shard_dir, filename)
                except (
                    ValueError,
                    pickle.UnpicklingError,
                    ModuleNotFoundError,
                    AttributeError,
                    ImportError,
                    TypeError,
                    OSError,
                ) as e:
                    raise LegacyCacheReadError(f"Failed to read legacy cache entry {key!r}") from e
        except sqlite3.OperationalError as e:
            raise LegacyCacheReadError(f"Failed to read diskcache shard {shard_id:03d}") from e
        finally:
            conn.close()


def _create_orjson_disk_cache(
    disk_cache_dir: str,
    effective_limit: int,
    fanout_kwargs: dict[str, str],
) -> diskcache.FanoutCache:
    return diskcache.FanoutCache(
        directory=disk_cache_dir,
        shards=_NUM_SHARDS,
        disk=OrjsonDisk,
        size_limit=effective_limit,
        eviction_policy="least-recently-stored",
        timeout=60,
        **fanout_kwargs,
    )


def _remove_staging_dir(staging_dir: str) -> None:
    """Remove a migration staging/backup directory, logging on failure."""
    if not os.path.exists(staging_dir):
        return

    try:
        shutil.rmtree(staging_dir)
    except OSError as e:
        logger.warning("Could not remove staging directory %s: %s", staging_dir, e)


def _promote_staging_dir(disk_cache_dir: str, staging_dir: str) -> None:
    """Promote a fully-migrated staging cache into place without deleting the source first."""
    backup_dir = None
    if os.path.exists(disk_cache_dir):
        backup_dir = f"{disk_cache_dir}.dspy_legacy_backup.{uuid4().hex}"

    try:
        if backup_dir is not None:
            os.replace(disk_cache_dir, backup_dir)
        os.replace(staging_dir, disk_cache_dir)
    except OSError as e:
        if backup_dir is not None and os.path.exists(backup_dir) and not os.path.exists(disk_cache_dir):
            try:
                os.replace(backup_dir, disk_cache_dir)
            except OSError as rollback_error:
                raise CacheMigrationError(
                    "Failed to promote the migrated cache and failed to restore the legacy "
                    f"cache from {backup_dir}: {rollback_error}"
                ) from e

        raise CacheMigrationError(
            f"Failed to promote the migrated cache from {staging_dir}; the legacy cache was preserved."
        ) from e

    if backup_dir is not None:
        _remove_staging_dir(backup_dir)


def migrate_legacy_cache(
    disk_cache_dir: str,
    effective_limit: int,
    fanout_kwargs: dict[str, str],
) -> None:
    """Copy legacy pickle entries into a staging orjson cache and swap on success."""
    staging_dir = f"{disk_cache_dir}.dspy_orjson_staging"
    _remove_staging_dir(staging_dir)

    staging_cache = _create_orjson_disk_cache(staging_dir, effective_limit, fanout_kwargs)
    migrated = 0
    read_failure: LegacyCacheReadError | None = None
    write_failure: TypeError | orjson.JSONEncodeError | diskcache.Timeout | OSError | sqlite3.OperationalError | None = None
    try:
        for key, value in _iter_legacy_entries(disk_cache_dir):
            try:
                staging_cache[key] = value
            except (TypeError, orjson.JSONEncodeError, diskcache.Timeout, OSError, sqlite3.OperationalError) as e:
                write_failure = e
                break
            migrated += 1
    except LegacyCacheReadError as e:
        read_failure = e
    finally:
        staging_cache.close()

    if read_failure is not None:
        _remove_staging_dir(staging_dir)
        raise CacheMigrationError(
            "Cache migration aborted after a read failure; the legacy cache was preserved. "
            "Fix the unreadable entry and re-run with DSPY_MIGRATE_CACHE=1."
        ) from read_failure

    if write_failure is not None:
        _remove_staging_dir(staging_dir)
        raise CacheMigrationError(
            "Cache migration aborted after a write failure; the legacy cache was preserved. "
            "Re-run with DSPY_MIGRATE_CACHE=1 after fixing the failing entry."
        ) from write_failure

    if migrated == 0:
        _remove_staging_dir(staging_dir)
        return

    _promote_staging_dir(disk_cache_dir, staging_dir)
