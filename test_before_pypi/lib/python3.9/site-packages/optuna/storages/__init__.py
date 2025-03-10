from __future__ import annotations

from optuna.storages._base import BaseStorage
from optuna.storages._cached_storage import _CachedStorage
from optuna.storages._callbacks import RetryFailedTrialCallback
from optuna.storages._grpc import GrpcStorageProxy
from optuna.storages._grpc import run_grpc_proxy_server
from optuna.storages._heartbeat import fail_stale_trials
from optuna.storages._in_memory import InMemoryStorage
from optuna.storages._rdb.storage import RDBStorage
from optuna.storages.journal._base import BaseJournalLogStorage
from optuna.storages.journal._file import (
    DeprecatedJournalFileSymlinkLock as JournalFileSymlinkLock,
)
from optuna.storages.journal._file import DeprecatedJournalFileOpenLock as JournalFileOpenLock
from optuna.storages.journal._file import JournalFileStorage
from optuna.storages.journal._redis import JournalRedisStorage
from optuna.storages.journal._storage import JournalStorage


__all__ = [
    "BaseStorage",
    "BaseJournalLogStorage",
    "InMemoryStorage",
    "RDBStorage",
    "JournalStorage",
    "JournalFileStorage",
    "JournalRedisStorage",
    "JournalFileSymlinkLock",
    "JournalFileOpenLock",
    "RetryFailedTrialCallback",
    "_CachedStorage",
    "fail_stale_trials",
    "GrpcStorageProxy",
    "run_grpc_proxy_server",
]


def get_storage(storage: None | str | BaseStorage) -> BaseStorage:
    """Only for internal usage. It might be deprecated in the future."""

    if storage is None:
        return InMemoryStorage()
    if isinstance(storage, str):
        if storage.startswith("redis"):
            raise ValueError(
                "RedisStorage is removed at Optuna v3.1.0. Please use JournalRedisBackend instead."
            )
        return _CachedStorage(RDBStorage(storage))
    elif isinstance(storage, RDBStorage):
        return _CachedStorage(storage)
    else:
        return storage
