from __future__ import annotations

import json
import time
from typing import Any

from optuna._deprecated import deprecated_class
from optuna._experimental import experimental_class
from optuna._imports import try_import
from optuna.storages.journal._base import BaseJournalBackend
from optuna.storages.journal._base import BaseJournalSnapshot


with try_import() as _imports:
    import redis


@experimental_class("3.1.0")
class JournalRedisBackend(BaseJournalBackend, BaseJournalSnapshot):
    """Redis storage class for Journal log backend.

    Args:
        url:
            URL of the redis storage, password and db are optional.
            (ie: ``redis://localhost:6379``)
        use_cluster:
            Flag whether you use the Redis cluster. If this is :obj:`False`, it is assumed that
            you use the standalone Redis server and ensured that a write operation is atomic. This
            provides the consistency of the preserved logs. If this is :obj:`True`, it is assumed
            that you use the Redis cluster and not ensured that a write operation is atomic. This
            means the preserved logs can be inconsistent due to network errors, and may
            cause errors.
        prefix:
            Prefix of the preserved key of logs. This is useful when multiple users work on one
            Redis server.
    """

    def __init__(self, url: str, use_cluster: bool = False, prefix: str = "") -> None:
        _imports.check()

        self._url = url
        self._redis = redis.Redis.from_url(url)
        self._use_cluster = use_cluster
        self._prefix = prefix

    def __getstate__(self) -> dict[Any, Any]:
        state = self.__dict__.copy()
        del state["_redis"]
        return state

    def __setstate__(self, state: dict[Any, Any]) -> None:
        self.__dict__.update(state)
        self._redis = redis.Redis.from_url(self._url)

    def read_logs(self, log_number_from: int) -> list[dict[str, Any]]:
        max_log_number_bytes = self._redis.get(f"{self._prefix}:log_number")
        if max_log_number_bytes is None:
            return []
        max_log_number = int(max_log_number_bytes)

        logs = []
        for log_number in range(log_number_from, max_log_number + 1):
            sleep_secs = 0.1
            while True:
                log = self._redis.get(self._key_log_id(log_number))
                if log is not None:
                    break
                time.sleep(sleep_secs)
                sleep_secs = min(sleep_secs * 2, 10)
            try:
                logs.append(json.loads(log))
            except json.JSONDecodeError as err:
                if log_number != max_log_number:
                    raise err
        return logs

    def append_logs(self, logs: list[dict[str, Any]]) -> None:
        self._redis.setnx(f"{self._prefix}:log_number", -1)
        for log in logs:
            if not self._use_cluster:
                self._redis.eval(  # type: ignore
                    "local i = redis.call('incr', string.format('%s:log_number', ARGV[1])) "
                    "redis.call('set', string.format('%s:log:%d', ARGV[1], i), ARGV[2])",
                    0,
                    self._prefix,
                    json.dumps(log),
                )
            else:
                log_number = self._redis.incr(f"{self._prefix}:log_number", 1)
                self._redis.set(self._key_log_id(log_number), json.dumps(log))

    def save_snapshot(self, snapshot: bytes) -> None:
        self._redis.set(f"{self._prefix}:snapshot", snapshot)

    def load_snapshot(self) -> bytes | None:
        snapshot_bytes = self._redis.get(f"{self._prefix}:snapshot")
        return snapshot_bytes

    def _key_log_id(self, log_number: int) -> str:
        return f"{self._prefix}:log:{log_number}"


@deprecated_class(
    "4.0.0", "6.0.0", text="Use :class:`~optuna.storages.journal.JournalRedisBackend` instead."
)
class JournalRedisStorage(JournalRedisBackend):
    pass
