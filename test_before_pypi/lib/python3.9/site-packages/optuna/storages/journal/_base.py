from __future__ import annotations

import abc
from typing import Any

from optuna._deprecated import deprecated_class


class BaseJournalBackend(abc.ABC):
    """Base class for Journal storages.

    Storage classes implementing this base class must guarantee process safety. This means,
    multiple processes might concurrently call ``read_logs`` and ``append_logs``. If the
    backend storage does not internally support mutual exclusion mechanisms, such as locks,
    you might want to use :class:`~optuna.storages.journal.JournalFileSymlinkLock` or
    :class:`~optuna.storages.journal.JournalFileOpenLock` for creating a critical section.

    """

    @abc.abstractmethod
    def read_logs(self, log_number_from: int) -> list[dict[str, Any]]:
        """Read logs with a log number greater than or equal to ``log_number_from``.

        If ``log_number_from`` is 0, read all the logs.

        Args:
            log_number_from:
                A non-negative integer value indicating which logs to read.

        Returns:
            Logs with log number greater than or equal to ``log_number_from``.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def append_logs(self, logs: list[dict[str, Any]]) -> None:
        """Append logs to the backend.

        Args:
            logs:
                A list that contains json-serializable logs.
        """

        raise NotImplementedError


class BaseJournalSnapshot(abc.ABC):
    """Optional base class for Journal storages.

    Storage classes implementing this base class may work faster when
    constructing the internal state from the large amount of logs.
    """

    @abc.abstractmethod
    def save_snapshot(self, snapshot: bytes) -> None:
        """Save snapshot to the backend.

        Args:
            snapshot: A serialized snapshot (bytes)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_snapshot(self) -> bytes | None:
        """Load snapshot from the backend.

        Returns:
            A serialized snapshot (bytes) if found, otherwise :obj:`None`.
        """
        raise NotImplementedError


@deprecated_class(
    "4.0.0", "6.0.0", text="Use :class:`~optuna.storages.journal.BaseJournalBackend` instead."
)
class BaseJournalLogStorage(BaseJournalBackend):
    """Base class for Journal storages.

    Storage classes implementing this base class must guarantee process safety. This means,
    multiple processes might concurrently call ``read_logs`` and ``append_logs``. If the
    backend storage does not internally support mutual exclusion mechanisms, such as locks,
    you might want to use :class:`~optuna.storages.journal.JournalFileSymlinkLock` or
    :class:`~optuna.storages.journal.JournalFileOpenLock` for creating a critical section.

    """
