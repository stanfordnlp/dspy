from __future__ import annotations

import abc
from collections.abc import Iterator
from contextlib import contextmanager
import errno
import json
import os
import time
from typing import Any
import uuid

from optuna._deprecated import deprecated_class
from optuna.storages.journal._base import BaseJournalBackend


LOCK_FILE_SUFFIX = ".lock"
RENAME_FILE_SUFFIX = ".rename"


class JournalFileBackend(BaseJournalBackend):
    """File storage class for Journal log backend.

    Compared to SQLite3, the benefit of this backend is that it is more suitable for
    environments where the file system does not support ``fcntl()`` file locking.
    For example, as written in the `SQLite3 FAQ <https://www.sqlite.org/faq.html#q5>`__,
    SQLite3 might not work on NFS (Network File System) since ``fcntl()`` file locking
    is broken on many NFS implementations. In such scenarios, this backend provides
    several workarounds for locking files. For more details, refer to the `Medium blog post`_.

    .. _Medium blog post: https://medium.com/optuna/distributed-optimization-via-nfs\
    -using-optunas-new-operation-based-logging-storage-9815f9c3f932

    It's important to note that, similar to SQLite3, this class doesn't support a high
    level of write concurrency, as outlined in the `SQLAlchemy documentation`_. However,
    in typical situations where the objective function is computationally expensive, Optuna
    users don't need to be concerned about this limitation. The reason being, the write
    operations are not the bottleneck as long as the objective function doesn't invoke
    :meth:`~optuna.trial.Trial.report` and :meth:`~optuna.trial.Trial.set_user_attr` excessively.

    .. _SQLAlchemy documentation: https://docs.sqlalchemy.org/en/20/dialects/sqlite.html\
    #database-locking-behavior-concurrency

    Args:
        file_path:
            Path of file to persist the log to.

        lock_obj:
            Lock object for process exclusivity. An instance of
            :class:`~optuna.storages.journal.JournalFileSymlinkLock` and
            :class:`~optuna.storages.journal.JournalFileOpenLock` can be passed.
    """

    def __init__(self, file_path: str, lock_obj: BaseJournalFileLock | None = None) -> None:
        self._file_path: str = file_path
        self._lock = lock_obj or JournalFileSymlinkLock(self._file_path)
        if not os.path.exists(self._file_path):
            open(self._file_path, "ab").close()  # Create a file if it does not exist.
        self._log_number_offset: dict[int, int] = {0: 0}

    def read_logs(self, log_number_from: int) -> list[dict[str, Any]]:
        logs = []
        with open(self._file_path, "rb") as f:
            # Maintain remaining_log_size to allow writing by another process
            # while reading the log.
            remaining_log_size = os.stat(self._file_path).st_size
            log_number_start = 0
            if log_number_from in self._log_number_offset:
                f.seek(self._log_number_offset[log_number_from])
                log_number_start = log_number_from
                remaining_log_size -= self._log_number_offset[log_number_from]

            last_decode_error = None
            for log_number, line in enumerate(f, start=log_number_start):
                byte_len = len(line)
                remaining_log_size -= byte_len
                if remaining_log_size < 0:
                    break
                if last_decode_error is not None:
                    raise last_decode_error
                if log_number + 1 not in self._log_number_offset:
                    self._log_number_offset[log_number + 1] = (
                        self._log_number_offset[log_number] + byte_len
                    )
                if log_number < log_number_from:
                    continue

                # Ensure that each line ends with line separators (\n, \r\n).
                if not line.endswith(b"\n"):
                    last_decode_error = ValueError("Invalid log format.")
                    del self._log_number_offset[log_number + 1]
                    continue
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError as err:
                    last_decode_error = err
                    del self._log_number_offset[log_number + 1]
            return logs

    def append_logs(self, logs: list[dict[str, Any]]) -> None:
        with get_lock_file(self._lock):
            what_to_write = (
                "\n".join([json.dumps(log, separators=(",", ":")) for log in logs]) + "\n"
            )
            with open(self._file_path, "ab") as f:
                f.write(what_to_write.encode("utf-8"))
                f.flush()
                os.fsync(f.fileno())


class BaseJournalFileLock(abc.ABC):
    @abc.abstractmethod
    def acquire(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def release(self) -> None:
        raise NotImplementedError


class JournalFileSymlinkLock(BaseJournalFileLock):
    """Lock class for synchronizing processes for NFSv2 or later.

    On acquiring the lock, link system call is called to create an exclusive file. The file is
    deleted when the lock is released. In NFS environments prior to NFSv3, use this instead of
    :class:`~optuna.storages.journal.JournalFileOpenLock`.

    Args:
        filepath:
            The path of the file whose race condition must be protected.
    """

    def __init__(self, filepath: str) -> None:
        self._lock_target_file = filepath
        self._lock_file = filepath + LOCK_FILE_SUFFIX

    def acquire(self) -> bool:
        """Acquire a lock in a blocking way by creating a symbolic link of a file.

        Returns:
            :obj:`True` if it succeeded in creating a symbolic link of ``self._lock_target_file``.

        """
        sleep_secs = 0.001
        while True:
            try:
                os.symlink(self._lock_target_file, self._lock_file)
                return True
            except OSError as err:
                if err.errno == errno.EEXIST:
                    time.sleep(sleep_secs)
                    sleep_secs = min(sleep_secs * 2, 1)
                    continue
                raise err
            except BaseException:
                self.release()
                raise

    def release(self) -> None:
        """Release a lock by removing the symbolic link."""

        lock_rename_file = self._lock_file + str(uuid.uuid4()) + RENAME_FILE_SUFFIX
        try:
            os.rename(self._lock_file, lock_rename_file)
            os.unlink(lock_rename_file)
        except OSError:
            raise RuntimeError("Error: did not possess lock")
        except BaseException:
            os.unlink(lock_rename_file)
            raise


class JournalFileOpenLock(BaseJournalFileLock):
    """Lock class for synchronizing processes for NFSv3 or later.

    On acquiring the lock, open system call is called with the O_EXCL option to create an exclusive
    file. The file is deleted when the lock is released. This class is only supported when using
    NFSv3 or later on kernel 2.6 or later. In prior NFS environments, use
    :class:`~optuna.storages.journal.JournalFileSymlinkLock`.

    Args:
        filepath:
            The path of the file whose race condition must be protected.
    """

    def __init__(self, filepath: str) -> None:
        self._lock_file = filepath + LOCK_FILE_SUFFIX

    def acquire(self) -> bool:
        """Acquire a lock in a blocking way by creating a lock file.

        Returns:
            :obj:`True` if it succeeded in creating a ``self._lock_file``.

        """
        sleep_secs = 0.001
        while True:
            try:
                open_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
                os.close(os.open(self._lock_file, open_flags))
                return True
            except OSError as err:
                if err.errno == errno.EEXIST:
                    time.sleep(sleep_secs)
                    sleep_secs = min(sleep_secs * 2, 1)
                    continue
                raise err
            except BaseException:
                self.release()
                raise

    def release(self) -> None:
        """Release a lock by removing the created file."""

        lock_rename_file = self._lock_file + str(uuid.uuid4()) + RENAME_FILE_SUFFIX
        try:
            os.rename(self._lock_file, lock_rename_file)
            os.unlink(lock_rename_file)
        except OSError:
            raise RuntimeError("Error: did not possess lock")
        except BaseException:
            os.unlink(lock_rename_file)
            raise


@contextmanager
def get_lock_file(lock_obj: BaseJournalFileLock) -> Iterator[None]:
    lock_obj.acquire()
    try:
        yield
    finally:
        lock_obj.release()


@deprecated_class(
    "4.0.0", "6.0.0", text="Use :class:`~optuna.storages.journal.JournalFileBackend` instead."
)
class JournalFileStorage(JournalFileBackend):
    pass


@deprecated_class(
    deprecated_version="4.0.0",
    removed_version="6.0.0",
    name="The import path :class:`~optuna.storages.JournalFileOpenLock`",
    text="Use :class:`~optuna.storages.journal.JournalFileOpenLock` instead.",
)
class DeprecatedJournalFileOpenLock(JournalFileOpenLock):
    pass


@deprecated_class(
    deprecated_version="4.0.0",
    removed_version="6.0.0",
    name="The import path :class:`~optuna.storages.JournalFileSymlinkLock`",
    text="Use :class:`~optuna.storages.journal.JournalFileSymlinkLock` instead.",
)
class DeprecatedJournalFileSymlinkLock(JournalFileSymlinkLock):
    pass
