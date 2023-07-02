import json
import os
import sqlite3
import sys
import time
from timeit import default_timer
import traceback
from typing import Any, Callable, Optional, cast
from functools import wraps
import uuid
from datetime import datetime
import hashlib
import threading
from dsp.utils.logger import get_logger

logger = get_logger(logging_level=int(os.getenv("DSP_LOGGING_LEVEL", "20")))

MAX_POLL_TIME = 10
POLL_INTERVAL = 0.005


def filter_keys(
    input_dict: dict[str, Any], keys_to_ignore: list[str]
) -> dict[str, Any]:
    return {
        key: value for key, value in input_dict.items() if key not in keys_to_ignore
    }


def _hash(
    func: Callable[..., Any], *args: dict[str, Any], **kwargs: dict[str, Any]
) -> str:
    func_name = func.__name__
    # TODO - check if should add a condition to exclude for lambdas
    hash_id = func_name + str(args) + str(kwargs)
    func_hash = hashlib.sha256(hash_id.encode()).hexdigest()

    return func_hash


class SQLiteCache:
    db_client: sqlite3.Connection
    lock: threading.Lock

    def __init__(self):
        """Initialise a SQLite database using the environment key DSP_CACHE_SQLITE_PATH."""
        self.lock = threading.Lock()
        cache_file_path = os.getenv("DSP_CACHE_SQLITE_PATH") or "sqlite_cache.db"
        self.db_client = sqlite3.connect(
            cache_file_path,
            check_same_thread=False,
            isolation_level=None,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        self.create_table_if_not_exists()

    def create_table_if_not_exists(self):
        """Create a cache table if it does not exist.
        : row_idx: a unique id for each row
        : branch_idx: a unique id for each branch; alias for version
        : operation_hash: a hash of the function name, args, and kwargs
        : insert_timestamp: the time when the row was inserted
        : timestamp: the time when the operation was started
        : status: the status of the operation (0 = FAILED, 1 = PENDING, 2 = COMPLETED)
        : payload: the payload of the operation
        : result: the result of the operation in JSON format
        """
        with self.lock:
            cursor = self.db_client.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    row_idx TEXT PRIMARY KEY,
                    branch_idx INTEGER,
                    operation_hash TEXT,
                    insert_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    timestamp FLOAT,
                    status INTEGER,
                    payload TEXT,
                    result TEXT
                )
                """
            )

    def insert_operation_started(
        self, operation_hash: str, branch_idx: int, timestamp: float
    ) -> str:
        """Insert a row into the table with the status "PENDING"."""
        with self.lock:
            row_idx = str(uuid.uuid4())
            cursor = self.db_client.cursor()
            cursor.execute(
                """
                INSERT INTO cache (row_idx, branch_idx, operation_hash, timestamp, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (row_idx, branch_idx, operation_hash, timestamp, 1),
            )
            return row_idx

    def update_operation_status(
        self,
        operation_hash: str,
        branch_idx: int,
        result: Any,
        start_time: float,
        end_time: float,
        row_idx: Optional[str] = None,
        status: int = 2,
    ):
        """update an existing operation record with new status and result fields."""
        with self.lock:
            result = json.dumps(result) if isinstance(result, dict) else result
            cursor = self.db_client.cursor()
            if row_idx is None:
                sql_query = """
                UPDATE cache
                SET status = ?, result = ?
                WHERE operation_hash = ? AND branch_idx = ? AND status = ? AND timestamp >= ?"""
                sql_inputs = (status, result, operation_hash, branch_idx, 1, start_time)
                if end_time != float("inf"):
                    sql_query += " AND timestamp <= ?"
            else:
                sql_query = """
                UPDATE cache
                SET status = ?, result = ?
                WHERE row_idx = ?"""
                sql_inputs = (status, result, row_idx)
            cursor.execute(
                sql_query,
                sql_inputs,
            )

    def check_if_record_exists_with_status(
        self,
        operation_hash: str,
        branch_idx: int,
        start_time: float,
        end_time: float,
        status: int,
    ) -> bool:
        """Check if a row with the specific status and timerange exists for the given (operation_hash, branch_idx, status)."""
        with self.lock:
            cursor = self.db_client.cursor()
            sql_query = """
            SELECT COUNT(*) FROM cache
            WHERE operation_hash = ? AND branch_idx = ? AND status = ? AND timestamp >= ?
            """
            sql_inputs = (operation_hash, branch_idx, status, start_time)
            if end_time != float("inf"):
                sql_query += " AND timestamp <= ?"
                sql_inputs += (end_time,)

            sql_query += " ORDER BY timestamp ASC LIMIT 1"

            cursor.execute(
                sql_query,
                sql_inputs,
            )
            return cursor.fetchone()[0] > 0

    def retrieve_earliest_record(
        self,
        function_hash: str,
        cache_branch: int,
        range_start_timestamp: float,
        range_end_timestamp: float,
        retrieve_status: Optional[int] = None,
        return_record_result: bool = False,
    ):
        """Retrieve the earliest record and use the status priority:
        "FINISHED" > "PENDING" > "FAILED".
        Filter by the range_start_timestamp and range_end_timestamp.
        Returns a tuple of (cache_exists: bool, insertion_timestamp: float)
        """
        with self.lock:
            cursor = self.db_client.cursor()
            sql_query = """
            SELECT row_idx, insert_timestamp, status, result FROM cache
            WHERE operation_hash LIKE ? AND branch_idx = ? AND timestamp >= ?
            """
            sql_inputs = (function_hash, cache_branch, range_start_timestamp)
            if range_end_timestamp != float("inf"):
                sql_query += " AND timestamp <= ?"
                sql_inputs += (range_end_timestamp,)
            if retrieve_status is not None:
                sql_query += " AND status = ?"
                sql_inputs += (retrieve_status,)
            sql_query += " ORDER BY status DESC, insert_timestamp ASC LIMIT 1"

            cursor.execute(
                sql_query,
                sql_inputs,
            )
            row = cursor.fetchone()
            if row is None:
                return False, None
            if return_record_result:
                logger.debug(f"returning record: {row[0]}")
                return True, json.loads(row[3]) if retrieve_status == 2 else row[3]
            return True, None

    def get_cache_record(
        self, operation_hash: str, branch_idx: int, start_time: float, end_time: float
    ):
        """Retrieve the cache record for the given operation_hash."""
        with self.lock:
            cursor = self.db_client.cursor()
            sql_query = """
            SELECT result FROM cache
            WHERE operation_hash = ? AND branch_idx = ? AND timestamp >= ?
            """
            sql_inputs = (operation_hash, branch_idx, start_time)
            if end_time != float("inf"):
                sql_query += " AND timestamp <= ?"
                sql_inputs += (end_time,)

            cursor.execute(sql_query, sql_inputs)
            row = cursor.fetchone()
            if row is None:
                return None
            else:
                return json.loads(row[0])


def cache_wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
    """The cache wrapper for the function. This is the function that is called when the user calls the function."""

    @wraps(func)
    def wrapper(*args: dict[str, Any], **kwargs: dict[str, Any]) -> Any:
        """The wrapper function uses the arguments worker_id, cache_branch, cache_start_timerange, cache_end_timerange to compute the operation_hash.
        1. It then checks if the operation_hash exists in the cache within the timerange. If it does, it returns the result.
        2. If it does not exists within the timerange, it throws an exception.
        3. If it does exists but the status is "FAILED", it throws the same exception as the reason why it failed.
        4. If it does exists but the status is "PENDING", it polls the cache for MAX_POLL_TIME seconds. If it does not get a result within that time, it throws an exception.
        5. If it does exists but the status is "COMPLETED", it returns the result.
        6. If the cache does not exists within the timerange and the end_timerange is the future, it recomputes and inserts the operation_hash with the status "PENDING" and returns the result.
        """
        request_time = datetime.now().timestamp()
        cache_branch = cast(int, kwargs.get("cache_branch", 0))
        start_time: float = cast(float, kwargs.get("experiment_start_timestamp", 0))
        end_time: float = cast(
            float, kwargs.get("experiment_end_timestamp", float("inf"))
        )
        timerange_in_iso: str = f"{datetime.fromtimestamp(start_time).isoformat()} and {datetime.fromtimestamp(end_time).isoformat() if end_time != float('inf') else 'future'}"

        # remove from kwargs so that don't get passed to the function & don't get hashed
        kwargs = filter_keys(
            kwargs,
            [
                "worker_id",
                "cache_branch",
                "experiment_start_timestamp",
                "experiment_end_timestamp",
            ],
        )

        # get a consistent hash for the function call
        function_hash = _hash(func, *args, **kwargs)

        # check if the cache exists
        if cache_client.check_if_record_exists_with_status(
            function_hash, cache_branch, start_time, end_time, 2
        ):
            logger.debug(
                f"Cached experiment result found between {timerange_in_iso}. Retrieving result from cache."
            )
            _, result = cache_client.retrieve_earliest_record(
                function_hash,
                cache_branch,
                start_time,
                end_time,
                retrieve_status=2,
                return_record_result=True,
            )
            return result

        # check if the cache is pending
        if cache_client.check_if_record_exists_with_status(
            function_hash, cache_branch, start_time, end_time, 1
        ):
            logger.debug("Operation is pending in the cache. Polling for result.")
            polling_start_time = default_timer()
            while default_timer() - polling_start_time < MAX_POLL_TIME:
                if cache_client.check_if_record_exists_with_status(
                    function_hash, cache_branch, start_time, end_time, 2
                ):
                    logger.debug("Result found after polling. Retrieving from cache.")
                    _, result = cache_client.retrieve_earliest_record(
                        function_hash,
                        cache_branch,
                        start_time,
                        end_time,
                        retrieve_status=2,
                        return_record_result=True,
                    )
                    return result
                time.sleep(POLL_INTERVAL)
            raise Exception("Failed to retrieve result from cache after polling.")

        # check if the cache failed
        if cache_client.check_if_record_exists_with_status(
            function_hash, cache_branch, start_time, end_time, 0
        ):
            if end_time < request_time:
                _, result = cache_client.retrieve_earliest_record(
                    function_hash,
                    cache_branch,
                    start_time,
                    end_time,
                    retrieve_status=0,
                    return_record_result=True,
                )
                logger.debug(
                    f"Failed operation found in the cache for experiment timerange between {timerange_in_iso}. Raising the same exception."
                )
                raise Exception(result)

        if end_time < request_time:
            raise Exception(
                f"Cache does not exist for the given experiment timerange of between {timerange_in_iso}."
            )

        # insert the operation as pending
        row_idx = cache_client.insert_operation_started(
            function_hash, cache_branch, request_time
        )

        logger.debug(
            f"Could not find a succesful experiment result in the cache for timerange between {timerange_in_iso}. Computing!"
        )
        try:
            result = func(*args, **kwargs)
            # update the cache as completed
            cache_client.update_operation_status(
                function_hash,
                cache_branch,
                result,
                start_time,
                end_time,
                row_idx=row_idx,
                status=2,
            )
            # redundant reads to handle concurrency issues
            _, result = cache_client.retrieve_earliest_record(
                function_hash,
                cache_branch,
                start_time,
                end_time,
                retrieve_status=2,
                return_record_result=True,
            )
            return result
        except Exception as exc:
            result = "\n".join(traceback.format_exception(*sys.exc_info()))
            # update the cache as completed
            cache_client.update_operation_status(
                function_hash,
                cache_branch,
                result,
                start_time,
                end_time,
                row_idx=row_idx,
                status=0,
            )
            raise exc

    return wrapper


cache_client = SQLiteCache()
