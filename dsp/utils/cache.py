import os
import sqlite3
import sys
import time
from timeit import default_timer
import traceback
from typing import Any, Callable, Optional, cast, Dict, List, Tuple
from functools import wraps
import uuid
from datetime import datetime
import hashlib
import threading
# from dsp.utils.logger import get_logger
from logger import get_logger

from collections import OrderedDict
import pickle # TODO: switch to CPickle? It's faster
import functools


logger = get_logger(logging_level=int(os.getenv("DSP_LOGGING_LEVEL", "20")))

MAX_POLL_TIME = os.getenv("DSP_CACHE_POLL_TIME") or 10
POLL_INTERVAL = os.getenv("DSP_CACHE_POLL_INTERVAL") or 0.003

# Statuses
COMPLETE = 2
PENDING = 1
FAILED = 0


def filter_keys(
    input_dict: Dict[str, Any], keys_to_ignore: List[str]
) -> Dict[str, Any]:
    return {
        key: value for key, value in input_dict.items() if key not in keys_to_ignore
    }


def _hash(
    func: Callable[..., Any], *args: Dict[str, Any], **kwargs: Dict[str, Any]
) -> str:
    func_name = func.__name__
    # TODO - check if should add a condition to exclude for lambdas

    # sort the kwargs to ensure consistent hash
    sorted_kwargs = OrderedDict(sorted(kwargs.items(), key=lambda x: x[0]))

    # Convert args and kwargs to strings
    args_str = ','.join(str(arg) for arg in args)
    kwargs_str = ','.join(f'{key}={value}' for key, value in sorted_kwargs.items())

    # Concatenate the function_name, args_str, and kwargs_str
    combined_str = f'{func_name}({args_str},{kwargs_str})'

    # Generate SHA-256 hash
    func_hash = hashlib.sha256(combined_str.encode()).hexdigest()

    return func_hash


class SQLiteCache:
    conn: sqlite3.Connection
    lock: threading.Lock

    def __init__(self):
        """Initialise a SQLite database using the environment key DSP_CACHE_SQLITE_PATH."""
        
        self.lock = threading.Lock()
        self.conn = None

    def connect(self):
        cache_file_path = os.getenv("DSP_CACHE_SQLITE_PATH") or "sqlite_cache.db"
        self.conn = sqlite3.connect(
            cache_file_path,
            check_same_thread=False,
            isolation_level=None,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        self.conn.execute('pragma journal_mode=wal')
        self.create_table_if_not_exists()

    def disconnect(self):
        self.conn.close()


    def create_table_if_not_exists(self):
        """Create a cache table if it does not exist.
        : row_idx: a unique id for each row
        : branch_idx: a unique id for each branch; alias for version
        : operation_hash: a hash of the function name, args, and kwargs
        : insert_timestamp: the time when the row was inserted. This is further updated any time the row is modified
        : timestamp: the time when the operation was started
        : status: the status of the operation (0 = FAILED, 1 = PENDING, 2 = COMPLETED)
        : payload: the payload of the operation
        : result: the result of the operation as a pickled object
        : update_timestamp: the time when the row was last updated
        """
        with self.lock:
            with self.conn:
                cursor = self.conn.cursor()
                try:
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS cache (
                            row_idx TEXT PRIMARY KEY,
                            branch_idx INTEGER,
                            operation_hash TEXT,
                            insert_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            timestamp FLOAT,
                            status INTEGER,
                            func_name TEXT,
                            args BLOB,
                            kwargs BLOB,
                            result BLOB
                        )
                        """
                    )
                    # cursor.execute(
                    #     """
                    #     CREATE TRIGGER IF NOT EXISTS update_timestamp
                    #     AFTER UPDATE on cache
                    #     FOR EACH ROW
                    #     BEGIN
                    #         UPDATE cache 
                    #         SET insert_timestamp = CURRENT_TIMESTAMP 
                    #         WHERE row_idx = OLD.row_idx;
                    #     END; 
                    #     """
                    # )
                    self.conn.commit()
                except Exception as e:
                    logger.warn(f"Could not create cache table with exception: {e}")
                    raise e
                finally:
                    cursor.close()
            

    def insert_new_operation(
        self, operation_hash: str, branch_idx: int, timestamp: float, status: int, func: Callable[..., Any], *args: Dict[str, Any], **kwargs: Dict[str, Any]
    ) -> str:
        """Insert a row into the table with the status "PENDING"."""
        
        row_idx = str(uuid.uuid4())
        with self.conn:
            cursor = self.conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO cache (row_idx, branch_idx, operation_hash, timestamp, status, func_name, args, kwargs)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (row_idx, branch_idx, operation_hash, timestamp, status, func.__name__, pickle.dumps(args), pickle.dumps(kwargs)),
                )
                self.conn.commit()
            finally:
                cursor.close()
        return row_idx

    def update_operation_status(
        self,
        operation_hash: str,
        branch_idx: int,
        result: Any,
        start_time: float,
        end_time: float,
        row_idx: Optional[str] = None,
        status: int = COMPLETE,
    ):
        """update an existing operation record with new status and result fields."""
        
        result = pickle.dumps(result)
        
        if row_idx is None:
            sql_query = """
            UPDATE cache
            SET status = ?, result = ?
            WHERE operation_hash = ? AND branch_idx = ? AND status = ? AND timestamp >= ?"""
            sql_inputs = (status, result, operation_hash, branch_idx, PENDING, start_time)
            if end_time != float("inf"):
                sql_query += " AND timestamp <= ?"
        else:
            sql_query = """
            UPDATE cache
            SET status = ?, result = ?
            WHERE row_idx = ?"""
            sql_inputs = (status, result, row_idx)
        with self.conn:
            cursor = self.conn.cursor()
            try:
                cursor.execute(
                    sql_query,
                    sql_inputs,
                )
                self.conn.commit()
            finally:
                cursor.close()

    def check_if_record_exists_with_status(
        self,
        operation_hash: str,
        branch_idx: int,
        start_time: float,
        end_time: float,
        status: List[int],
        row_idx: int = None
    ) -> Tuple[bool, float]:
        """Check if a row with the specific status and timerange exists for the given (operation_hash, branch_idx, status)."""

        sql_query = """
        SELECT COUNT(*), MAX(timestamp) FROM cache
        WHERE operation_hash = ? AND branch_idx = ? AND status IN ({})
        """.format(",".join(["?"] * len(status)))

        sql_inputs = [operation_hash, branch_idx, *status]  # Unpack status list with *

        if row_idx is not None:
            sql_query += " AND row_idx != ?"
            sql_inputs.append(row_idx)

        if end_time != float("inf"):
            sql_query += " AND timestamp >= ? AND timestamp <= ?"
            sql_inputs.extend([start_time, end_time])
        else:
            sql_query += " AND timestamp >= ?"
            sql_inputs.append(start_time)

        with self.conn:
            cursor = self.conn.cursor()
            try:
                cursor.execute(sql_query, sql_inputs)
                row = cursor.fetchone()
                found = row[0] > 0
                timestamp = row[1] if found else None
                return found, timestamp
            finally:
                cursor.close()

    # TODO: do we want the earliest record or the latest?
    def retrieve_earliest_record(
        self,
        function_hash: str,
        cache_branch: int,
        range_start_timestamp: float,
        range_end_timestamp: float,
        retrieve_status: Optional[List[int]] = None,
        return_record_result: bool = False,
    ):
        """Retrieve the earliest record and use the status priority:
        "COMPLETE" > "PENDING" > "FAILED".
        Filter by the range_start_timestamp and range_end_timestamp.
        Returns a tuple of (cache_exists: bool, insertion_timestamp: float, status: str, result: Any)
        """
        sql_query = """
        SELECT row_idx, insert_timestamp, status, result FROM cache
        WHERE operation_hash LIKE ? AND branch_idx = ? AND timestamp >= ?
        """
        sql_inputs = (function_hash, cache_branch, range_start_timestamp)
        if range_end_timestamp != float("inf"):
            sql_query += " AND timestamp <= ?"
            sql_inputs += (range_end_timestamp,)

        if retrieve_status is not None and isinstance(retrieve_status, list):
            # Concatenate the status values with 'OR' condition to check if status is one of the values in the list
            sql_query += " AND status IN ({})".format(', '.join(['?']*len(retrieve_status)))
            sql_inputs += tuple(retrieve_status)

        sql_query += " ORDER BY status DESC, insert_timestamp ASC LIMIT 1"

        with self.conn:
            cursor = self.conn.cursor()
            try:
                cursor.execute(
                    sql_query,
                    sql_inputs,
                )
                row = cursor.fetchone()
                if row is None:
                    return False, None, None, None
                if return_record_result:
                    logger.debug(f"returning record: {row[0]}")
                    return True, row[1], row[2], pickle.loads(row[3])
                return True, row[1], row[2], None
            finally:
                cursor.close()

    def poll_for_complete(
            self,
            function_hash: str,
            cache_branch: int,
            start_time: float,
            end_time: float,
            row_idx: int = None
    ):
        """
        Returns (bool, result) where the boolean indicates if a COMPLETE transaction was found. 
        """
        # check if the cache is pending
        exists, insert_ts = self.check_if_record_exists_with_status(
            function_hash, cache_branch, start_time, end_time, [PENDING], row_idx
        )
        # if it is, also make sure the PENDING entry is within the MAX_POLL_TIME (so we don't wait for indefinitely PENDING entries)
        logger.debug(f"POLL - insert_ts: {insert_ts}")
        logger.debug(f"POLL - now: {datetime.now().timestamp()}")
        if exists and insert_ts >= datetime.now().timestamp() - MAX_POLL_TIME:

            logger.debug("Operation is pending in the cache. Polling for result.")
            polling_start_time = default_timer()
            while default_timer() - polling_start_time < MAX_POLL_TIME:
                logger.debug(f"MAX_POLL_TIME hasn't elapsed: {default_timer() - polling_start_time}")
                if self.check_if_record_exists_with_status(
                    function_hash, cache_branch, start_time, end_time, [2]
                ):
                    logger.debug("Result found after polling. Retrieving from cache.")
                    cache_exists, insertion_timestamp, status, result = self.retrieve_earliest_record(
                        function_hash,
                        cache_branch,
                        start_time,
                        end_time,
                        retrieve_status=[COMPLETE],
                        return_record_result=True,
                    )
                    return True, result
                time.sleep(POLL_INTERVAL)

        # if doesn't exist or didn't find a COMPLETE entry after polling, return False
        return False, None
        

def sqlite_cache_wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
    """The cache wrapper for the function. This is the function that is called when the user calls the function."""

    @functools.wraps(func)
    def wrapper(*args: Dict[str, Any], **kwargs: Dict[str, Any]) -> Any:
        """The wrapper function uses the function name, args and kwargs to compute the operation_hash
        cache_branch, cache_start_timerange, cache_end_timerange are used for versioning
        It accepts 2 possible flags:
            - smart_concurrency - if True, then a PENDING entry is added to the cache while a computation is in progress. If False, entries only reflect after completion
            - ignore_exceptions - if True, if an Exception is thrown, it is not cached. If False, Exceptions cached as results, and are raised if returned.
        1. Checks if the operation_hash exists in the cache within the timerange. If it does, it returns the result.
        2. If it does not exists within the timerange, it throws an exception.
        3. If it does exists but the status is "FAILED", it throws the same exception as the reason why it failed.
        4. If it does exists but the status is "PENDING", it polls the cache for MAX_POLL_TIME seconds. If it does not get a result within that time, it throws an exception.
        5. If it does exists but the status is "COMPLETED", it returns the result.
        6. If the cache does not exists within the timerange and the end_timerange is the future, it recomputes and inserts the operation_hash with the status "PENDING" and returns the result.
        """
        cache_client = SQLiteCache()
        cache_client.connect()

        try:

            request_time = datetime.now().timestamp()
            cache_branch = cast(int, kwargs.get("cache_branch", 0))
            start_time: float = cast(float, kwargs.get("experiment_start_timestamp", 0))
            end_time: float = cast(
                float, kwargs.get("experiment_end_timestamp", float("inf"))
            )
            timerange_in_iso: str = f"{datetime.fromtimestamp(start_time).isoformat()} and {datetime.fromtimestamp(end_time).isoformat() if end_time != float('inf') else 'future'}"

            # flags
            smart_concurrency = kwargs.get("smart_concurrency", True)
            ignore_exceptions = kwargs.get("ignore_exceptions", False)

            # remove from kwargs so that don't get passed to the function & don't get hashed
            kwargs = filter_keys(
                kwargs,
                [
                    "cache_branch",
                    "experiment_start_timestamp",
                    "experiment_end_timestamp",
                    "smart_concurrency",
                    "ignore_exceptions"
                ],
            )

            # get a consistent hash for the function call
            function_hash = _hash(func, *args, **kwargs)

            if not(ignore_exceptions):
                end_statuses = [COMPLETE, FAILED]
            else:
                end_statuses = [COMPLETE]

            # check if there is a cached result
            exists, _ = cache_client.check_if_record_exists_with_status(
                function_hash, cache_branch, start_time, end_time, end_statuses
            )
            if exists:
                logger.debug(
                    f"Cached experiment result found between {timerange_in_iso}. Retrieving result from cache."
                )
                cache_exists, insertion_timestamp, status, result = cache_client.retrieve_earliest_record(
                    function_hash,
                    cache_branch,
                    start_time,
                    end_time,
                    retrieve_status=end_statuses,
                    return_record_result=True,
                )
                # if COMPLETE, return result
                if status == COMPLETE:
                    return result
                # if end_time is in the past, if there's an exception cached, just return it
                # automatically won't get here if ignore_exceptions is True since don't search for FAILED entries
                elif status == FAILED and end_time < request_time:
                    logger.debug(
                            f"Failed operation found in the cache for experiment timerange between {timerange_in_iso}. Raising the same exception."
                    )
                    raise Exception(result)

            # Didn't find a COMPLETE cached result (or a FAILED cache result in the case that end_time < request_time)
            # So see if there are any pending transactions that COMPLETE within MAX_POLL_TIME
            logger.debug("Starting to poll")
            is_complete, res = cache_client.poll_for_complete(function_hash, cache_branch, start_time, end_time)
            if is_complete:
                return res

            # Didn't find a COMPLETE transaction after waiting. If the end_time is in the past, we don't compute, and raise an Exception
            if end_time < request_time:
                raise Exception(f"Oops. Cache does not exist for the given experiment timerange of between {timerange_in_iso}.")
             
            # end_time is in the future, and we did not find a cached result, so compute it and cache it.
            # insert the operation as pending if smart_concurrency = True
            if smart_concurrency:
                row_idx = cache_client.insert_new_operation(
                    function_hash, cache_branch, datetime.now().timestamp(), PENDING, func, *args, **kwargs # TODO: check if should use request_time or current timestamp.
                )
        
            # Poll once more to see if another thread/process might have completed, before computing. Use row_idx to ensure don't wait for own PENDING entry 
            is_complete2, res2 = cache_client.poll_for_complete(function_hash, cache_branch, start_time, end_time, row_idx=row_idx)
            if is_complete2:
                # TODO: In this case we have an infinite PENDING entry - should we clean it up or should we still compute the result? Or just let it stay PENDING?
                return res2

            logger.debug(
                f"Could not find a succesful experiment result in the cache for timerange between {timerange_in_iso}. Computing!"
            )
            try:    
                result = func(*args, **kwargs)
            except Exception as exc:
                # Only cache the exception if the flag is set to False
                if not(ignore_exceptions) and smart_concurrency:
                    result = "\n".join(traceback.format_exception(*sys.exc_info()))
                    # update the cache as completed
                    cache_client.update_operation_status(
                        function_hash,
                        cache_branch,
                        result,
                        start_time,
                        end_time,
                        row_idx=row_idx,
                        status=FAILED,
                    )
                elif not(ignore_exceptions) and not(smart_concurrency):
                    cache_client.insert_new_operation(
                        function_hash,
                        cache_branch,
                        datetime.now().timestamp(), 
                        FAILED, 
                        func, 
                        *args, 
                        **kwargs
                    )
                # TODO: Ideally, if ignore_exceptions = True, we should delete the PENDING entry if we get an Exception.
                raise exc
            
            # update/insert into the cache as completed
            if smart_concurrency:
                cache_client.update_operation_status(
                    function_hash,
                    cache_branch,
                    result,
                    start_time,
                    end_time,
                    row_idx=row_idx,
                    status=COMPLETE,
                )
            else:
                cache_client.insert_new_operation(
                    function_hash,
                    cache_branch,
                    datetime.now().timestamp(), 
                    COMPLETE, 
                    func, 
                    *args, 
                    **kwargs
                )

            # redundant reads to handle concurrency issues. Latest cached COMPLETE entry returned, which may or may not be the one just computed.
            cache_exists, insertion_timestamp, status, result = cache_client.retrieve_earliest_record(
                function_hash,
                cache_branch,
                start_time,
                end_time,
                retrieve_status=COMPLETE,
                return_record_result=True,
            )
            return result
        
        finally:
            cache_client.disconnect()

    return wrapper

###################################### Temp tests  - to delete ##################################################

# threaded tests
import threading 

@sqlite_cache_wrapper
def test_function_cached2(**kwargs):
    kwargs_ = ','.join(f'{key}={value}' for key, value in kwargs.items())
    time.sleep(5)
    return hash(kwargs_)

def run_function_in_thread(kwargs):
    result = test_function_cached2(**kwargs)
    return result 

if __name__ == "__main__":

    start_time = time.time()
    # thread_results = []
    threads = []

    kwargs = [{"kwarg1": 10, "kwarg2": 2, "kwarg3": 5}, {"kwarg1": 10, "kwarg2": 2, "kwarg3": 5}, {"kwarg1": 10, "kwarg2": 2, "kwarg3": 5}, {"kwarg1": 10, "kwarg2": 2, "kwarg3": 5}]

    # Create a thread for each call to test_function_cached
    for kwarg in kwargs:
        thread = threading.Thread(target=run_function_in_thread, args=(kwarg,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    end_time = time.time()

    print(f'Cached function writes (threaded) took {end_time - start_time} seconds')

    # test read performance + save outputs
    outputs = []
    start_time2 = time.time()
    threads = []

    # Create a thread for each call to test_function_cached
    for kwarg in kwargs:
        thread = threading.Thread(target=run_function_in_thread, args=(kwarg,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    end_time2 = time.time()
    print(f'Cached function reads (threaded) took {end_time2 - start_time2} seconds')