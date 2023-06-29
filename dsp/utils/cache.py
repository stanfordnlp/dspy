import sys
import time
import traceback
from typing import Any, Callable, Literal, Optional, cast
import uuid
import redis
import hashlib
from functools import wraps
from datetime import datetime, timezone

# TODO: Create env variables
POLL_INTERVAL = 0.005
MAX_POLL_TIME = 0.200

try:
    cache = redis.Redis(host="localhost", port=6379, db=0, charset="utf-8")
    cache.ping()
except Exception:
    cache = None
    print("Could not connect to Redis. Falling back to basic caching.")


def filter_keys(
    input_dict: dict[str, Any], keys_to_ignore: list[str]
) -> dict[str, Any]:
    return {
        key: value for key, value in input_dict.items() if key not in keys_to_ignore
    }


def get_redis_key(
    function_hash: str,
    cache_branch: str,
    key_type: Literal["COMPLETE", "STARTED", "FAILED"],
) -> str:
    return function_hash + "_" + "branch" + "_" + cache_branch + "_" + key_type


def check_if_exists_in_cache(
    function_hash: str,
    cache_branch: str,
    range_start_timestamp: float,
    range_end_timestamp: float,
    return_status: bool = False,
) -> tuple[bool, Optional[str], Optional[str]]:
    """Returns a tuple of (cache_exists, worker_id, operation_timestamp)"""

    # GET all existing caches for this cache key and filter by timestamp
    states = ["COMPLETE", "STARTED", "FAILED"]
    state_results: dict[str, list[tuple[bytes, float]]] = {}
    for state in states:
        key = get_redis_key(function_hash, cache_branch, state)
        state_results[state] = cache.zrange(key, 0, -1, withscores=True)
        for idx, (worker_id, operation_timestamp) in enumerate(state_results[state]):
            # retrieved cache should obey the query filter timestamp range
            if range_start_timestamp <= operation_timestamp <= range_end_timestamp:
                pass
            # If the timestamp is not in the range, we do not return it
            else:
                del state_results[state][idx]
    cache_exists = any(len(items) > 0 for items in state_results.values())
    if not cache_exists:
        return (False, None, None)

    if not return_status:
        return (True, None, None)

    # If we have more than one with different states, we need to decide which to return
    flatten_list = [
        (sublist_state, worker_id, operation_timestamp)
        for sublist_state, sublist in state_results.items()
        for (worker_id, operation_timestamp) in sublist
    ]
    # first, we sort by timestamp in asc order
    # fetching earliest completed operations helps to keep consistency
    flatten_list_sorted_by_timestamp = sorted(
        flatten_list, key=lambda x: x[2], reverse=False
    )
    earliest_valid_item_to_return: Optional[tuple[str, str]] = None
    for (
        sublist_state,
        worker_id,
        operation_timestamp,
    ) in flatten_list_sorted_by_timestamp:
        # If within the timerange, we have a completed operation, we return it
        # priority should be 'completed' > 'started' > 'failed'
        if sublist_state == "COMPLETE":
            return (True, sublist_state, worker_id.decode("utf-8"))

        # we likely would have only one started operation, but we return the latest even if we have more than one
        if sublist_state == "STARTED":
            earliest_valid_item_to_return = (
                sublist_state,
                worker_id.decode("utf-8"),
            )

        # if we have neither complete nor started, we update the latest but keep looking for a complete or started one
        if earliest_valid_item_to_return is None and sublist_state == "FAILED":
            earliest_valid_item_to_return = (
                sublist_state,
                worker_id.decode("utf-8"),
            )

    # just to keep linter happy
    if earliest_valid_item_to_return is None:
        return (False, None, None)

    # we return the failed or started one if we have no complete ones
    return (
        True,
        earliest_valid_item_to_return[0],
        earliest_valid_item_to_return[1],
    )


def _hash(
    func: Callable[..., Any], *args: dict[str, Any], **kwargs: dict[str, Any]
) -> str:
    func_name = (
        func.__name__
    )  # Joblib code doesn't get func name for lambda functions (see joblib/memory.py:Memory:line 684).
    # TODO - check if should add a condition to exclude for lambdas
    hash_id = func_name + str(args) + str(kwargs)
    func_hash = hashlib.sha256(hash_id.encode()).hexdigest()

    return func_hash


def get_cache_data(completed_hash: str, worker_id: str) -> dict[str, Any]:
    cache_data_key = completed_hash + "_" + worker_id + ".result"
    return cast(dict[str, Any], cache.json().get(cache_data_key))


def _compute(
    function_hash: str,
    func: Callable[..., Any],
    worker_id: str,
    cache_branch: str,
    *args: dict[str, Any],
    **kwargs: dict[str, Any]
):
    """_compute assumes that the specified operation isn't present in the cache.
    Any new addition to the cache would only be added with the current time.
    """
    # worker_id = kwargs.get("worker_id") # required to be there. TODO: see if we can do this without requiring the user to pass in a unique id per thread
    # cache_branch = str(kwargs.get("cache_branch", 0))

    key_started = get_redis_key(function_hash, cache_branch, "STARTED")
    cache.sadd(key_started, worker_id)

    res: Optional[Any] = None
    err = None
    try:
        res = func(*args, **kwargs)
        return res
    except Exception as exc:
        err = "\n".join(traceback.format_exception(*sys.exc_info()))
        raise exc
    finally:
        ts_complete = datetime.now(timezone.utc).timestamp()
        key_complete = get_redis_key(function_hash, cache_branch, "COMPLETE")

        # Completed operations are removed from the STARTED set and added to the COMPLETE or FAILED set
        cache.srem(key_started, worker_id)
        key_to_store: str = ""
        status: str = ""
        if res is not None:
            # if the operation completed successfully, add to the COMPLETE set
            cache.zadd(key_complete, {worker_id: ts_complete})
            key_to_store = key_complete + "_" + worker_id
            status = "COMPLETE"
        else:
            # if the operation failed, add to the FAILED set
            key_failed = get_redis_key(function_hash, cache_branch, "FAILED")
            cache.zadd(key_failed, {worker_id: ts_complete})
            key_to_store = key_failed + "_" + worker_id
            status = "FAILED"

        # irrespective of success or failure, add the result to the cache
        val = {
            "status": status,
            "result": res or err,
            "timestamp": ts_complete,
            "fn_name": func.__name__,
            "args": args,
            "kwargs": kwargs,
        }

        cache.json().set(key_to_store + ".result", "$", val)


def cache_wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: dict[str, Any], **kwargs: dict[str, Any]) -> Any:
        """TODO: we need to probably set some TTL for older caches"""
        request_time = datetime.now().timestamp()
        # see if cache_version in kwargs
        worker_id = kwargs.get(
            "worker_id",
            str(uuid.uuid4()),
        )  # required to be there. TODO: see if we can do this without requiring the user to pass in a unique id per thread
        cache_branch = str(kwargs.get("cache_branch", 0))
        start_time: float = cast(float, kwargs.get("cache_start_timerange", 0))
        end_time: float = cast(float, kwargs.get("cache_end_timerange", float("inf")))

        # remove from kwargs so that don't get passed to the function & don't get hashed
        kwargs = filter_keys(
            kwargs,
            [
                "worker_id",
                "cache_branch",
                "cache_start_timerange",
                "cache_end_timerange",
            ],
        )

        if not cache:
            return func(*args, **kwargs)

        # get a consistent hash for the function call
        function_hash = _hash(func, *args, **kwargs)

        # get the key for the cache
        key_complete = get_redis_key(function_hash, cache_branch, "COMPLETE")
        key_started = get_redis_key(function_hash, cache_branch, "STARTED")
        key_failed = get_redis_key(function_hash, cache_branch, "FAILED")

        (
            cache_exists,
            cache_exists_operation_status,
            cache_exists_worker_id,
        ) = check_if_exists_in_cache(
            function_hash, cache_branch, start_time, end_time, return_status=True
        )

        if cache_exists and cache_exists_operation_status == "COMPLETE":
            cache_data = get_cache_data(key_complete, cache_exists_worker_id)["result"]
            return cache_data
        if (
            cache_exists
            and cache_exists_operation_status == "FAILED"
            and end_time <= request_time
        ):
            cache_data = get_cache_data(key_failed, cache_exists_worker_id)["result"]
            raise Exception(cache_data)
        if cache_exists and cache_exists_operation_status == "STARTED":
            start_time = time.time()
            while time.time() - start_time < MAX_POLL_TIME:
                (
                    cache_exists,
                    cache_exists_operation_status,
                    cache_exists_worker_id,
                ) = check_if_exists_in_cache(
                    function_hash,
                    cache_branch,
                    start_time,
                    end_time,
                    return_status=True,
                )
                if cache_exists and cache_exists_operation_status == "COMPLETE":
                    cache_data = get_cache_data(key_complete, cache_exists_worker_id)[
                        "result"
                    ]
                    return cache_data
                time.sleep(POLL_INTERVAL)

        if end_time > request_time:
            return _compute(
                function_hash, func, worker_id, cache_branch, *args, **kwargs
            )
        else:
            raise Exception(
                "No cached entries found and the specified time range is in the past. Modify the time range to run the operation again."
            )

    return wrapper
