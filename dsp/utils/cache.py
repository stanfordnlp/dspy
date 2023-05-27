import json
import time
import redis
import hashlib
import threading
from functools import wraps
from datetime import datetime, timezone

# TODO: Create env variables
POLL_INTERVAL = 0.005
MAX_POLL_TIME = 0.200


cache = redis.Redis(host='localhost', port=6379, db=0)

def _hash(func, *args, **kwargs):
    
    func_name = func.__name__ # Joblib code doesn't get func name for lambda functions (see joblib/memory.py:Memory:line 684). TODO - check if should exclude
    id = func_name + str(args) + str(kwargs)
    func_hash = hashlib.sha256(id.encode()).hexdigest()

    return func_hash


def _sort_redis_values(redis_values):
    
    # boolean for status gives highest priority to COMPLETE, followed by timestamp ordering
    # TODO: Better way than sorting each time?
    sorted_values = sorted(redis_values, key=lambda x: (x["status"] != "COMPLETE", float(x["timestamp"])))
    return sorted_values


def _find_complete_entry(redis_values):
    for entry in redis_values:
        if entry["status"] == "COMPLETE":
            return entry
    return None

def _compute(key, func, *args, **kwargs):
    cache_version = kwargs["cache_version"]
    # add entry to cache with status = RUNNING
    ts_str = str(datetime.now(timezone.utc).timestamp())
    cache.lpush(key, json.dumps({"timestamp": ts_str, "status": "RUNNING", "cache_version": cache_version, "args": args, "kwargs": kwargs}))

    # delete cache_version from kwargs before calling the underlying function
    del kwargs["cache_version"]
    # add an entry to the cache with status = RUNNING
    try:
        res = func(*args, **kwargs)
    except Exception as e:
        # don't cache, so remove RUNNING entry
        entries = cache.lrange(key, 0, -1)
        entries = [json.loads(element.decode()) for element in entries]
        for idx, entry in enumerate(entries):
            # match the cache version
            if entry["cache_version"] == cache_version and entry["timestamp"] == ts_str and entry["status"] == "RUNNING":
                break
            entries.pop(idx)
            cache.delete(key)
            cache.lpush(key, *entries)
        raise e # TODO raise or return?
    else:
        ts_complete_str = str(datetime.now(timezone.utc).timestamp())
        # update cache entry with status = COMPLETE, timestamp and result
        entries = cache.lrange(key, 0, -1)
        entries = [json.loads(element.decode()) for element in entries]
        for idx, entry in enumerate(entries):
            # match the cache version
            if entry["cache_version"] == cache_version and entry["timestamp"] == ts_str: # TODO: check if this is the right way to match. Can we have multiple entries for the same version with different timestamps?
                curr = entry
                break
        entries.pop(idx)
        curr["status"] = "COMPLETE"
        curr["timestamp"] = ts_complete_str
        curr["result"] = res

        entries.append(curr)
        entries = [json.dumps(element).encode() for element in entries]
        cache.delete(key)
        cache.lpush(key, *entries)
        return res


def cache_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        thread_name = threading.current_thread().name
        # see if cache_version in kwargs

        cache_version = kwargs.get("cache_version", False)
        if not cache_version:   
            cache_version = -1 # default cache version. Should document that cache versions start at 0
            kwargs["cache_version"] = -1

        key = _hash(func, *args, **kwargs)

        if cache.exists(key):
            # load the list of cached results
            cached_vals = cache.lrange(key, 0, -1)
            cached_vals = [json.loads(element.decode()) for element in cached_vals]
            # sort
            sorted_vals = _sort_redis_values(cached_vals)
            # CASE #1 - COMPLETED entry exists in cache - return as is
            if sorted_vals[0]["status"] == "COMPLETE":
                print(thread_name + ": CASE 1")
                return sorted_vals[0]["result"]
            else:
                # CASE #2 - Cached entries but COMPLETED entry does not exist in cache - poll
                print(thread_name + ": CASE 2")
                start_time = time.time()

                while time.time() - start_time < MAX_POLL_TIME:
                    # check if an entry has been completed
                    completed = _find_complete_entry(sorted_vals)
                    if completed:
                        return completed["result"]
                    # Wait for 5 milliseconds
                    time.sleep(POLL_INTERVAL)
                
                # CASE #3A - Cached entries but COMPLETED entry does not yet exist in cache - polling timed out
                print(thread_name + ":CASE 3A")
                return _compute(key, func, *args, **kwargs)
        
        else:
            # CASE #3B - No cached entries - run function and cache
            print(thread_name + ":CASE 3B")
            return _compute(key, func, *args, **kwargs)
        
    return wrapper

#TODO: Add lru cache
@cache_wrapper
def add3numbers(a,b,c):
    thread_name = threading.current_thread().name
    time.sleep(0.050)

    # print(thread_name + ": running function - not cached")
    return a+b+c
        

def test_function():
    
    thread1 = threading.Thread(target=add3numbers, args=(1,2,3), kwargs={"cache_version": 7}, name="Thread 1")
    thread2 = threading.Thread(target=add3numbers, args=(5,5,7), name="Thread 2")
    thread3 = threading.Thread(target=add3numbers, args=(5,5,7), name="Thread 3")
    thread4 = threading.Thread(target=add3numbers, args=(5,5,7), kwargs={"cache_version": 3}, name="Thread 4")

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()

    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()


        
if __name__ == "__main__":
    # print("--1--")
    # res = add3numbers(1,2,3)
    # print("--2--")
    # res2 = add3numbers(1,2,3)
    # print("--3--")
    # res3 = add3numbers(61,2,0)
    # print("--4--")
    # res4 = add3numbers(9,1,6,cache_version=3)
    test_function()