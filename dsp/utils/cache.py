import json
import time
import uuid
import redis
import random
import hashlib
import threading
from functools import wraps
from datetime import datetime, timezone

# TODO: Create env variables
POLL_INTERVAL = 0.005
MAX_POLL_TIME = 0.200


cache = redis.Redis(host='localhost', port=6379, db=0, charset='utf-8') # redis' default encoding is also utf-8


def _hash(func, *args, **kwargs):
    func_name = func.__name__ # Joblib code doesn't get func name for lambda functions (see joblib/memory.py:Memory:line 684). 
    # TODO - check if should add a condition to exclude for lambdas 
    id = func_name + str(args) + str(kwargs)
    func_hash = hashlib.sha256(id.encode()).hexdigest()

    return func_hash


def _compute(hash, func, worker_id, cache_branch, *args, **kwargs):
    # worker_id = kwargs.get("worker_id") # required to be there. TODO: see if we can do this without requiring the user to pass in a unique id per thread
    # cache_branch = str(kwargs.get("cache_branch", 0))

    key_started = hash + "_" + "branch" + "_" + cache_branch + "_STARTED"
    cache.sadd(key_started, worker_id)

    try:
        res = func(*args, **kwargs)
    except Exception as e:
        # don't cache on error, so remove STARTED entry
        cache.srem(key_started, worker_id)
        raise e # TODO: check if should raise the error immediately
        # TODO: see if should also remove the key (does it stick around if val is empty?) 
        # from redis explicitly (since logic depends on the STARTED key existing)
    else:
        ts_complete = datetime.now(timezone.utc).timestamp()
        # check if more recently completed jobs exist in the cache
        key_complete = hash + "_" + "branch" + "_" + cache_branch + "_COMPLETE"
        if cache.exists(key_complete):
            latest = cache.zrevrange(key_complete, 0, 0, withscores=True)[0] # get the latest
            id, ts = latest[0].decode("utf-8"), latest[1]
            if ts_complete < ts:
                thread_key = hash + "_" + "branch" + "_" + cache_branch + "_" + "worker" + "_" + id
                cache.srem(key_started, worker_id) # remove from started
                result = cache.json().get(thread_key, '.result') 
                return result
                # TODO: still need to save the results of this computation? I think not since that would never be the latest? 
                # But may be needed in advanced ts filtering scenarios?
            
        # current computation more recent
        thread_key = hash + "_" + "branch" + "_" + cache_branch + "_" + "worker" + "_" + worker_id
        cache.srem(key_started, worker_id) # remove from started
        val = {
            "status": "COMPLETE", # TODO: I think status is now redundant. 
            "result": res,
            "timestamp": ts_complete, # TODO: might not be needed here
            "fn_name": func.__name__,
            "args": args,
            "kwargs": kwargs
        }

        cache.json().set(thread_key, '$', val)
        cache.zadd(key_complete, {worker_id: ts_complete})
        return res


def _get_latest_result(key_complete, cache_branch, hash):
    latest = cache.zrevrange(key_complete, 0, 0, withscores=True)[0] # get the latest
    id = latest[0].decode("utf-8")
    
    thread_key = hash + "_" + "branch" + "_" + cache_branch + "_" + "worker" + "_" + id
    print("returning cached result")
    return cache.json().get(thread_key, '.result')


def cache_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # see if cache_version in kwargs
        worker_id = kwargs.get("worker_id") # required to be there. TODO: see if we can do this without requiring the user to pass in a unique id per thread
        cache_branch = str(kwargs.get("cache_branch", 0))
        
        # remove from kwargs so that don't get passed to the function & don't get hashed
        del kwargs["worker_id"] 
        if "cache_branch" in kwargs:
            del kwargs["cache_branch"]

        hash = _hash(func, *args, **kwargs)
        key_complete = hash + "_" + "branch" + "_" + cache_branch + "_COMPLETE"
        key_started = hash + "_" + "branch" + "_" + cache_branch + "_STARTED"

        # CASE #1 - COMPLETED entry exists in cache - return as is
        if cache.exists(key_complete):
            # >= 1 COMPLETE result for this hash + cache_branch exist in the cache
            # Return result for latest cached result
            return _get_latest_result(key_complete, cache_branch, hash)

        # CASE #2 - Cached entries but COMPLETED entry does not exist in cache - poll
        elif cache.exists(key_started):
            # No COMPLETE result, but >= 1 STARTED jobs
            start_time = time.time()

            while time.time() - start_time < MAX_POLL_TIME:
                # TODO: turn the code below into a function as repeat of CASE 1
                if cache.exists(key_complete):
                # >= 1 COMPLETE result for this hash + cache_branch exist in the cache
                # Return result for latest cached result
                    return _get_latest_result(key_complete, cache_branch, hash)
                
                time.sleep(POLL_INTERVAL)
        
        # CASE #3 - No cached entries or polling timed out - run function and cache
        return _compute(hash, func, worker_id, cache_branch, *args, **kwargs)
                                    
    return wrapper


#TODO: Add lru cache
@cache_wrapper
def add3numbers(a,b,c):
    thread_name = threading.current_thread().name
    time.sleep(random.uniform(0, 0.1))

    print(thread_name + ": running function - not cached")
    return a+b+c
        

def test_function():
    thread1 = threading.Thread(target=add3numbers, args=(1,2,3), kwargs={"cache_branch": 7, "worker_id": str(uuid.uuid4())}, name="Thread 1")
    thread2 = threading.Thread(target=add3numbers, args=(5,5,7), kwargs={"worker_id": str(uuid.uuid4())}, name="Thread 2")
    thread3 = threading.Thread(target=add3numbers, args=(5,5,7), kwargs={"worker_id": str(uuid.uuid4())}, name="Thread 3")
    thread4 = threading.Thread(target=add3numbers, args=(5,5,7), kwargs={"cache_branch": 3, "worker_id": str(uuid.uuid4())}, name="Thread 4")

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()

    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()

        
if __name__ == "__main__":
    test_function()
