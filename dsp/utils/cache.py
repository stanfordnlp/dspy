import json
import redis
import hashlib
from functools import wraps
from datetime import datetime, timezone



# def connect(host="localhost", port=6379, db=0):
cache = redis.Redis(host='localhost', port=6379, db=0)

def _hash(func, *args, **kwargs):
    
    func_name = func.__name__ # Joblib code doesn't do this for lambda functions (see joblib/memory.py:Memory:line 684). TODO
    id = func_name + str(args) + str(kwargs)
    func_hash = hashlib.sha256(id.encode()).hexdigest()

    ts_str = str(datetime.now(timezone.utc).timestamp())

    # TODO: check if we're assuming no duplicate function names. 
    # Otherwise should probably include a hash of the function code
    return func_hash, ts_str


def cache_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        key, ts_str = _hash(func, *args, **kwargs)

        if cache.exists(key):
            # Get the first element - which should be the most recent entry because of lpush - then deserilize
            latest = json.loads(cache.lindex(key, 0)) 
            # TODO: Should update the timestamp here?
            return latest["result"]
        
        else:
            res = func(*args, **kwargs)
            hash, ts = _hash(func, *args, **kwargs)
            cache.lpush(hash, json.dumps({"timestamp": ts, "result": res})) # TODO: should avoid serializing? Alternative?
            return res
        
    return wrapper
        
@cache_wrapper
def add3numbers(a,b,c):
    return a+b+c
        
        
if __name__ == "__main__":
    print("--1--")
    res = add3numbers(1,2,3)
    print("--2--")
    res2 = add3numbers(1,2,3)
    print("--3--")
    res3 = add3numbers(61,2,0)
