from abc import ABC, abstractmethod
from contextvars import ContextVar
import functools
import inspect
import time
import uuid

import dspy
from typing import Any, Dict


ACTIVE_CALL_ID = ContextVar("active_call_id", default=None)

class BaseCallback(ABC):

    @abstractmethod
    def on_start(
        self,
        call_id: str,
        function_name: str,
        instance: Any,
        inputs: Dict[str, Any],
        start_time: int
    ):
        pass

    @abstractmethod
    def on_success(
        self,
        call_id: str,
        outputs: Dict[str, Any],
        end_time: int
    ):
        pass

    @abstractmethod
    def on_failure(
        self,
        call_id: str,
        exception: Exception,
        end_time: int
    ):
        pass



def with_callbacks(fn):

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        callbacks = dspy.settings.get("callbacks", [])

        # if no callbacks are provided, just call the function
        if not callbacks:
            return fn(self, *args, **kwargs)

        # random call ID to connect start/end handlers if needed
        call_id = uuid.uuid4().hex

        try:
            inputs = inspect.getcallargs(fn, self, *args, **kwargs)

            # unpack "kwargs" if exists
            if "kwargs" in inputs:
                inputs.update(inputs.pop("kwargs"))

            inputs.pop("self") # Not logging self as input
        except:
            print("Failed to inspect inputs, falling back to kwargs only")
            inputs = kwargs

        start_time = time.time()
        for callback in callbacks:
            try:
                callback.on_start(
                    call_id=call_id,
                    function_name=fn.__name__,
                    instance=self,
                    inputs=inputs,
                    start_time=start_time
                )
            except Exception as e:
                print(f"Error in on_start callback {callback}: {e}")

        is_success = False
        try:
            # Check if the function is bounded method or not
            parent_call_id = ACTIVE_CALL_ID.get()
            ACTIVE_CALL_ID.set(call_id)
            results = fn(self, *args, **kwargs)
            is_success = True
            return results
        except Exception as e:
            exception = e
            raise exception
        finally:
            ACTIVE_CALL_ID.set(parent_call_id)
            end_time = time.time()
            if is_success:
                for callback in callbacks:
                    try:
                        callback.on_success(
                            call_id=call_id,
                            outputs=results,
                            end_time=end_time
                        )
                    except Exception as e:
                        print(f"Error in on_success callback {callback}: {e}")
            else:
                for callback in callbacks:
                    try:
                        callback.on_failure(
                            call_id=call_id,
                            exception=exception,
                            end_time=end_time
                        )
                    except Exception as e:
                        print(f"Error in on_failure callback {callback}: {e}")

    return wrapper

