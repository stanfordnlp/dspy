import dsp
import copy

from functools import wraps

def shallow_copy_example_args(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        args = [dsp.Example(arg) if isinstance(arg, dsp.Example) else arg for arg in args]
        kwargs = {key: dsp.Example(value) if isinstance(value, dsp.Example) else value for key, value in kwargs.items()}
        return func(*args, **kwargs)
    return wrapper

transformation = shallow_copy_example_args
