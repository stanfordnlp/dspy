import decorator

import dspy


def clean_up_lm_test(func):
    def wrapper(func, *args, **kwargs):
        dspy.settings.configure(lm=None, backend=None, cache=False)
        func(*args, **kwargs)
        dspy.settings.configure(lm=None, backend=None, cache=False)

    return decorator.decorator(wrapper, func)
