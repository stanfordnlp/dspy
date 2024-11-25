"""
  ,-~~-.___.
 / |  '     |
(  )         0
 \_/-, ,----'
    ====           //
   /  \-'~;    /~~~(O)
  /  __/~|   /       |
=(  _____| (_________|

A decorator for debugging DSPy programs.

This module provides the `snoopy` decorator which wraps functions to trace their execution
and capture detailed debugging information. It builds on Python's `snoop` package to provide
rich debugging output specifically tailored for DSPy programs.

The traced output includes:
- Line-by-line execution
- Variable values at each step
- Function calls and returns
- Exceptions and stack traces

Example:
    @snoopy
    def my_dspy_function():
        # Function code here
        pass

    # The function will now output detailed trace information when called
    # Access the trace via my_dspy_function.trace
    # Access the source code via my_dspy_function.source

The decorator can also take arguments to customize the tracing behavior:

    @snoopy(depth=2)  # Trace nested function calls up to depth 2
    def my_function():
        pass
"""

from dataclasses import dataclass, field
from functools import wraps, partial
from inspect import getsourcelines

from snoop import Config


@dataclass
class Trace:
    lines: list[str] = field(default_factory=list)

    def write(self, line: str):
        self.lines.append(line.rstrip())

    def __str__(self):
        return "\n".join(self.lines)


def source_code(f):
    lines, _ = getsourcelines(f)
    return "\n".join(f"{i:>5} | {line}".rstrip() for i, line in enumerate(lines, 1))


def tracer(f=None, **snoop_kwargs):
    if not f:
        return partial(tracer, **snoop_kwargs)

    @wraps(f)
    def wrapper(*args, **kwargs):
        trace = Trace()
        snoop = Config(out=trace, color=False, columns=[])
        wrapper.snoop = snoop
        wrapper.trace = trace
        return snoop.snoop(f, **snoop_kwargs)(*args, **kwargs)

    wrapper.source = source_code(f)

    return wrapper
