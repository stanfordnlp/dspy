"""
  ,-~~-.___.
 / |  '     \
(  )         0
 \_/-, ,----'
    ====           //
   /  \-'~;    /~~~(O)
  /  __/~|   /       |
=(  _____| (_________|
"""

from dataclasses import dataclass, field
from functools import wraps, partial
from inspect import getsourcelines

from snoop import Config as SnoopConfig


@dataclass
class SnoopFile:
    lines: list[str] = field(default_factory=list)

    def write(self, line: str):
        self.lines.append(line.rstrip())

    def __str__(self):
        return "\n".join(self.lines)


def source_code(f):
    lines, _ = getsourcelines(f)
    return "\n".join(f"{i:>5} | {line}".rstrip() for i, line in enumerate(lines, 1))


def snoopy(f=None, **snoop_kwargs):
    if not f:
        return partial(snoopy, **snoop_kwargs)

    @wraps(f)
    def wrapper(*args, **kwargs):
        trace = SnoopFile()
        snoopy = SnoopConfig(out=trace, color=False, columns=[])
        wrapper.snoopy = snoopy
        wrapper.trace = trace
        return snoopy.snoop(f, **snoop_kwargs)(*args, **kwargs)

    wrapper.source = source_code(f)

    return wrapper
