"""Use the lm15 types and providers bundled with DSPy.

Import public names here rather than from DSPy's private vendor directory.
These are the original lm15 objects, not copies or subclasses. `dspy.LM`
accepts explicit Request objects and uses them in its native engine path.

Examples:
    >>> from dspy.lm15 import Message, Request
    >>> request = Request(model="example", messages=(Message.user("Hello"),))
    >>> request.model
    'example'
"""

from dspy._vendor.lm15 import *
from dspy._vendor.lm15 import __all__ as __all__
