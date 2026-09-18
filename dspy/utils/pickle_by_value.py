"""Serialize modules by value for the duration of one save.

cloudpickle's ``register_pickle_by_value`` is process-wide state keyed by
module name and lasts until unregistered. DSPy's ``save(modules_to_serialize=...)``
registered and never unregistered, so one save changed how every later
pickle in the process treated any module of that name — including a
program saved without ``modules_to_serialize`` that then loaded where the
module was absent, instead of failing (surfaced by two tests sharing a
module name under xdist).

The registration now lasts exactly as long as the saves that need it.
Because the registry is shared by every thread, ownership is counted here
under one lock: a module is registered on the first active save that asks
for it and unregistered when the last of them finishes, so a save that
overlaps another never loses the registration mid-dump. A module the
caller registered themselves before any save is theirs and is never
unregistered.
"""

import threading
from contextlib import contextmanager

import cloudpickle

_lock = threading.Lock()
_active: dict[str, int] = {}  # module name -> saves currently relying on DSPy's registration


@contextmanager
def serialize_by_value(modules):
    modules = list(modules or [])
    owned: list = []
    with _lock:
        registry = set(cloudpickle.list_registry_pickle_by_value())
        for module in modules:
            name = module.__name__
            if _active.get(name, 0) == 0 and name in registry:
                continue  # registered by the caller, not by a save: not ours to count or remove
            if _active.get(name, 0) == 0:
                cloudpickle.register_pickle_by_value(module)
            _active[name] = _active.get(name, 0) + 1
            owned.append(module)
    try:
        yield
    finally:
        with _lock:
            for module in owned:
                name = module.__name__
                remaining = _active.get(name, 0) - 1
                if remaining > 0:
                    _active[name] = remaining
                    continue
                _active.pop(name, None)
                try:
                    cloudpickle.unregister_pickle_by_value(module)
                except ValueError:
                    pass  # the caller unregistered it during the save
