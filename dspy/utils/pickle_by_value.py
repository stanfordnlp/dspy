"""Serialize modules by value for the duration of one save.

cloudpickle's ``register_pickle_by_value`` is process-wide state keyed by
module name and lasts until unregistered. DSPy's ``save(modules_to_serialize=...)``
registered and never unregistered, so one save changed how every later
pickle in the process treated any module of that name — including a
program saved without ``modules_to_serialize`` that then loaded where the
module was absent, instead of failing (surfaced by two tests sharing a
module name under xdist). The registration now lasts exactly as long as the
save; a module the caller registered themselves before the save stays
registered.
"""

from contextlib import contextmanager

import cloudpickle


@contextmanager
def serialize_by_value(modules):
    modules = list(modules or [])
    already = set(cloudpickle.list_registry_pickle_by_value())
    registered = []
    try:
        for module in modules:
            if module.__name__ not in already:
                cloudpickle.register_pickle_by_value(module)
                registered.append(module)
        yield
    finally:
        for module in registered:
            try:
                cloudpickle.unregister_pickle_by_value(module)
            except ValueError:
                pass  # already unregistered by the caller during the save
