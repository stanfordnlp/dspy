"""Helpers for importing classes recorded in serialized program state."""

import importlib


def import_class_from_path(class_path: str, *, description: str = "class") -> type:
    """Import a class from its module-qualified path, e.g. `my_pkg.my_module.MyClass`.

    The split between module path and attribute path is discovered by trying the
    longest module prefix first, so nested classes (`pkg.mod.Outer.Inner`) resolve too.

    Args:
        class_path: Module-qualified class path recorded in serialized state.
        description: Human-readable noun used in error messages, e.g. "LM class".

    Returns:
        The resolved class object.

    Raises:
        ImportError: If no module prefix of `class_path` can be imported.
        TypeError: If the path resolves to something that is not a class.
    """
    parts = class_path.split(".")
    last_error = None

    for split_index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split_index])
        try:
            obj = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or module_name.startswith(f"{exc.name}."):
                last_error = exc
                continue
            raise

        try:
            for attr in parts[split_index:]:
                obj = getattr(obj, attr)
        except AttributeError as exc:
            last_error = exc
            continue

        if not isinstance(obj, type):
            raise TypeError(f"Serialized {description} `{class_path}` did not resolve to a class.")
        return obj

    raise ImportError(f"Could not import serialized {description} `{class_path}`.") from last_error
