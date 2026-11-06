"""Libraries a ``dspy.Flex`` module's code may import, and what the code optimizer is told about them."""

from __future__ import annotations

import ast
import contextlib
import enum
import functools
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import types
from typing import Any


class Library:
    """A library that a ``dspy.Flex`` module's code may import, such as your own SDK.

    The code optimizer gets an API reference generated from the module: its public functions and classes, with their
    signatures and the first paragraph of each docstring. ``docs`` adds anything the reference can't show, like how
    the library is meant to be used, examples, or pitfalls. The library's version and a fingerprint of the reference
    are saved with the optimized program, so that loading it next to another version warns.

    ```python
    import acme

    program = dspy.Flex(
        "ticket -> reply",
        libraries=[dspy.Library(acme, docs="Create one acme.Client per call and close it.")],
        interpreter_factory=dspy.LocalInterpreter,
    )
    ```

    Args:
        module: The module, or its importable name.
        docs: Notes for the optimizer, added after the generated reference.
        members: The names to document. Defaults to the module's ``__all__``, or else to its public names.
        max_chars: Upper bound on the generated reference's length.
    """

    def __init__(
        self, module: types.ModuleType | str, *, docs: str = "", members: list[str] | None = None, max_chars: int = 8000
    ) -> None:
        loaded = importlib.import_module(module) if isinstance(module, str) else module
        if not isinstance(loaded, types.ModuleType):
            raise TypeError(f"dspy.Library takes a module or a module's name, not {module!r}")
        self.name = loaded.__name__
        self.docs = docs
        self.members = list(members) if members is not None else None
        self.max_chars = max_chars
        self._reference: str | None = None

    def __repr__(self) -> str:
        return f"Library({self.name!r})"

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_reference": None}  # read again from the module where it is loaded

    def __deepcopy__(self, memo: dict) -> Library:
        copied = object.__new__(Library)  # a copy in this process: the same module, so the same reference
        copied.__dict__.update(self.__dict__)
        return copied

    @property
    def module(self) -> types.ModuleType:
        return importlib.import_module(self.name)

    @property
    def version(self) -> str | None:
        """The installed version of the distribution that provides the module, if any."""
        for distribution in _distributions().get(self.name.split(".")[0], []):
            with contextlib.suppress(metadata.PackageNotFoundError):
                return metadata.version(distribution)
        version = getattr(self.module, "__version__", None)
        return version if isinstance(version, str) else None

    def reference(self) -> str:
        """The API reference the optimizer reads."""
        if self._reference is None:
            self._reference = self._read_reference()
        return self._reference

    def _read_reference(self) -> str:
        module = self.module
        lines = [f"import {self.name}", *_comment(_doc(module))]
        names = self.members if self.members is not None else _public_names(module)
        for name in names:
            if hasattr(module, name):
                lines.extend(_describe(f"{self.name}.{name}", getattr(module, name)))
        return _truncate(lines, self.max_chars)

    @property
    def fingerprint(self) -> str:
        """A hash of the reference: code written against it may break when it changes, version or not."""
        return hashlib.sha256(self.reference().encode()).hexdigest()[:12]

    def render(self) -> str:
        """The reference and the notes, as the optimizer sees them."""
        version = f" {self.version}" if self.version else ""
        parts = [f"### {self.name}{version}", self.reference()]
        if self.docs.strip():
            parts.append(self.docs.strip())
        return "\n\n".join(parts)


@functools.cache
def _distributions() -> dict[str, list[str]]:
    return metadata.packages_distributions()


def _public_names(module: types.ModuleType) -> list[str]:
    if isinstance(listed := getattr(module, "__all__", None), (list, tuple)):
        return [name for name in listed if isinstance(name, str) and not name.startswith("__")]
    root = module.__name__.split(".")[0]
    assigned = _assigned_names(module)
    names = []
    for name, value in vars(module).items():
        if name.startswith("_") or isinstance(value, types.ModuleType):
            continue
        # What the module defines, not what it imported from elsewhere (``import json``, ``from typing import Any``).
        if callable(value):
            defined_in = getattr(value, "__module__", None)
            if isinstance(defined_in, str) and defined_in.split(".")[0] == root:
                names.append(name)
        elif assigned is None or name in assigned:
            names.append(name)
    return names


def _assigned_names(module: types.ModuleType) -> set[str] | None:
    """The names the module's own source assigns at its top level, or ``None`` without source."""
    try:
        tree = ast.parse(inspect.getsource(module))
    except (OSError, TypeError, SyntaxError):
        return None
    names = set()
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        names |= {target.id for target in targets if isinstance(target, ast.Name)}
    return names


def _describe(qualified: str, value: Any) -> list[str]:
    if inspect.isclass(value):
        lines = [f"class {qualified}{_signature(value, returns=False)}", *_comment(_doc(value), "    ")]
        if issubclass(value, enum.Enum):
            lines.append(f"    # members: {', '.join(member.name for member in value)}")
            return lines
        for name, attribute in vars(value).items():
            function = attribute.__func__ if isinstance(attribute, (staticmethod, classmethod)) else attribute
            if name.startswith("_") or not callable(function):
                continue
            lines.append(f"    def {name}{_signature(function, drop_self=not isinstance(attribute, staticmethod))}")
            lines.extend(_comment(_doc(function), "        "))
        return lines
    if callable(value):
        return [f"def {qualified}{_signature(value)}", *_comment(_doc(value), "    ")]
    try:
        text = repr(value)
    except Exception:
        return []
    return [f"{qualified} = {text if len(text) <= 80 else text[:77] + '...'}"]


def _signature(value: Any, drop_self: bool = False, returns: bool = True) -> str:
    try:
        signature = inspect.signature(value, eval_str=True)
    except Exception:
        try:
            signature = inspect.signature(value)
        except (TypeError, ValueError):
            return "(...)"
    parameters = list(signature.parameters.values())
    if drop_self and parameters and parameters[0].name in ("self", "cls"):
        signature = signature.replace(parameters=parameters[1:])
    if not returns:
        signature = signature.replace(return_annotation=inspect.Signature.empty)
    return str(signature)


def _doc(value: Any, limit: int = 800) -> str:
    """The docstring it defines itself (not one inherited from ``object``), cut at ``limit`` characters."""
    doc = (value.__doc__ if isinstance(value.__doc__, str) else "") if not inspect.ismodule(value) else value.__doc__
    doc = inspect.cleandoc(doc or "")
    return doc if len(doc) <= limit else doc[: limit - 3].rstrip() + "..."


def _comment(text: str, indent: str = "") -> list[str]:
    return [f"{indent}# {line}".rstrip() for line in text.splitlines()]


def _truncate(lines: list[str], max_chars: int) -> str:
    out, size = [], 0
    for index, line in enumerate(lines):
        if size + len(line) + 1 > max_chars:
            out.append(f"# ... {len(lines) - index} more lines. Pass members=[...] to choose what is documented.")
            break
        out.append(line)
        size += len(line) + 1
    return "\n".join(out)
