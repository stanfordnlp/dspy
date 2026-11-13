"""Libraries that code run by an interpreter may import, such as a ``dspy.Flex`` module's own: what they reach, what
an environment resolves them to, and how their API reads.

Everything here uses only the standard library and describes the Python environment it runs in. The host uses
``reach`` to choose what a sandbox needs. ``describe``, ``verify`` and ``reference`` are run inside the interpreter
(``dspy.Flex`` does), so that the versions, file hashes and signatures are those of the code that will run: a Pyodide
sandbox has its own Python and its own builds of many packages, and a remote one may share nothing with the host.

A library is named by a module (``shopkit.pricing``) or, for one class or function of it, in the form of a packaging
entry point (``shopkit.cart:Cart``).
"""

from __future__ import annotations

import ast
import contextlib
import enum
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import inspect
import io
import json
import re
import sys
import types
from pathlib import Path
from typing import Any, Iterable

# The length of each library's API reference, in characters.
REFERENCE_CHARS = 8000


def spec_of(library: Any) -> str:
    """How code names ``library``: a module's name, or ``module:qualified.name`` for a class or function of one."""
    if isinstance(library, str):
        module, _, attribute = library.partition(":")
        named = all(part.isidentifier() for part in module.split("."))
        if not named or (":" in library and not all(part.isidentifier() for part in attribute.split("."))):
            raise ValueError(
                f"{library!r} does not name a library: use a module's name, like 'shopkit.pricing', or "
                "'module:name' for a class or function of it."
            )
        return library
    if isinstance(library, types.ModuleType):
        return library.__name__
    module, name = getattr(library, "__module__", None), getattr(library, "__qualname__", None)
    if not (inspect.isclass(library) or inspect.isroutine(library)) or not isinstance(module, str) or not isinstance(name, str):
        raise TypeError(
            f"libraries takes modules, classes and functions, or their names, not {library!r}. An object with "
            "state, like a client, stays on the host: pass what the code may call on it as tools."
        )
    found: Any = sys.modules.get(module)
    for part in name.split("."):
        found = getattr(found, part, None)
    if module == "__main__" or (found is not library and found != library):  # a bound method is made anew each time
        raise ValueError(
            f"{name} cannot be imported from {module}, so an interpreter cannot import it either. Define it in a "
            "module that can be imported."
        )
    return f"{module}:{name}"


def module_of(spec: str) -> str:
    """The module a library's spec imports."""
    return spec.partition(":")[0]


# -- what libraries reach ---------------------------------------------------------------------------------------------


def reach(modules: Iterable[str]) -> tuple[dict[str, Path], set[str]]:
    """What importing ``modules`` leads to in this environment, read from their source without importing them.

    Returns the source files of code that no installed package provides (a project's own), by their path relative
    to the folder of their top-level package, and the top-level names of the installed packages they import.
    Imports are followed through the project's own files only: an installed package stands for itself and for what
    its metadata requires (``requirements``). An import inside a function counts. An import by a computed name is
    not seen: name such a module among ``modules``.
    """
    known = providers()
    tops: dict[str, tuple[str | None, list[Path]]] = {}
    files: dict[str, Path] = {}
    packages: set[str] = set()
    seen: set[str] = set()
    todo = list(modules)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        parts = name.split(".")
        if parts[0] not in tops:
            tops[parts[0]] = _resolve(parts[0], known)
        kind, roots = tops[parts[0]]
        if kind == "package":
            packages.add(parts[0])
        if kind != "source":
            continue
        for depth in range(1, len(parts) + 1):  # importing a.b.c runs a and a.b first
            found = _source_file(roots, parts[:depth])
            if found is None or found[0] in files:
                continue
            key, path = found
            files[key] = path
            module = ".".join(parts[:depth])
            todo += _imports(path, module if path.name == "__init__.py" else module.rpartition(".")[0])
    return files, packages


def _resolve(top: str, providers: dict[str, list[str]]) -> tuple[str | None, list[Path]]:
    """What the top-level module ``top`` is here: an installed ``package``, the ``source`` of a project's own code with
    where it is, or neither (the standard library's, dspy itself, which every interpreter of a dspy program has, a
    compiled module, or one that is not here)."""
    if top in sys.stdlib_module_names or top == "dspy":
        return None, []
    if any(not _from_directory(name) for name in providers.get(top, ())):
        return "package", []
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError, AttributeError):
        spec = None
    if spec is None:
        return None, []
    if spec.submodule_search_locations:
        return "source", [Path(location) for location in spec.submodule_search_locations]
    if spec.origin and spec.origin.endswith(".py"):
        return "source", [Path(spec.origin)]
    return None, []


def _from_directory(distribution: str) -> bool:
    """Whether a distribution was installed from a local directory, like a project into its own environment: its
    version says nothing about the code, which is the project's own (PEP 610 records where an install came from)."""
    try:
        origin = json.loads(metadata.distribution(distribution).read_text("direct_url.json") or "{}")
    except (metadata.PackageNotFoundError, ValueError):
        return False
    return isinstance(origin, dict) and "dir_info" in origin


def providers() -> dict[str, list[str]]:
    """The installed distributions that provide each top-level module."""
    if sys.version_info >= (3, 11):
        return dict(metadata.packages_distributions())
    out: dict[str, list[str]] = {}  # before 3.11, packages_distributions misses distributions without a top_level.txt
    for dist in metadata.distributions():
        tops = set((dist.read_text("top_level.txt") or "").split()) or {
            file.parts[0] if len(file.parts) > 1 else file.stem
            for file in dist.files or ()
            if file.suffix == ".py" and not file.parts[0].endswith((".dist-info", ".data", ".egg-info"))
        }
        for top in tops:
            out.setdefault(top, []).append(dist.metadata["Name"])
    return out


def _source_file(roots: list[Path], parts: list[str]) -> tuple[str, Path] | None:
    """The source file of the module ``parts`` name, below the top-level module's ``roots``, with its path relative
    to the folder the top-level module is in."""
    for root in roots:
        if root.suffix == ".py":  # a top-level module in one file
            if len(parts) == 1:
                return root.name, root
            continue
        base = root.joinpath(*parts[1:])
        for path in (base / "__init__.py", base.with_suffix(".py")):
            if path.is_file() and (path.name == "__init__.py" or len(parts) > 1):
                return Path(parts[0], *path.relative_to(root).parts).as_posix(), path
    return None


def _imports(path: Path, package: str) -> list[str]:
    """The modules the source file at ``path`` imports, anywhere in it, by their absolute names. What a ``from``
    import names may be a module too, so each is listed as one."""
    try:
        tree = ast.parse(path.read_bytes())
    except (OSError, SyntaxError, ValueError):
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            target = node.module
            if node.level:  # relative to the package the file is in
                base = package.split(".") if package else []
                base = base[: len(base) - node.level + 1]
                if not base:
                    continue
                target = ".".join([*base, node.module] if node.module else base)
            if target:
                out.append(target)
                out += [f"{target}.{alias.name}" for alias in node.names if alias.name != "*"]
    return out


def missing(modules: Iterable[str]) -> list[str]:
    """Those of ``modules`` that this environment cannot import, going by its files and installed packages."""
    known = providers()
    absent = []
    for module in modules:
        top = module.split(".")[0]
        kind, roots = _resolve(top, known)
        if kind == "source":
            here = _exists(roots, module.split("."))
        else:  # an installed package, or one every interpreter has
            here = kind == "package" or top in sys.stdlib_module_names or top == "dspy"
        if not here:
            absent.append(module)
    return absent


def requirements(tops: Iterable[str]) -> dict[str, str]:
    """The installed distributions that provide the top-level modules ``tops`` and those their metadata requires, at
    any depth, by normalized name, with the version each has here. dspy's own are left out: an interpreter has the
    dspy of its host, whose version dspy saves with every program."""
    known = providers()
    own = {normalize(name) for name in known.get("dspy", ())}
    found: dict[str, str] = {}
    todo = [name for top in tops for name in known.get(top, ())]
    while todo:
        name = normalize(todo.pop())
        if name in found or name in own:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:  # one this platform does not need
            continue
        found[name] = dist.version
        todo += [re.match(r"[\w.-]+", requirement).group(0) for requirement in dist.requires or ()
                 if "extra ==" not in requirement]
    return found


def normalize(name: str) -> str:
    """A distribution's name as package indexes compare it."""
    return re.sub(r"[-_.]+", "-", name).lower()


# -- what an environment resolves them to -----------------------------------------------------------------------------


def describe(libraries: list[str]) -> str:
    """What the ``libraries`` (their specs) resolve to in this environment, as JSON: its Python version, each
    installed package they reach with its version, as a requirement (``name==version``), and a hash of each source
    file of a project's own code they reach. Nothing is imported."""
    modules = [module_of(spec) for spec in libraries]
    if absent := missing(modules):
        raise ImportError(f"{', '.join(absent)} cannot be imported in this interpreter.")
    files, packages = reach(modules)
    sources: dict[str, dict[str, str]] = {}
    for key, path in sorted(files.items()):
        sources.setdefault(key.split("/")[0].removesuffix(".py"), {})[key] = _digest(path)
    return json.dumps({
        "python": _python(),
        "requires": sorted(f"{name}=={version}" for name, version in requirements(packages).items()),
        "sources": sources,
    })


def verify(record: dict[str, Any]) -> str:
    """What this environment has of what ``record`` (from ``describe``) lists, as JSON: its Python version, each
    package's version here, and each source file's hash here, ``None`` for one that is not here. Nothing is imported
    and no import is followed: a file that imports something new has changed itself."""
    versions: dict[str, str | None] = {}
    for requirement in record.get("requires") or ():
        name = requirement.partition("==")[0]
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    known = providers()
    hashes: dict[str, str | None] = {}
    for top, files in (record.get("sources") or {}).items():
        _, roots = _resolve(top, known)
        for key in files:
            path = _locate(roots, key)
            hashes[key] = _digest(path) if path is not None else None
    return json.dumps({"python": _python(), "versions": versions, "hashes": hashes})


def _exists(roots: list[Path], parts: list[str]) -> bool:
    """Whether the module ``parts`` name is below ``roots``: a source file, or a folder (a package without one)."""
    return _source_file(roots, parts) is not None or any(
        root.suffix != ".py" and root.joinpath(*parts[1:]).is_dir() for root in roots
    )


def _locate(roots: list[Path], key: str) -> Path | None:
    """The source file recorded as ``key``, its path relative to the folder of its top-level module."""
    for root in roots:
        path = root if root.suffix == ".py" else root.joinpath(*key.split("/")[1:])
        if path.is_file() and (root.suffix != ".py" or root.name == key):
            return path
    return None


def _digest(path: Path) -> str:
    # Line endings differ between checkouts of the same commit.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def _python() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


# -- how their API reads ----------------------------------------------------------------------------------------------


def reference(libraries: list[str]) -> str:
    """An API reference of the ``libraries`` (their specs) as this environment has them, for whoever writes code
    against them: how to import each, its docstring, and its public functions and classes with their signatures and
    the first paragraph of their docstrings. A library named as one class or function gets that one, in full."""
    parts = []
    with contextlib.redirect_stdout(io.StringIO()):  # a library may print as it is imported
        for spec in libraries:
            module_name, _, attribute = spec.partition(":")
            module = importlib.import_module(module_name)
            if attribute:
                value: Any = module
                for name in attribute.split("."):
                    value = getattr(value, name)
                head = attribute.split(".")[0]
                lines = [f"from {module_name} import {head}", *_api(attribute, value, doc_chars=4000)]
                title = f"{module_name}.{attribute}"
            else:
                lines = [f"import {module_name}", *_comment(_doc(module, 4000))]
                for name in _public_names(module):
                    if hasattr(module, name):
                        lines.extend(_api(f"{module_name}.{name}", getattr(module, name)))
                title = module_name
            parts.append(f"### {title}{_version(module_name)}\n\n{_truncate(lines, REFERENCE_CHARS)}")
    return "\n\n".join(parts)


def _version(module: str) -> str:
    """The version of the installed package that provides ``module``, to follow its name, if one does."""
    names = [name for name in providers().get(module.split(".")[0], ()) if not _from_directory(name)]
    with contextlib.suppress(metadata.PackageNotFoundError):
        return f" {metadata.version(names[0])}" if len(names) == 1 else ""
    return ""


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


def _api(qualified: str, value: Any, doc_chars: int = 800) -> list[str]:
    if inspect.isclass(value):
        lines = [f"class {qualified}{_signature(value, returns=False)}", *_comment(_doc(value, doc_chars), "    ")]
        if issubclass(value, enum.Enum):
            lines.append(f"    # members: {', '.join(member.name for member in value)}")
            return lines
        attributes, elsewhere = _interface(value)
        if elsewhere:  # what those give it is theirs to document
            lines.append(f"    # inherits from {', '.join(elsewhere)}")
        for name, attribute in attributes.items():
            function = attribute.__func__ if isinstance(attribute, (staticmethod, classmethod)) else attribute
            if name.startswith("_") or not callable(function):
                continue
            lines.append(f"    def {name}{_signature(function, drop_self=not isinstance(attribute, staticmethod))}")
            lines.extend(_comment(_doc(function, doc_chars), "        "))
        return lines
    if callable(value):
        return [f"def {qualified}{_signature(value)}", *_comment(_doc(value, doc_chars), "    ")]
    try:
        text = repr(value)
    except Exception:
        return []
    return [f"{qualified} = {text if len(text) <= 80 else text[:77] + '...'}"]


def _interface(cls: type) -> tuple[dict[str, Any], list[str]]:
    """What a class has as its library defines it: its own attributes and those of its bases in the same library,
    where the class nearest to it wins, and the names of the classes from elsewhere that those derive from."""
    library = (cls.__module__ or "").split(".")[0]
    attributes: dict[str, Any] = {}
    elsewhere: list[str] = []
    for base in cls.__mro__:
        if base is object or (base.__module__ or "").split(".")[0] != library:
            continue
        for name, attribute in vars(base).items():
            attributes.setdefault(name, attribute)
        for parent in base.__bases__:
            if parent is object or (parent.__module__ or "").split(".")[0] == library:
                continue
            named = parent.__qualname__ if parent.__module__ == "builtins" else f"{parent.__module__}.{parent.__qualname__}"
            if named not in elsewhere:
                elsewhere.append(named)
    return attributes, elsewhere


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
            out.append(
                f"# ... {len(lines) - index} more lines. Name a class or a function among the libraries to have "
                "it documented in full."
            )
            break
        out.append(line)
        size += len(line) + 1
    return "\n".join(out)
