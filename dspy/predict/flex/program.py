"""The text of a ``dspy.Flex`` program: the source of one file, or of several Python files, each after a line
``# === name.py ===``. ``program.py`` holds the class the Flex runs; ``test_*.py`` files hold the program's tests,
which its forward never imports. A text without such a line is ``program.py``."""

from __future__ import annotations

import ast
import re

from dspy.predict.flex.ctx import _strip_code_fences
from dspy.primitives.code_interpreter import CodeInterpreterError

ENTRY = "program.py"
_HEADER = re.compile(r"^# === ([A-Za-z_]\w*\.py) ===[ \t]*$", re.MULTILINE)


def is_test(path: str) -> bool:
    return path.startswith("test_")


def parse(text: str) -> dict[str, str]:
    """The files of a program's ``text``, by name. Text before the first header line is ``program.py``'s, and a file
    written in a code fence is read without it."""
    parts = _HEADER.split(text or "")
    files = {ENTRY: parts[0]} if parts[0].strip() else {}
    for name, source in zip(parts[1::2], parts[2::2], strict=True):
        if name in files:
            raise CodeInterpreterError(f"The program writes {name} twice.")
        files[name] = source
    return {name: _strip_code_fences(source) for name, source in files.items()}


def render(files: dict[str, str]) -> str:
    """The text of ``files``: ``program.py`` alone is its source; more files each follow their header line, with
    ``program.py`` first and the tests last."""
    if set(files) == {ENTRY}:
        return files[ENTRY]
    order = sorted(files, key=lambda name: (name != ENTRY, is_test(name), name))
    return "\n\n\n".join(f"# === {name} ===\n{files[name].strip()}" for name in order)


def revise(text: str, revision: str) -> str:
    """``text`` with the files ``revision`` writes: each one added or replaced, or removed if written empty. A
    revision without header lines writes ``program.py``."""
    files = parse(text)
    for name, source in parse(revision).items():
        if source.strip():
            files[name] = source
        else:
            files.pop(name, None)
    return render(files)


def _with_forward(tree: ast.Module) -> list[ast.ClassDef]:
    return [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(isinstance(b, ast.FunctionDef) and b.name == "forward" for b in node.body)
    ]


def _runs(tree: ast.Module, name: str) -> bool:
    """Whether ``tree`` says which class runs: one named ``name``, or else its only class with ``forward``."""
    return any(isinstance(n, ast.ClassDef) and n.name == name for n in tree.body) or len(_with_forward(tree)) == 1


def entry(files: dict[str, str], name: str) -> str:
    """The class of ``program.py`` the Flex runs: the one named ``name``, or else the first class with a
    ``forward``, as a program written before that rule has one."""
    tree = ast.parse(files.get(ENTRY, ""))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if any(node.name == name for node in classes):
        return name
    chosen = _with_forward(tree) or classes
    if not chosen:
        raise CodeInterpreterError("module_src must define a dspy.Module subclass with a `forward` method")
    return chosen[0].name


def problems(files: dict[str, str], name: str) -> list[str]:
    """What keeps the program from running, found without running it: a file that does not parse, a
    ``program.py`` that does not say which class runs, or an import of a file the program does not have."""
    if ENTRY not in files:
        return [f"There is no {ENTRY}."]
    found = []
    for path, source in files.items():
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError as e:
            found.append(f"{path}, line {e.lineno}: {e.msg}")
            continue
        if path == ENTRY and not _runs(tree, name):
            found.append(f"{ENTRY} must define {name}, the class the Flex runs, or exactly one class with forward.")
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            if node.level > 1:
                found.append(f"{path}, line {node.lineno}: the program's files are one package: import with one dot.")
                continue
            imported = [node.module] if node.module else [alias.name for alias in node.names]
            found += [
                f"{path}, line {node.lineno}: there is no {module}.py."
                for module in imported
                if f"{module}.py" not in files
            ]
    return found
