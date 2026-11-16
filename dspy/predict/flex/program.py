"""The text of a ``dspy.Flex`` program: the source of one file, or of several Python files, each after a line
``# === name.py ===``. ``program.py`` holds the class the Flex runs; ``test_*.py`` files hold the program's tests,
which its forward never imports. A text without such a line is ``program.py``.

``parse`` and ``render`` read and write that text exactly. ``read`` reads what a reflection model writes, leniently.
"""

from __future__ import annotations

import ast
import re

from dspy.primitives.code_interpreter import CodeInterpreterError

ENTRY = "program.py"
_HEADER = re.compile(r"^# === ([A-Za-z_]\w*\.py) ===[ \t]*$", re.MULTILINE)
_FENCE = re.compile(r"^[ \t]*```[\w+-]*[ \t]*$")


def is_test(path: str) -> bool:
    return path.startswith("test_")


def parse(text: str) -> dict[str, str]:
    """The files of a program's ``text``, by name."""
    parts = _HEADER.split(text or "")
    files = {ENTRY: parts[0].strip()} if parts[0].strip() else {}
    for name, source in zip(parts[1::2], parts[2::2], strict=True):
        if name in files:
            raise CodeInterpreterError(f"The program has {name} twice.")
        files[name] = source.strip()
    return files


def render(files: dict[str, str]) -> str:
    """The text of ``files``: ``program.py`` alone is its source; more files each follow their header line, with
    ``program.py`` first and the tests last."""
    if set(files) == {ENTRY}:
        return files[ENTRY]
    order = sorted(files, key=lambda name: (name != ENTRY, is_test(name), name))
    return "\n\n\n".join(f"# === {name} ===\n{files[name].strip()}" for name in order)


def read(revision: str) -> dict[str, str]:
    """The files a reflection model's ``revision`` writes, by name, an empty one to remove it. Code fences around a
    file, Windows line endings and words before the first file are dropped; a fence inside the code, as in a string,
    stays. A revision without header lines is ``program.py``; with them, code before the first is ``program.py`` only
    if it defines a class, as ``program.py`` does."""
    parts = _HEADER.split((revision or "").replace("\r\n", "\n"))
    head = _unfenced(parts[0])
    files = {ENTRY: head} if head and (len(parts) == 1 or _defines_a_class(head)) else {}
    for name, source in zip(parts[1::2], parts[2::2], strict=True):
        if name in files:
            raise CodeInterpreterError(f"The revision writes {name} twice: write each file once, after its line.")
        files[name] = _unfenced(source)
    return {name: source.expandtabs(4) for name, source in files.items()}


def _unfenced(source: str) -> str:
    """``source`` without the blank and fence lines at its start and end."""
    lines = source.split("\n")
    while lines and (not lines[0].strip() or _FENCE.match(lines[0])):
        lines.pop(0)
    while lines and (not lines[-1].strip() or _FENCE.match(lines[-1])):
        lines.pop()
    return "\n".join(lines).strip()


def revise(text: str, revision: str) -> str:
    """``text`` with the files ``revision`` writes (``read``): each one added or replaced, or removed if empty."""
    files = parse(text)
    for name, source in read(revision).items():
        if source:
            files[name] = source
        else:
            files.pop(name, None)
    return render(files)


def _defines_a_class(source: str) -> bool:
    try:
        return any(isinstance(node, ast.ClassDef) for node in ast.parse(source).body)
    except SyntaxError:
        return False


def _with_forward(tree: ast.Module) -> list[ast.ClassDef]:
    return [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(isinstance(b, ast.FunctionDef) and b.name == "forward" for b in node.body)
    ]


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
