"""Adapt the DSPy facade to Monty's native Python subset.

Classes, methods, closures, and calls belong to Monty. We only remove the shim's
no-op Module base/initializer and supply its attribute and subscription hooks.
Validate first so unsupported object-model features fail before guest execution.
"""

import ast
from dataclasses import dataclass

from dspy.primitives.code_interpreter import CodeExecutionError


def _name(name, ctx=None):
    return ast.Name(id=name, ctx=ctx or ast.Load())


def _call(name, *args):
    return ast.Call(func=_name(name), args=list(args), keywords=[])


def _unsupported(node, message):
    raise CodeExecutionError(f"Unsupported Monty syntax at sandbox:{node.lineno}: {message}")


def _module_init(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "__init__"
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "super"
        and not node.func.value.args
        and not node.func.value.keywords
    )


class _Validate(ast.NodeVisitor):
    """Reject semantics we cannot preserve before executing any generated code."""

    _dynamic = frozenset({"eval", "exec", "compile", "globals", "locals", "vars", "dir", "__import__"})
    _unsupported_nodes = frozenset(
        {
            "Delete",
            "Yield",
            "YieldFrom",
            "Match",
            "TryStar",
            "AsyncWith",
            "AsyncFor",
            "AsyncFunctionDef",
        }
    )

    def __init__(self):
        self.in_function = False
        self.module_inits = set()
        self.direct_method = None

    def generic_visit(self, node):
        if type(node).__name__ in self._unsupported_nodes:
            _unsupported(node, type(node).__name__)
        super().generic_visit(node)

    def _identifier(self, node, name):
        if name.startswith(("_dspy", "_Dspy", "__dspy")):
            _unsupported(node, "the _dspy / _Dspy / __dspy namespaces are reserved for the guest runtime")
        if name in {"super", "__class__"}:
            _unsupported(node, "only direct super().__init__(...) in the module initializer is supported")

    def visit_Name(self, node):
        self._identifier(node, node.id)
        if node.id in self._dynamic:
            _unsupported(node, f"{node.id} exposes or replaces compiled execution scope")

    def visit_arg(self, node):
        self._identifier(node, node.arg)
        self.generic_visit(node)

    def visit_alias(self, node):
        self._identifier(node, node.asname or node.name.split(".")[0])
        if node.name == "*":
            _unsupported(node, "wildcard imports")

    def visit_ClassDef(self, node):
        self._identifier(node, node.name)
        if self.in_function:
            _unsupported(node, "nested class definitions")
        if node.decorator_list or node.keywords or len(node.bases) != 1:
            _unsupported(node, "use one dspy.Module base, without class decorators or metaclasses")
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.decorator_list:
                    _unsupported(item, "method decorators and descriptors")
                if item.name.startswith("__") and item.name != "__init__":
                    _unsupported(item, "custom special methods; call forward explicitly")
                if item.name == "__init__":
                    self.module_inits.update(
                        stmt.value for stmt in item.body if isinstance(stmt, ast.Expr) and _module_init(stmt.value)
                    )
            elif not (
                isinstance(item, ast.Pass) or (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant))
            ):
                _unsupported(item, "class bodies may contain only methods and docstrings")
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._identifier(node, node.name)
        previous = self.in_function
        self.in_function = True
        self.generic_visit(node)
        self.in_function = previous

    def visit_Attribute(self, node):
        if node.attr.startswith("__") and not node.attr.endswith("__"):
            _unsupported(node, "name-mangled private attributes; use a single underscore")
        # Only diagnose receivers whose type follows from syntax. Inferring
        # names from assignments or annotations would misclassify rebinding,
        # branch joins, supplied tools, and compiled Flex methods.
        literal_type = {
            ast.List: list,
            ast.ListComp: list,
            ast.Tuple: tuple,
            ast.Dict: dict,
            ast.DictComp: dict,
            ast.Set: set,
            ast.SetComp: set,
            ast.JoinedStr: str,
        }.get(type(node.value))
        if isinstance(node.value, ast.Constant):
            literal_type = type(node.value.value)
        if (
            node is not self.direct_method
            and isinstance(node.ctx, ast.Load)
            and literal_type is not None
            and callable(getattr(literal_type, node.attr, None))
        ):
            _unsupported(
                node,
                f"native method '{node.attr}' used as a value. Call it directly, or use a named helper: "
                f"def helper(value, *args, **kwargs): return value.{node.attr}(*args, **kwargs)",
            )
        self.generic_visit(node)

    def visit_Call(self, node):
        if node in self.module_inits:
            for argument in [*node.args, *node.keywords]:
                self.visit(argument)
            return
        previous = self.direct_method
        self.direct_method = node.func
        self.generic_visit(node)
        self.direct_method = previous

    def visit_comprehension(self, node):
        if any(isinstance(n, (ast.Attribute, ast.Subscript)) for n in ast.walk(node.target)):
            _unsupported(node.target, "attribute/subscript targets in comprehensions; use an explicit loop")
        if node.is_async:
            _unsupported(node.target, "async comprehensions")
        self.generic_visit(node)


class _Compile(ast.NodeTransformer):
    def __init__(self):
        self.augments = 0

    def visit(self, node):
        result = super().visit(node)
        if hasattr(node, "lineno"):
            for item in result if isinstance(result, list) else [result]:
                if isinstance(item, ast.AST):
                    ast.copy_location(item, node)
        return result

    def visit_Import(self, node):
        return [
            ast.Assign(targets=[_name(alias.asname or "dspy", ast.Store())], value=_name("_dspy_namespace"))
            if alias.name == "dspy"
            else ast.Import(names=[alias])
            for alias in node.names
        ]

    def visit_ImportFrom(self, node):
        if node.module != "dspy" or node.level:
            return node
        return [
            ast.Assign(
                targets=[_name(alias.asname or alias.name, ast.Store())],
                value=_call("_dspy_getattr", _name("_dspy_namespace"), ast.Constant(alias.name)),
            )
            for alias in node.names
        ]

    def visit_ClassDef(self, node):
        base = self.visit(node.bases[0])
        node.bases = []
        node = self.generic_visit(node)
        return [
            ast.Expr(value=_call("_dspy_check_base", base)),
            node,
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(value=_name("_dspy_module_types"), attr="append", ctx=ast.Load()),
                    args=[_name(node.name)],
                    keywords=[],
                )
            ),
        ]

    def visit_Attribute(self, node):
        node.value = self.visit(node.value)
        if isinstance(node.ctx, ast.Load):
            return _call("_dspy_getattr", node.value, ast.Constant(node.attr))
        return node

    def visit_Call(self, node):
        if _module_init(node):
            node.func = _name("_dspy_module_init")
        elif isinstance(node.func, ast.Attribute):
            # Native method calls work in Monty even when fetching that method
            # with getattr does not. Do not route calls through an emulator.
            node.func.value = self.visit(node.func.value)
        else:
            node.func = self.visit(node.func)
        node.args = [self.visit(arg) for arg in node.args]
        node.keywords = [self.visit(kw) for kw in node.keywords]
        return node

    def key(self, value):
        if isinstance(value, ast.Slice):
            return _call(
                "_dspy_slice",
                *(self.visit(part) if part else ast.Constant(None) for part in (value.lower, value.upper, value.step)),
            )
        if isinstance(value, ast.Tuple):
            return ast.Tuple(elts=[self.key(item) for item in value.elts], ctx=ast.Load())
        return self.visit(value)

    def visit_Subscript(self, node):
        if isinstance(node.ctx, ast.Load):
            return _call("_dspy_getitem", self.visit(node.value), self.key(node.slice))
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        if not isinstance(node.target, ast.Attribute):
            return self.generic_visit(node)
        # A prediction field needs the same read hook in `p.x += rhs` as in
        # `p.x`. Evaluate the receiver once, and preserve in-place mutation.
        self.augments += 1
        obj, value = f"_dspy_obj_{self.augments}", f"_dspy_value_{self.augments}"
        return [
            ast.Assign(targets=[_name(obj, ast.Store())], value=self.visit(node.target.value)),
            ast.Assign(
                targets=[_name(value, ast.Store())],
                value=_call("_dspy_getattr", _name(obj), ast.Constant(node.target.attr)),
            ),
            ast.AugAssign(target=_name(value, ast.Store()), op=node.op, value=self.visit(node.value)),
            ast.Assign(
                targets=[ast.Attribute(value=_name(obj), attr=node.target.attr, ctx=ast.Store())],
                value=_name(value),
            ),
        ]


@dataclass
class CompiledCode:
    source: str
    original: str
    lines: dict[int, int]

    def annotate(self, error):
        """Attach original statement locations, without depending on feed numbers."""
        generated, original = self.source.splitlines(), self.original.splitlines()
        locations = []
        for frame in error.traceback():
            line = frame.line
            if line in self.lines and frame.source_line.strip() == generated[line - 1].strip():
                source_line = self.lines[line]
                locations.append(f"  sandbox:{source_line}: {original[source_line - 1].strip()}")
        return CodeExecutionError("\n".join([str(error), *locations]))


def compile_source(source: str) -> CompiledCode:
    tree = ast.parse(source)
    _Validate().visit(tree)
    tree = ast.fix_missing_locations(_Compile().visit(tree))
    rendered = ast.unparse(tree)
    # Pair statements, not expression nodes: unparse may normalize expressions
    # but preserves statement structure. Inner statement spans override outer
    # spans, yielding a source map without relying on ast's private unparser.
    before = [node for node in ast.walk(tree) if isinstance(node, ast.stmt)]
    after = [node for node in ast.walk(ast.parse(rendered)) if isinstance(node, ast.stmt)]
    lines = {}
    for old, new in zip(before, after, strict=True):
        assert type(old) is type(new)
        for line in range(new.lineno, new.end_lineno + 1):
            lines[line] = old.lineno
    return CompiledCode(rendered, source, lines)
