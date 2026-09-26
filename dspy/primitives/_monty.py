"""Adapt the DSPy facade to Monty's native Python subset.

Classes, methods, closures, and calls belong to Monty. We only remove the shim's
no-op Module base/initializer and supply its attribute and subscription hooks.
Check names required by that adaptation; Monty owns its Python feature support.
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
    """Protect compiler-owned names and the initializer whose base we remove."""

    def __init__(self):
        self.module_inits = set()

    def _identifier(self, node, name):
        if name.startswith(("_dspy", "_Dspy", "__dspy")):
            _unsupported(node, "the _dspy / _Dspy / __dspy namespaces are reserved for the guest runtime")
        if name in {"super", "__class__"}:
            _unsupported(node, "only direct super().__init__(...) in the module initializer is supported")

    def visit_Name(self, node):
        self._identifier(node, node.id)

    def visit_arg(self, node):
        self._identifier(node, node.arg)
        self.generic_visit(node)

    def visit_alias(self, node):
        self._identifier(node, node.asname or node.name.split(".")[0])

    def visit_ClassDef(self, node):
        self._identifier(node, node.name)
        if len(node.bases) == 1 and not node.keywords:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    self.module_inits.update(
                        stmt.value for stmt in item.body if isinstance(stmt, ast.Expr) and _module_init(stmt.value)
                    )
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._identifier(node, node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_Call(self, node):
        if node in self.module_inits:
            for argument in [*node.args, *node.keywords]:
                self.visit(argument)
            return
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
        if any(alias.name == "*" for alias in node.names):
            _unsupported(node, "import DSPy facade names explicitly")
        return [
            ast.Assign(
                targets=[_name(alias.asname or alias.name, ast.Store())],
                value=_call("_dspy_getattr", _name("_dspy_namespace"), ast.Constant(alias.name)),
            )
            for alias in node.names
        ]

    def visit_ClassDef(self, node):
        if len(node.bases) != 1 or node.keywords:
            return self.generic_visit(node)
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
