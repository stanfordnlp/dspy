"""Compiler for batching independent ``llm_query`` loops."""

import ast
import symtable
from dataclasses import dataclass, field

_PURE_FUNCTION_NAMES = frozenset(
    "bool chr dict enumerate float format int len list max min range repr set sorted str sum tuple zip".split()
)
_STRING_METHODS = frozenset("format join lower lstrip replace rstrip strip upper".split())
_COLLECTION_METHODS = frozenset("add append extend insert setdefault update".split())
_RESERVED_RUNTIME_NAMES = frozenset(
    {"llm_query", "llm_query_batched", "__dspy_llm_query_batched", "__dspy_replay_llm_query", "print"}
)
_REFLECTIVE_MODULES = frozenset({"builtins", "gc", "importlib", "inspect", "operator"})
_DANGEROUS_ATTRIBUTES = frozenset(
    {
        "__builtins__",
        "__code__",
        "__defaults__",
        "__delattr__",
        "__dict__",
        "__getattribute__",
        "__globals__",
        "__import__",
        "__kwdefaults__",
        "__self__",
        "__setattr__",
        "__subclasses__",
        "_getframe",
        "attrgetter",
        "currentframe",
        "delattr",
        "eval",
        "exec",
        "f_builtins",
        "f_globals",
        "f_locals",
        "get_referrers",
        "getattr",
        "getattr_static",
        "getmembers",
        "globals",
        "import_module",
        "locals",
        "methodcaller",
        "modules",
        "setattr",
        "vars",
    }
)
_DANGEROUS_NAMES = frozenset(
    {
        "__builtins__",
        "__import__",
        "attrgetter",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "methodcaller",
        "setattr",
        "type",
    }
)
_DANGEROUS_CONSTANTS = frozenset(
    {
        "__builtins__",
        "__code__",
        "__defaults__",
        "__delattr__",
        "__dict__",
        "__getattribute__",
        "__globals__",
        "__import__",
        "__kwdefaults__",
        "__self__",
        "__setattr__",
        "__subclasses__",
        "f_builtins",
        "f_globals",
        "f_locals",
    }
)
_PURE_BLOCKED_NODES = (ast.Await, ast.Lambda, ast.NamedExpr, ast.Yield, ast.YieldFrom)
_BUILDER_BLOCKED_NODES = (
    ast.Await,
    ast.Lambda,
    ast.NamedExpr,
    ast.Yield,
    ast.YieldFrom,
    ast.Break,
    ast.Continue,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
    ast.Raise,
    ast.Return,
    ast.Try,
    ast.While,
    ast.With,
)
_LOOP_BLOCKED_NODES = (
    ast.Await,
    ast.AsyncWith,
    ast.Import,
    ast.ImportFrom,
    ast.Return,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)
_FORBIDDEN_QUERY_PARENTS = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.BoolOp,
    ast.ClassDef,
    ast.For,
    ast.FunctionDef,
    ast.If,
    ast.IfExp,
    ast.Lambda,
    ast.Match,
    ast.Try,
    ast.While,
    ast.With,
    ast.comprehension,
)
_REPLAY_FUNCTIONS = frozenset()


def _indent_source(source: str, depth: int = 1) -> str:
    return "\n".join("    " * depth + line for line in source.splitlines())


def _indent(node: ast.AST, depth: int = 1) -> str:
    return _indent_source(ast.unparse(node), depth)


def _conditional_source(condition: str, true_body: str, false_body: str) -> str:
    return "\n".join(
        [
            f"if {condition}:",
            _indent_source(true_body),
            "else:",
            _indent_source(false_body),
        ]
    )


def _try_finally_source(body: str, cleanup: str) -> str:
    return "\n".join(["try:", _indent_source(body), "finally:", _indent_source(cleanup)])


def _symbol_tables(root: symtable.SymbolTable) -> list[symtable.SymbolTable]:
    tables = [root]
    for table in tables:
        tables.extend(table.get_children())
    return tables


@dataclass
class ProgramAnalysis:
    """Whole-module facts shared by each candidate loop analysis."""

    tree: ast.Module
    tables: list[symtable.SymbolTable]
    parents: dict[ast.AST, ast.AST]
    used_names: set[str]
    shadowed_names: set[str]
    imported_names: set[str]
    pure_functions: set[str]
    pure_modules: set[str]
    assignments: list[tuple[list[ast.expr], ast.expr]] = field(default_factory=list)
    alias_pairs: list[set[str]] = field(default_factory=list)
    uncertain_alias_pairs: list[set[str]] = field(default_factory=list)
    one_shot_names: set[str] = field(default_factory=set)
    unsafe_callbacks: set[str] = field(default_factory=set)

    @classmethod
    def from_code(cls, code: str) -> "ProgramAnalysis":
        tree = ast.parse(code)
        tables = _symbol_tables(symtable.symtable(code, "<rlm>", "exec"))
        symbols = [symbol for table in tables for symbol in table.get_symbols()]
        return cls(
            tree=tree,
            tables=tables,
            parents={child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)},
            used_names={symbol.get_name() for symbol in symbols},
            shadowed_names={symbol.get_name() for symbol in symbols if symbol.is_assigned() or symbol.is_parameter()},
            imported_names={symbol.get_name() for symbol in symbols if symbol.is_imported()},
            pure_functions=set(_PURE_FUNCTION_NAMES),
            pure_modules={
                alias.asname or "json"
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
                if alias.name == "json"
            },
        )

    def prepare(self) -> None:
        """Build alias and purity facts after module safety has been established."""
        self.pure_functions.difference_update(self.shadowed_names | self.imported_names)
        self.pure_modules.difference_update(self.shadowed_names | self._non_json_import_names())
        self.assignments = self._assignments()
        self.alias_pairs = self._alias_pairs(self.assignments)
        self.uncertain_alias_pairs = self._uncertain_alias_pairs()
        self.one_shot_names = self._one_shot_names()
        self.unsafe_callbacks = self._unsafe_callback_names()

    def _non_json_import_names(self) -> set[str]:
        from_imports = {
            alias.asname or alias.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        other_imports = {
            alias.asname or alias.name.split(".", 1)[0]
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name != "json"
        }
        return from_imports | other_imports

    def _assignments(self) -> list[tuple[list[ast.expr], ast.expr]]:
        assignments = [(node.targets, node.value) for node in ast.walk(self.tree) if isinstance(node, ast.Assign)]
        assignments.extend(
            ([node.target], node.value)
            for node in ast.walk(self.tree)
            if isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None
        )
        return assignments

    def _alias_pairs(self, assignments: list[tuple[list[ast.expr], ast.expr]]) -> list[set[str]]:
        return [
            {left, right}
            for targets, value in assignments
            if self._may_retain_alias(value)
            for target in targets
            for left in self.names(target, ast.Store) | ({self.root_name(target)} - {None})
            for right in self.loaded_names(value)
        ]

    def _may_retain_alias(self, value: ast.expr) -> bool:
        if isinstance(
            value,
            (ast.Name, ast.Attribute, ast.Subscript, ast.Tuple, ast.List, ast.BinOp, ast.BoolOp, ast.IfExp, ast.Dict),
        ):
            return True
        if not isinstance(value, ast.Call) or self.is_query_call(value):
            return False
        safe_named_call = isinstance(value.func, ast.Name) and value.func.id in {
            "bool",
            "chr",
            "float",
            "format",
            "int",
            "len",
            "range",
            "repr",
            "str",
            "sum",
        }
        known_fresh_method_call = isinstance(value.func, ast.Attribute) and value.func.attr not in {
            "copy",
            "get",
            "pop",
            "setdefault",
        }
        return not safe_named_call and not known_fresh_method_call

    def _uncertain_alias_pairs(self) -> list[set[str]]:
        bindings = [
            ([node.target], node.iter)
            for node in ast.walk(self.tree)
            if isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(node.iter, (ast.List, ast.Tuple))
        ]
        bindings.extend(
            ([item.optional_vars], item.context_expr)
            for node in ast.walk(self.tree)
            if isinstance(node, (ast.With, ast.AsyncWith))
            for item in node.items
            if item.optional_vars
        )
        return [
            {left, right}
            for targets, value in bindings
            for target in targets
            for left in self.names(target, ast.Store) | ({self.root_name(target)} - {None})
            for right in self.loaded_names(value)
        ]

    def _one_shot_names(self) -> set[str]:
        generator_functions = {
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(isinstance(child, (ast.Yield, ast.YieldFrom)) for child in ast.walk(node))
        }
        return {
            left
            for targets, value in self.assignments
            if isinstance(value, ast.GeneratorExp)
            or (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in generator_functions | {"enumerate", "filter", "iter", "map", "zip"}
            )
            for target in targets
            for left in self.names(target, ast.Store)
        }

    def _unsafe_callback_names(self) -> set[str]:
        defined_functions = {
            node.name for node in self.tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        lambdas = {
            left
            for targets, value in self.assignments
            if isinstance(value, ast.Lambda)
            for target in targets
            for left in self.names(target, ast.Store)
        }
        assignment_pairs = [
            {left, *self.loaded_names(value)}
            for targets, value in self.assignments
            for target in targets
            for left in self.names(target, ast.Store)
        ]
        return self.with_aliases((defined_functions | lambdas) - _REPLAY_FUNCTIONS, assignment_pairs)

    def is_safe_program(self) -> bool:
        if _RESERVED_RUNTIME_NAMES & (self.shadowed_names | self.imported_names):
            return False
        return not any(self.is_unsafe_node(node) for node in ast.walk(self.tree))

    def is_namespace_membership(self, node: ast.Name) -> bool:
        call = self.parents.get(node)
        comparison = self.parents.get(call)
        return (
            isinstance(call, ast.Call)
            and not call.args
            and not call.keywords
            and isinstance(comparison, ast.Compare)
            and len(comparison.ops) == 1
            and isinstance(comparison.ops[0], (ast.In, ast.NotIn))
            and comparison.comparators == [call]
            and isinstance(comparison.left, ast.Constant)
            and type(comparison.left.value) is str
        )

    def is_unsafe_node(self, node: ast.AST) -> bool:
        if isinstance(node, (ast.ClassDef, ast.Delete, ast.Match)):
            return True
        if isinstance(node, ast.Import):
            return any(alias.name.split(".", 1)[0] in _REFLECTIVE_MODULES for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            return self._is_unsafe_import_from(node)
        if isinstance(node, ast.Attribute):
            return node.attr in _DANGEROUS_ATTRIBUTES or (
                isinstance(node.value, ast.Name) and node.value.id in self.pure_functions | {"print"}
            )
        if isinstance(node, ast.Name):
            return self._is_unsafe_name(node)
        return isinstance(node, ast.Constant) and node.value in _DANGEROUS_CONSTANTS

    def _is_unsafe_import_from(self, node: ast.ImportFrom) -> bool:
        return (
            node.module is None
            or node.module.split(".", 1)[0] in _REFLECTIVE_MODULES
            or any(alias.name == "*" or alias.name in {"_getframe", "modules"} for alias in node.names)
        )

    def _is_unsafe_name(self, node: ast.Name) -> bool:
        if node.id in _DANGEROUS_NAMES or node.id.startswith("__dspy_"):
            return True
        return node.id in {"globals", "locals", "vars"} and (
            node.id in self.shadowed_names | self.imported_names or not self.is_namespace_membership(node)
        )

    def fresh_name(self, prefix: str) -> str:
        name, suffix = prefix, 0
        while name in self.used_names:
            suffix += 1
            name = f"{prefix}_{suffix}"
        self.used_names.add(name)
        return name

    @staticmethod
    def is_query_call(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "llm_query"
            and len(node.args) == 1
            and not node.keywords
        )

    @staticmethod
    def bound_names(node: ast.AST) -> list[str] | None:
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, (ast.Tuple, ast.List)) and all(isinstance(value, ast.Name) for value in node.elts):
            return [value.id for value in node.elts]
        return None

    @staticmethod
    def names(node: ast.AST, context: type[ast.expr_context]) -> set[str]:
        return {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, context)}

    def stored_names(self, node: ast.AST) -> set[str]:
        return self.names(node, ast.Store) | {
            child.name
            for child in ast.walk(node)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            or (isinstance(child, ast.ExceptHandler) and child.name)
        }

    def loaded_names(self, node: ast.AST) -> set[str]:
        return self.names(node, ast.Load) - self.names(node, ast.Store) | {
            load.id
            for expression in ast.walk(node)
            if isinstance(expression, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp))
            for load in ast.walk(node)
            if isinstance(load, ast.Name)
            and isinstance(load.ctx, ast.Load)
            and load.id
            in set().union(*(self.names(generator.target, ast.Store) for generator in expression.generators))
            and (
                load not in set(ast.walk(expression))
                or any(load in set(ast.walk(generator.iter)) for generator in expression.generators)
            )
        }

    @staticmethod
    def root_name(node: ast.AST) -> str | None:
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    def has_callback_options(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Call) and (
            (
                isinstance(node.func, ast.Name)
                and node.func.id in {"max", "min", "sorted"}
                and any(keyword.arg in {None, "key"} for keyword in node.keywords)
            )
            or (
                isinstance(node.func, ast.Attribute)
                and (
                    (node.func.attr == "sort" and any(keyword.arg in {None, "key"} for keyword in node.keywords))
                    or self._json_call_has_callback(node)
                )
            )
        )

    def _json_call_has_callback(self, node: ast.Call) -> bool:
        if not isinstance(node.func, ast.Attribute):
            return False
        if not isinstance(node.func.value, ast.Name) or node.func.value.id not in self.pure_modules:
            return False
        if node.func.attr not in {"dumps", "loads"}:
            return False
        callback_keywords = {
            None,
            "cls",
            "default",
            "object_hook",
            "object_pairs_hook",
            "parse_constant",
            "parse_float",
            "parse_int",
        }
        return any(keyword.arg in callback_keywords for keyword in node.keywords)

    def allowed_call(self, node: ast.AST, mutable: set[str] = frozenset()) -> bool:
        return not isinstance(node, ast.Call) or (
            (
                isinstance(node.func, ast.Name)
                and node.func.id in self.pure_functions
                and not self.has_callback_options(node)
            )
            or (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in self.pure_modules
                and node.func.attr in {"dumps", "loads"}
                and not self.has_callback_options(node)
            )
            or self._allowed_string_call(node)
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _COLLECTION_METHODS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in mutable
            )
        )

    @staticmethod
    def _allowed_string_call(node: ast.Call) -> bool:
        return (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in _STRING_METHODS
            and (
                isinstance(node.func.value, (ast.Constant, ast.JoinedStr))
                or (
                    isinstance(node.func.value, ast.Call)
                    and isinstance(node.func.value.func, ast.Name)
                    and node.func.value.func.id in {"chr", "format", "str"}
                )
            )
        )

    def is_pure(self, node: ast.AST) -> bool:
        return all(
            not isinstance(child, _PURE_BLOCKED_NODES)
            and not (isinstance(child, ast.comprehension) and child.is_async)
            and self.allowed_call(child)
            for child in ast.walk(node)
        )

    def query_runs_first(self, statement: ast.stmt, query: ast.Call, loop: ast.For) -> bool:
        return (
            (isinstance(statement, ast.Expr) and statement.value is query)
            or (
                isinstance(statement, ast.Assign)
                and statement.value is query
                and all(isinstance(target, ast.Name) for target in statement.targets)
            )
            or (
                isinstance((value := getattr(statement, "value", None)), ast.Call)
                and isinstance(value.func, ast.Attribute)
                and isinstance(value.func.value, ast.Name)
                and value.func.attr == "append"
                and value.args == [query]
                and not value.keywords
                and self.is_owned_name(value.func.value.id, loop)
            )
        )

    def mutation_roots(self, node: ast.AST) -> set[str]:
        children = list(ast.walk(node))
        roots = {self.root_name(child.target) for child in children if isinstance(child, ast.AugAssign)}
        roots.update(
            self.root_name(child)
            for child in children
            if isinstance(child, (ast.Attribute, ast.Subscript)) and isinstance(child.ctx, ast.Store)
        )
        roots.update(
            self.root_name(child.func.value)
            for child in children
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr in _COLLECTION_METHODS
        )
        return roots - {None}

    def is_owned_value(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Constant)
            or (
                isinstance(node, (ast.List, ast.Set, ast.Tuple))
                and all(self.is_owned_value(value) for value in node.elts)
            )
            or (
                isinstance(node, ast.Dict)
                and all(
                    key is not None and self.is_owned_value(key) and self.is_owned_value(value)
                    for key, value in zip(node.keys, node.values, strict=True)
                )
            )
            or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in self.pure_functions & {"dict", "list", "set", "str", "tuple"}
                and not node.args
                and not node.keywords
            )
        )

    def precedes(self, statement: ast.stmt, node: ast.AST) -> bool:
        return (parent := self.parents.get(node)) is not None and (
            any(
                statement in values[: values.index(node)]
                for _, values in ast.iter_fields(parent)
                if isinstance(values, list) and node in values
            )
            or self.precedes(statement, parent)
        )

    def is_owned_name(self, name: str, loop: ast.For) -> bool:
        return (
            name not in self.imported_names
            and sum(self._binds_name(child, name) for child in ast.walk(self.tree)) == 1
            and any(
                self._is_preceding_empty_list_assignment(statement, name, loop) for statement in ast.walk(self.tree)
            )
        )

    @staticmethod
    def _binds_name(node: ast.AST, name: str) -> bool:
        return (
            (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == name)
            or (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name)
            or (isinstance(node, ast.ExceptHandler) and node.name == name)
        )

    def _is_preceding_empty_list_assignment(self, statement: ast.AST, name: str, loop: ast.For) -> bool:
        return (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == name
            and isinstance(statement.value, ast.List)
            and not statement.value.elts
            and self.precedes(statement, loop)
        )

    def enclosing_function(self, node: ast.AST) -> ast.AST | None:
        parent = self.parents.get(node)
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return parent
        return self.enclosing_function(parent) if parent else None

    def guarded_name(self, name: str, node: ast.AST) -> tuple[str, str]:
        scope = self.enclosing_function(node)
        is_local = any(self._table_has_local_name(table, scope, name) for table in self.tables)
        if is_local:
            return f"{name!r} in locals()", f"locals()[{name!r}]"
        return (
            f"({name!r} in locals() or {name!r} in globals())",
            f"(locals() if {name!r} in locals() else globals())[{name!r}]",
        )

    @staticmethod
    def _table_has_local_name(table: symtable.SymbolTable, scope: ast.AST | None, name: str) -> bool:
        return (
            scope is not None
            and table.get_type() == "function"
            and table.get_name() == scope.name
            and table.get_lineno() == scope.lineno
            and any(
                symbol.get_name() == name and (symbol.is_local() or symbol.is_parameter())
                for symbol in table.get_symbols()
            )
        )

    def is_local_builder(self, node: ast.AST, roots: set[str]) -> bool:
        return all(self._allowed_builder_node(child, roots) for child in ast.walk(node))

    def _allowed_builder_node(self, child: ast.AST, roots: set[str]) -> bool:
        if isinstance(child, _BUILDER_BLOCKED_NODES) or not self.allowed_call(child, roots):
            return False
        if (
            isinstance(child, (ast.Attribute, ast.Subscript))
            and isinstance(child.ctx, ast.Store)
            and self.root_name(child) not in roots
        ):
            return False
        if isinstance(child, ast.AugAssign) and not isinstance(child.target, ast.Name):
            return False
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr in _COLLECTION_METHODS - {"append"}
        ):
            return False
        return not (
            isinstance(child, (ast.For, ast.AsyncFor))
            and self.loaded_names(child.iter) & self.with_aliases(self.one_shot_names)
        )

    def with_aliases(self, values: set[str], pairs: list[set[str]] | None = None) -> set[str]:
        pairs = self.alias_pairs + self.uncertain_alias_pairs if pairs is None else pairs
        expanded = set(values)
        while any(pair & expanded and not pair <= expanded for pair in pairs):
            for pair in pairs:
                if pair & expanded:
                    expanded.update(pair)
        return expanded

    def statement_queries(self, statement: ast.stmt) -> list[ast.Call] | None:
        collector = StatementQueryCollector(self)
        return collector.collect(statement)


class StatementQueryCollector:
    """Collect unconditional, well-formed query calls from one statement."""

    def __init__(self, program: ProgramAnalysis):
        self.program = program
        self.queries: list[ast.Call] = []

    def collect(self, statement: ast.stmt) -> list[ast.Call] | None:
        return self.queries if self._visit(statement) else None

    def _visit(self, node: ast.AST, conditional: bool = False) -> bool:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "llm_query":
            if conditional or not self.program.is_query_call(node):
                return False
            self.queries.append(node)
        child_conditional = conditional or isinstance(node, _FORBIDDEN_QUERY_PARENTS)
        return all(self._visit(child, child_conditional) for child in ast.iter_child_nodes(node))


@dataclass
class BlockAnalysis:
    """Mutable facts accumulated while deciding whether one loop is batchable."""

    node: ast.For
    generated_start: set[str]
    loop_names: list[str] = field(default_factory=list)
    parameter_aliases: set[str] = field(default_factory=set)
    query_statements: dict[int, list[ast.Call]] = field(default_factory=dict)
    first_query: int = -1
    last_query: int = -1
    guards: set[int] = field(default_factory=set)
    prequery_prints: dict[int, str] = field(default_factory=dict)
    stores: list[set[str]] = field(default_factory=list)
    candidates: dict[int, tuple[str, ast.expr]] = field(default_factory=dict)
    selected: set[int] = field(default_factory=set)
    duplicated: set[int] = field(default_factory=set)
    dependencies: set[str] = field(default_factory=set)
    mutations: set[str] = field(default_factory=set)
    runtime_mutations: set[str] = field(default_factory=set)
    callback_roots: set[str] = field(default_factory=set)
    receiver_roots: set[str] = field(default_factory=set)
    runtime_aliases: list[tuple[str, str]] = field(default_factory=list)
    runtime_callbacks: set[str] = field(default_factory=set)
    runtime_receivers: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class LoopPlan:
    """Complete, behavior-preserving recipe for lowering one query loop."""

    node: ast.For
    generated_start: set[str]
    loop_names: list[str]
    query_statements: dict[int, list[ast.Call]]
    guards: set[int]
    prequery_prints: dict[int, str]
    candidates: dict[int, tuple[str, ast.expr]]
    selected: set[int]
    duplicated: set[int]
    runtime_aliases: list[tuple[str, str]]
    runtime_callbacks: set[str]
    runtime_receivers: set[str]


class DependencyResolver:
    """Find pure assignments required to evaluate gathered prompts and guards."""

    def __init__(self, program: ProgramAnalysis, block: BlockAnalysis):
        self.program = program
        self.block = block
        self.resolving: set[int] = set()

    def resolve(self, name: str, before: int) -> bool:
        for index in range(before - 1, -1, -1):
            if name not in self.block.stores[index]:
                continue
            if name in self.program.mutation_roots(self.block.node.body[index]):
                continue
            if index not in self.block.candidates or index in self.resolving:
                return False
            if index in self.block.selected:
                return True
            self.resolving.add(index)
            if not self._resolve_candidate_dependencies(index):
                return False
            self.resolving.remove(index)
            self.block.selected.add(index)
            return True
        return name in self.block.loop_names or name not in set().union(*self.block.stores)

    def _resolve_candidate_dependencies(self, index: int) -> bool:
        value = self.block.candidates[index][1]
        return all(self.resolve(dependency, index) for dependency in self.program.loaded_names(value))


class LoopAnalyzer:
    """Run named eligibility phases for one candidate loop."""

    def __init__(self, program: ProgramAnalysis, node: ast.For):
        self.program = program
        self.block = BlockAnalysis(node=node, generated_start=set(program.used_names))
        self.resolver = DependencyResolver(program, self.block)

    def analyze(self) -> LoopPlan | None:
        phases = (
            self._supports_loop_shape,
            self._binds_stable_loop_target,
            self._collect_queries,
            self._classify_control_flow,
            self._resolve_prompt_dependencies,
            self._validate_local_builders,
            self._validate_selected_assignments,
            self._classify_replay_prefix,
            self._analyze_mutations,
            self._finalize_runtime_guards,
        )
        if not all(phase() for phase in phases):
            return None
        return self._plan()

    def _supports_loop_shape(self) -> bool:
        node = self.block.node
        return (
            not node.orelse
            and self.program.is_pure(node.iter)
            and not any(self._unsupported_loop_node(child) for child in ast.walk(node))
        )

    def _unsupported_loop_node(self, child: ast.AST) -> bool:
        if isinstance(child, _LOOP_BLOCKED_NODES):
            return True
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self._has_eager_function_expressions(child)
        return isinstance(child, ast.Lambda) and bool(child.args.defaults or any(child.args.kw_defaults))

    @staticmethod
    def _has_eager_function_expressions(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        arguments = [
            *node.args.posonlyargs,
            *node.args.args,
            node.args.vararg,
            *node.args.kwonlyargs,
            node.args.kwarg,
        ]
        return bool(
            node.decorator_list
            or node.returns
            or node.args.defaults
            or any(node.args.kw_defaults)
            or any(argument.annotation for argument in arguments if argument)
            or getattr(node, "type_params", [])
        )

    def _binds_stable_loop_target(self) -> bool:
        node = self.block.node
        self.block.loop_names = self.program.bound_names(node.target) or []
        function = self.program.enclosing_function(node)
        parameters = {
            arg.arg
            for enclosing in [function]
            if enclosing is not None
            for arg in ast.walk(enclosing.args)
            if isinstance(arg, ast.arg)
        }
        self.block.parameter_aliases = self.program.with_aliases(parameters)
        body_stores = set().union(*(self.program.stored_names(statement) for statement in node.body))
        return bool(
            self.block.loop_names
            and len(self.block.loop_names) == len(set(self.block.loop_names))
            and not set(self.block.loop_names) & body_stores
        )

    def _collect_queries(self) -> bool:
        for index, statement in enumerate(self.block.node.body):
            queries = self.program.statement_queries(statement)
            if queries is None or len(queries) > 1:
                return False
            if queries and not self.program.query_runs_first(statement, queries[0], self.block.node):
                return False
            if queries:
                self.block.query_statements[index] = queries
        if not self.block.query_statements:
            return False
        self.block.first_query = min(self.block.query_statements)
        self.block.last_query = max(self.block.query_statements)
        return True

    def _classify_control_flow(self) -> bool:
        for index, statement in enumerate(self.block.node.body):
            breaks = any(isinstance(child, ast.Break) for child in ast.walk(statement))
            continues = any(isinstance(child, ast.Continue) for child in ast.walk(statement))
            if not breaks and not continues:
                continue
            if self._is_simple_prequery_guard(index, statement):
                self.block.guards.add(index)
            elif breaks or index <= self.block.last_query:
                return False
        return True

    def _is_simple_prequery_guard(self, index: int, statement: ast.stmt) -> bool:
        return (
            index < self.block.first_query
            and isinstance(statement, ast.If)
            and not statement.orelse
            and len(statement.body) == 1
            and isinstance(statement.body[0], (ast.Break, ast.Continue))
            and self.program.is_pure(statement.test)
        )

    def _resolve_prompt_dependencies(self) -> bool:
        node = self.block.node
        self.block.stores = [self.program.stored_names(statement) for statement in node.body]
        self.block.candidates = {
            index: (statement.targets[0].id, statement.value)
            for index, statement in enumerate(node.body)
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and index not in self.block.query_statements
            and self.program.is_pure(statement.value)
        }
        self.block.dependencies = self.program.loaded_names(node.iter)
        expressions = [
            *((index, node.body[index].test) for index in self.block.guards),
            *((index, query.args[0]) for index, queries in self.block.query_statements.items() for query in queries),
        ]
        return all(self._resolve_expression(index, expression) for index, expression in expressions)

    def _resolve_expression(self, index: int, expression: ast.expr) -> bool:
        expression_names = self.program.loaded_names(expression)
        self.block.dependencies.update(expression_names)
        return self.program.is_pure(expression) and all(self.resolver.resolve(name, index) for name in expression_names)

    def _validate_local_builders(self) -> bool:
        for index in range(self.block.first_query):
            statement = self.block.node.body[index]
            roots = self.program.mutation_roots(statement)
            if not roots:
                continue
            initializers = {self._selected_initializer(root) for root in roots}
            if self._has_bad_initializer(index, initializers):
                return False
            if not self.program.is_local_builder(statement, roots):
                return False
            self.block.duplicated.update({index, *initializers})
        if not all(self._resolve_duplicated_dependencies(index) for index in self.block.duplicated):
            return False
        if self.block.guards and self.block.duplicated and max(self.block.guards) > min(self.block.duplicated):
            return False
        self.block.selected.difference_update(self.block.duplicated)
        return True

    def _selected_initializer(self, root: str) -> int | None:
        selected_by_name = {self.block.candidates[index][0]: index for index in self.block.selected}
        return selected_by_name.get(root)

    def _has_bad_initializer(self, index: int, initializers: set[int | None]) -> bool:
        if None in initializers:
            return True
        concrete = {initializer for initializer in initializers if initializer is not None}
        if any(
            initializer >= index or not self.program.is_owned_value(self.block.candidates[initializer][1])
            for initializer in concrete
        ):
            return True
        has_augassign = any(isinstance(child, ast.AugAssign) for child in ast.walk(self.block.node.body[index]))
        return has_augassign and any(
            not isinstance(self.block.candidates[initializer][1], ast.Constant) for initializer in concrete
        )

    def _resolve_duplicated_dependencies(self, index: int) -> bool:
        return all(
            self.resolver.resolve(name, index) for name in self.program.loaded_names(self.block.node.body[index])
        )

    def _validate_selected_assignments(self) -> bool:
        selected_names = [self.block.candidates[index][0] for index in self.block.selected]
        if set(self.block.loop_names) & set(selected_names) or len(selected_names) != len(set(selected_names)):
            return False
        selected_positions = {self.block.candidates[index][0]: index for index in self.block.selected}
        if any(
            position > index and name in self.program.loaded_names(statement)
            for index, statement in enumerate(self.block.node.body)
            for name, position in selected_positions.items()
        ):
            return False
        self.block.dependencies.update(
            selected_names,
            *(self.program.loaded_names(self.block.candidates[index][1]) for index in self.block.selected),
            *(self.program.loaded_names(self.block.node.body[index]) for index in self.block.duplicated),
        )
        return True

    def _classify_replay_prefix(self) -> bool:
        classified = self.block.selected | self.block.duplicated | self.block.guards | set(self.block.query_statements)
        for index in range(self.block.last_query + 1):
            if index in classified:
                continue
            statement = self.block.node.body[index]
            print_values = self._print_values(statement)
            if print_values is None or not self._is_safe_prequery_print(index, statement, print_values):
                return False
            if index < self.block.first_query and print_values:
                self.block.prequery_prints[index] = ", ".join(ast.unparse(value) for value in print_values)
        return True

    @staticmethod
    def _print_values(statement: ast.stmt) -> list[ast.expr] | None:
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "print"
        ):
            return None
        return [*statement.value.args, *(keyword.value for keyword in statement.value.keywords)]

    def _is_safe_prequery_print(self, index: int, statement: ast.stmt, print_values: list[ast.expr]) -> bool:
        if self.program.stored_names(statement):
            return False
        if index >= self.block.first_query:
            return True
        safe_names = self._safe_print_names()
        return (
            all(self.program.is_pure(value) for value in print_values)
            and not (self.program.loaded_names(statement) - safe_names)
            and not any(self._unsafe_print_argument(value) for value in statement.value.args)
            and not statement.value.keywords
        )

    def _safe_print_names(self) -> set[str]:
        selected_names = {self.block.candidates[index][0] for index in self.block.selected}
        duplicated_roots = set().union(
            *(self.program.mutation_roots(self.block.node.body[index]) for index in self.block.duplicated)
        )
        return (
            set(self.block.loop_names)
            | self.program.loaded_names(self.block.node.iter)
            | selected_names
            | duplicated_roots
            | self.program.pure_functions
            | {"print"}
        )

    @staticmethod
    def _unsafe_print_argument(value: ast.expr) -> bool:
        return isinstance(value, ast.Starred) or any(
            isinstance(child, ast.GeneratorExp)
            or (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id in {"enumerate", "zip"}
            )
            for child in ast.walk(value)
        )

    def _analyze_mutations(self) -> bool:
        for index, statement in enumerate(self.block.node.body):
            if index in self.block.selected | self.block.duplicated:
                continue
            statement_mutations = self.program.mutation_roots(statement)
            self.block.mutations.update(self.program.stored_names(statement), statement_mutations)
            self.block.runtime_mutations.update(statement_mutations)
            if index > self.block.last_query:
                self._track_replay_receivers(index, statement, statement_mutations)
            if not self._analyze_statement_calls(index, statement, statement_mutations):
                return False
        return True

    def _track_replay_receivers(self, index: int, statement: ast.stmt, statement_mutations: set[str]) -> None:
        prior_stores = set().union(
            *(
                self.program.names(target, ast.Store)
                for position in range(index)
                for target in self._assignment_targets(self.block.node.body[position])
            )
        )
        self.block.receiver_roots.update(
            (
                self.program.loaded_names(statement)
                | (self.program.names(statement, ast.Load) & self.program.names(statement, ast.Store))
                | statement_mutations
            )
            - set(self.block.loop_names)
            - prior_stores
        )

    @staticmethod
    def _assignment_targets(statement: ast.stmt) -> list[ast.expr]:
        if isinstance(statement, ast.Assign):
            return statement.targets
        if isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
            return [statement.target]
        return []

    def _analyze_statement_calls(self, index: int, statement: ast.stmt, statement_mutations: set[str]) -> bool:
        calls = (
            child
            for child in ast.walk(statement)
            if isinstance(child, ast.Call) and not self.program.is_query_call(child)
        )
        return all(self._analyze_call(index, call, statement_mutations) for call in calls)

    def _analyze_call(self, index: int, call: ast.Call, statement_mutations: set[str]) -> bool:
        known = self.program.allowed_call(call, statement_mutations) or (
            isinstance(call.func, ast.Name) and call.func.id == "print"
        )
        if not known and index <= self.block.last_query:
            return False
        if known:
            return True
        if self._has_unsafe_callback(call):
            return False
        self._track_unknown_call(call)
        return True

    def _has_unsafe_callback(self, call: ast.Call) -> bool:
        if self.program.has_callback_options(call):
            return True
        if isinstance(call.func, ast.Name) and call.func.id not in _REPLAY_FUNCTIONS:
            return True
        values = [*call.args, *(keyword.value for keyword in call.keywords)]
        return any(
            self.program.with_aliases(self.program.loaded_names(value))
            & (self.program.unsafe_callbacks | self.block.parameter_aliases)
            for value in values
        )

    def _track_unknown_call(self, call: ast.Call) -> None:
        argument_values = [*call.args, *(keyword.value for keyword in call.keywords)]
        argument_roots = [self.program.loaded_names(value) for value in argument_values]
        receiver_roots = [self.program.loaded_names(call.func.value)] if isinstance(call.func, ast.Attribute) else []
        self.block.mutations.update(*argument_roots, *receiver_roots)
        self.block.runtime_mutations.update(*argument_roots, *receiver_roots)
        callback_values = [
            *argument_values,
            *([call.func] if not isinstance(call.func, (ast.Name, ast.Attribute)) else []),
        ]
        self.block.callback_roots.update(*(self.program.loaded_names(value) for value in callback_values))
        self.block.receiver_roots.update(*receiver_roots)

    def _finalize_runtime_guards(self) -> bool:
        dependency_aliases = self.program.with_aliases(self.block.dependencies)
        mutation_aliases = self.program.with_aliases(self.block.mutations)
        if set(self.block.loop_names) & self.block.mutations:
            return False
        if dependency_aliases & mutation_aliases:
            return False
        if any(pair & dependency_aliases and pair & mutation_aliases for pair in self.program.uncertain_alias_pairs):
            return False
        runtime_mutations = self.program.with_aliases(self.block.runtime_mutations)
        excluded = self.program.pure_functions | _REPLAY_FUNCTIONS | {"llm_query", "print"}
        self.block.runtime_aliases = [
            (left, right)
            for left in dependency_aliases - self.program.pure_functions - {"llm_query", "print"}
            for right in runtime_mutations - self.program.pure_functions - {"llm_query", "print"}
            if left != right
        ]
        self.block.runtime_callbacks = self.program.with_aliases(self.block.callback_roots) - excluded
        self.block.runtime_receivers = (
            self.program.with_aliases(self.block.receiver_roots | (self.block.dependencies & self.program.pure_modules))
            - excluded
        )
        return True

    def _plan(self) -> LoopPlan:
        return LoopPlan(
            node=self.block.node,
            generated_start=self.block.generated_start,
            loop_names=self.block.loop_names,
            query_statements=self.block.query_statements,
            guards=self.block.guards,
            prequery_prints=self.block.prequery_prints,
            candidates=self.block.candidates,
            selected=self.block.selected,
            duplicated=self.block.duplicated,
            runtime_aliases=self.block.runtime_aliases,
            runtime_callbacks=self.block.runtime_callbacks,
            runtime_receivers=self.block.runtime_receivers,
        )


@dataclass(frozen=True)
class LoweringNames:
    frames: str
    prompts: str
    responses: str
    frame: str
    frame_index: str
    final_target: str
    final_values: str
    gather_error: str
    gather_position: str
    gather_failed: str
    frame_error: str

    @classmethod
    def allocate(cls, program: ProgramAnalysis) -> "LoweringNames":
        names = (
            program.fresh_name(f"__dspy_{name}")
            for name in (
                "frames",
                "prompts",
                "responses",
                "frame",
                "frame_index",
                "final_target",
                "final_values",
                "gather_error",
                "gather_position",
                "gather_failed",
            )
        )
        return cls(*names, frame_error=program.fresh_name("__dspy_frame_error"))

    def cleanup_names(self) -> tuple[str, ...]:
        return (
            self.frames,
            self.prompts,
            self.responses,
            self.frame,
            self.frame_index,
            self.final_target,
            self.final_values,
            self.gather_error,
            self.gather_position,
            self.gather_failed,
        )


class ResponseReplacer(ast.NodeTransformer):
    """Replace scalar queries with ordered accesses into batched responses."""

    def __init__(self, program: ProgramAnalysis, names: LoweringNames, queries_per_frame: int):
        self.program = program
        self.names = names
        self.queries_per_frame = queries_per_frame
        self.query_index = 0

    def visit_Call(self, call: ast.Call) -> ast.AST:
        if not self.program.is_query_call(call):
            return self.generic_visit(call)
        expression = ast.parse(
            f"__dspy_replay_llm_query({self.names.responses}[{self.names.frame_index} * "
            f"{self.queries_per_frame} + {self.query_index}])",
            mode="eval",
        ).body
        self.query_index += 1
        return ast.copy_location(expression, call)


class LoopLowerer:
    """Lower an eligible loop into gather, batch, replay, and fallback stages."""

    def __init__(self, program: ProgramAnalysis, plan: LoopPlan):
        self.program = program
        self.plan = plan
        self.names = LoweringNames.allocate(program)
        self.selected_order = sorted(plan.selected)
        self.original_loop = ast.unparse(plan.node)

    def lower(self) -> list[ast.stmt] | None:
        gather = self._gather_stage()
        replayed_body = self._replayed_body()
        if replayed_body is None:
            return None
        self._append_replay_stage(gather, replayed_body)
        compiled = self._compiled_gather(gather)
        compiled = self._wrap_runtime_guards(compiled)
        compiled = self._wrap_instrumentation_guard(compiled)
        compiled = self._wrap_collision_guard(compiled)
        return ast.parse(compiled).body

    def _gather_stage(self) -> list[str]:
        names, plan = self.names, self.plan
        target_source = ast.unparse(plan.node.target)
        gather = [
            "try:",
            f"    for {target_source} in {ast.unparse(plan.node.iter)}:",
            f"        {names.final_target} = [{', '.join(plan.loop_names)}]",
            f"        {names.frame} = [{', '.join([*plan.loop_names, *(['None'] * len(self.selected_order))])}]",
            "        try:",
        ]
        for index, statement in enumerate(plan.node.body):
            self._append_gather_statement(gather, index, statement)
        gather.extend(
            [
                "        except Exception:",
                f"            {names.gather_failed} = True",
                f"            {names.frames}.append({names.frame})",
                "            raise",
                f"        {names.frames}.append({names.frame})",
                f"except Exception as {names.frame_error}:",
                f"    {names.gather_error} = {names.frame_error}",
                f"{names.responses} = __dspy_llm_query_batched({names.prompts})",
                f"for {names.frame_index}, {names.frame} in enumerate({names.frames}):",
                f"    {self._replay_target()} = "
                f"{', '.join(f'{names.frame}[{index}]' for index in range(len(plan.loop_names)))}",
            ]
        )
        return gather

    def _append_gather_statement(self, gather: list[str], index: int, statement: ast.stmt) -> None:
        plan, names = self.plan, self.names
        staged = plan.selected | plan.duplicated | plan.guards | set(plan.query_statements) | set(plan.prequery_prints)
        if index in staged:
            printed = f"\n            ({plan.prequery_prints[index]},)" if index in plan.prequery_prints else ""
            gather.append(f"            {names.gather_position} = {index}{printed}")
        if index in plan.selected | plan.duplicated:
            gather.append(_indent(statement, 3))
        if index in plan.selected:
            offset = len(plan.loop_names) + self.selected_order.index(index)
            candidate = plan.candidates[index][0]
            gather.extend(
                [
                    f"            {names.frame}[{offset}] = {candidate}",
                    f"            {names.final_values}[{index}] = {candidate}",
                ]
            )
        elif index in plan.guards:
            gather.append(_indent(statement, 3))
        for query in plan.query_statements.get(index, []):
            gather.append(f"            {names.prompts}.append({ast.unparse(query.args[0])})")

    def _replayed_body(self) -> list[ast.stmt] | None:
        replacer = ResponseReplacer(self.program, self.names, len(self.plan.query_statements))
        body = [replacer.visit(statement) for statement in self.plan.node.body]
        return body if replacer.query_index == len(self.plan.query_statements) else None

    def _append_replay_stage(self, gather: list[str], replayed_body: list[ast.stmt]) -> None:
        names = self.names
        gather.append(f"    if {names.gather_failed} and {names.frame_index} == len({names.frames}) - 1:")
        for index, statement in enumerate(replayed_body):
            comparison = ">" if index in self.plan.query_statements else ">="
            if index in self.plan.query_statements:
                gather.extend(
                    [
                        f"        if {names.gather_position} == {index}:",
                        f"            raise {names.gather_error}",
                    ]
                )
            gather.append(f"        if {names.gather_position} {comparison} {index}:")
            gather.append(_indent(statement, 3))
        gather.extend([f"        raise {names.gather_error}", "    else:"])
        for index, statement in enumerate(replayed_body):
            gather.append(self._successful_replay_statement(index, statement))
        self._append_final_locals(gather)

    def _successful_replay_statement(self, index: int, statement: ast.stmt) -> str:
        if index not in self.plan.selected:
            return _indent(statement, 2)
        candidate = self.plan.candidates[index][0]
        offset = len(self.plan.loop_names) + self.selected_order.index(index)
        return f"        {candidate} = {self.names.frame}[{offset}]"

    def _append_final_locals(self, gather: list[str]) -> None:
        names = self.names
        gather.extend(
            [
                f"if {names.final_target} is not None:",
                f"    {self._replay_target()} = "
                f"{', '.join(f'{names.final_target}[{index}]' for index in range(len(self.plan.loop_names)))}",
            ]
        )
        for index in self.selected_order:
            candidate = self.plan.candidates[index][0]
            gather.extend(
                [
                    f"if {index} in {names.final_values}:",
                    f"    {candidate} = {names.final_values}[{index}]",
                ]
            )
        gather.extend([f"if {names.gather_error} is not None:", f"    raise {names.gather_error}"])

    def _replay_target(self) -> str:
        return ", ".join(self.plan.loop_names)

    def _compiled_gather(self, gather: list[str]) -> str:
        names = self.names
        initializers = [
            f"{names.frames} = []",
            f"{names.prompts} = []",
            f"{names.responses} = None",
            f"{names.final_target} = None",
            f"{names.final_values} = {{}}",
            f"{names.gather_error} = {names.gather_position} = None",
            f"{names.gather_failed} = False",
            f"{names.frame} = {names.frame_index} = None",
        ]
        cleanup = f"del {', '.join(names.cleanup_names())}"
        return "\n".join([*initializers, _try_finally_source("\n".join(gather), cleanup)])

    def _wrap_runtime_guards(self, compiled: str) -> str:
        plan = self.plan
        if not (plan.runtime_aliases or plan.runtime_callbacks or plan.runtime_receivers):
            return compiled
        alias_ids = self.program.fresh_name("__dspy_alias_ids")
        checks = [
            *self._runtime_alias_checks(alias_ids),
            *self._runtime_callback_checks(alias_ids),
            *self._runtime_receiver_checks(alias_ids),
        ]
        guarded_body = _conditional_source(" or ".join(checks), self.original_loop, compiled)
        return "\n".join([self._alias_helper_source(alias_ids), _try_finally_source(guarded_body, f"del {alias_ids}")])

    def _runtime_alias_checks(self, alias_ids: str) -> list[str]:
        checks = []
        for left, right in self.plan.runtime_aliases:
            left_exists, left_value = self.program.guarded_name(left, self.plan.node)
            right_exists, right_value = self.program.guarded_name(right, self.plan.node)
            checks.append(
                f"({left_exists} and {right_exists} and {alias_ids}({left_value})[0] & {alias_ids}({right_value})[0])"
            )
        return checks

    def _runtime_callback_checks(self, alias_ids: str) -> list[str]:
        checks = []
        for name in self.plan.runtime_callbacks:
            exists, value = self.program.guarded_name(name, self.plan.node)
            checks.append(f"({exists} and ({alias_ids}({value})[1] or not {alias_ids}({value})[2]))")
        return checks

    def _runtime_receiver_checks(self, alias_ids: str) -> list[str]:
        checks = []
        for name in self.plan.runtime_receivers:
            exists, value = self.program.guarded_name(name, self.plan.node)
            checks.append(f"({exists} and not {alias_ids}({value})[2])")
        return checks

    @staticmethod
    def _alias_helper_source(alias_ids: str) -> str:
        return "\n".join(
            [
                f"def {alias_ids}(value):",
                "    stack, seen, mutable, callback, native = [value], set(), set(), False, True",
                "    while stack:",
                "        value = stack.pop()",
                "        if id(value) in seen:",
                "            continue",
                "        seen.add(id(value))",
                "        callback = callback or callable(value)",
                "        if type(value) in (dict, list, tuple, set):",
                "            if type(value) is not tuple:",
                "                mutable.add(id(value))",
                "            stack.extend([*value.keys(), *value.values()] if type(value) is dict else value)",
                "        elif type(value) not in (str, int, float, bool, type(None)):",
                "            native = False",
                "    return mutable, callback, native",
            ]
        )

    def _wrap_instrumentation_guard(self, compiled: str) -> str:
        system, gettrace, getprofile = (
            self.program.fresh_name(f"__dspy_{name}") for name in ("sys", "gettrace", "getprofile")
        )
        trusted = " and ".join(
            [
                f"type({gettrace}) is type(len)",
                f"{gettrace}.__self__ is {system}",
                f"{gettrace}.__module__ == 'sys'",
                f"{gettrace}.__name__ == 'gettrace'",
                f"type({getprofile}) is type(len)",
                f"{getprofile}.__self__ is {system}",
                f"{getprofile}.__module__ == 'sys'",
                f"{getprofile}.__name__ == 'getprofile'",
            ]
        )
        guard = f"not ({trusted}) or {gettrace}() is not None or {getprofile}() is not None"
        instrumented = _conditional_source(guard, self.original_loop, compiled)
        setup = "\n".join(
            [
                f"{system} = __import__('sys')",
                f"{gettrace}, {getprofile} = {system}.gettrace, {system}.getprofile",
            ]
        )
        cleanup = f"del {system}, {gettrace}, {getprofile}"
        return "\n".join([setup, _try_finally_source(instrumented, cleanup)])

    def _wrap_collision_guard(self, compiled: str) -> str:
        generated_names = sorted(self.program.used_names - self.plan.generated_start)
        collision_checks = [f"{name!r} in globals()" for name in generated_names]
        return _conditional_source(" or ".join(collision_checks), self.original_loop, compiled)


class QueryBatchTransformer(ast.NodeTransformer):
    """Replace each eligible loop without descending into a successful rewrite."""

    def __init__(self, program: ProgramAnalysis):
        self.program = program
        self.rewrite_count = 0

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        return node

    def visit_For(self, node: ast.For) -> ast.AST | list[ast.stmt]:
        plan = LoopAnalyzer(self.program, node).analyze()
        if plan is None:
            return self.generic_visit(node)
        replacement = LoopLowerer(self.program, plan).lower()
        if replacement is None:
            return self.generic_visit(node)
        self.rewrite_count += 1
        return replacement


def compile_llm_query_loops(code: str) -> tuple[str, int]:
    """Split independent query loops into prompt-gather and ordered replay stages."""
    try:
        program = ProgramAnalysis.from_code(code)
    except SyntaxError:
        return code, 0
    if not program.is_safe_program():
        return code, 0
    program.prepare()
    transformer = QueryBatchTransformer(program)
    transformed = transformer.visit(program.tree)
    if not transformer.rewrite_count:
        return code, 0
    return ast.unparse(ast.fix_missing_locations(transformed)), transformer.rewrite_count
