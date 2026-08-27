"""Compiler for batching independent ``llm_query`` loops."""

import ast
import symtable


def compile_llm_query_loops(code: str) -> tuple[str, int]:
    """Split independent query loops into prompt-gather and ordered replay stages."""
    try:
        tree = ast.parse(code)
        tables = [symtable.symtable(code, "<rlm>", "exec")]
    except SyntaxError:
        return code, 0
    pure_functions = set(
        "bool chr dict enumerate float format int len list max min range repr set sorted str sum tuple zip".split()
    )
    string_methods, collection_methods = (
        frozenset("format join lower lstrip replace rstrip strip upper".split()),
        frozenset("add append extend insert setdefault update".split()),
    )
    pure_modules = {
        alias.asname or "json"
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "json"
    }
    for table in tables:
        tables.extend(table.get_children())
    symbols = [symbol for table in tables for symbol in table.get_symbols()]
    used_names = {symbol.get_name() for symbol in symbols}
    shadowed_names = {symbol.get_name() for symbol in symbols if symbol.is_assigned() or symbol.is_parameter()}
    imported_names = {symbol.get_name() for symbol in symbols if symbol.is_imported()}
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    reserved_runtime_names = {
        "llm_query",
        "llm_query_batched",
        "__dspy_llm_query_batched",
        "__dspy_replay_llm_query",
        "print",
    }
    reflective_modules = {"builtins", "gc", "importlib", "inspect", "operator"}
    dangerous_attributes = {
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
    dangerous_names = {
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
    dangerous_constants = {
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

    def is_namespace_membership(node: ast.Name) -> bool:
        call = parents.get(node)
        comparison = parents.get(call)
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

    def is_unsafe_node(node: ast.AST) -> bool:
        if isinstance(node, (ast.ClassDef, ast.Delete, ast.Match)):
            return True
        if isinstance(node, ast.Import):
            return any(alias.name.split(".", 1)[0] in reflective_modules for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            return (
                node.module is None
                or node.module.split(".", 1)[0] in reflective_modules
                or any(alias.name == "*" or alias.name in {"_getframe", "modules"} for alias in node.names)
            )
        if isinstance(node, ast.Attribute):
            return node.attr in dangerous_attributes or (
                isinstance(node.value, ast.Name) and node.value.id in pure_functions | {"print"}
            )
        if isinstance(node, ast.Name):
            if node.id in dangerous_names or node.id.startswith("__dspy_"):
                return True
            return node.id in {"globals", "locals", "vars"} and (
                node.id in shadowed_names | imported_names or not is_namespace_membership(node)
            )
        return isinstance(node, ast.Constant) and node.value in dangerous_constants

    if reserved_runtime_names & (shadowed_names | imported_names) or any(
        is_unsafe_node(node) for node in ast.walk(tree)
    ):
        return code, 0
    pure_functions.difference_update(shadowed_names | imported_names)
    pure_modules.difference_update(
        shadowed_names
        | {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        | {
            alias.asname or alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name != "json"
        }
    )
    rewrite_count = 0

    def fresh_name(prefix: str) -> str:
        name, suffix = prefix, 0
        while name in used_names:
            suffix += 1
            name = f"{prefix}_{suffix}"
        used_names.add(name)
        return name

    def is_query_call(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "llm_query"
            and len(node.args) == 1
            and not node.keywords
        )

    def bound_names(node: ast.AST) -> list[str] | None:
        if isinstance(node, ast.Name):
            return [node.id]
        return (
            [value.id for value in node.elts]
            if isinstance(node, (ast.Tuple, ast.List)) and all(isinstance(value, ast.Name) for value in node.elts)
            else None
        )

    def names(node: ast.AST, context: type[ast.expr_context]) -> set[str]:
        return {
            child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, context)
        }

    def stored_names(node: ast.AST) -> set[str]:
        return names(node, ast.Store) | {
            child.name
            for child in ast.walk(node)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            or (isinstance(child, ast.ExceptHandler) and child.name)
        }

    def loaded_names(node: ast.AST) -> set[str]:
        return names(node, ast.Load) - names(node, ast.Store) | {
            load.id
            for expression in ast.walk(node)
            if isinstance(expression, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp))
            for load in ast.walk(node)
            if isinstance(load, ast.Name)
            and isinstance(load.ctx, ast.Load)
            and load.id in set().union(*(names(generator.target, ast.Store) for generator in expression.generators))
            and (
                load not in set(ast.walk(expression))
                or any(load in set(ast.walk(generator.iter)) for generator in expression.generators)
            )
        }

    def root_name(node: ast.AST) -> str | None:
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    assignments = [(node.targets, node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign)] + [
        ([node.target], node.value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None
    ]
    alias_pairs = [
        {left, right}
        for targets, value in assignments
        if isinstance(
            value,
            (
                ast.Name,
                ast.Attribute,
                ast.Subscript,
                ast.Tuple,
                ast.List,
                ast.BinOp,
                ast.BoolOp,
                ast.IfExp,
                ast.Dict,
            ),
        )
        or (
            isinstance(value, ast.Call)
            and not is_query_call(value)
            and (
                not isinstance(value.func, ast.Name)
                or value.func.id
                not in {"bool", "chr", "float", "format", "int", "len", "range", "repr", "str", "sum"}
            )
            and (
                not isinstance(value.func, ast.Attribute) or value.func.attr in {"copy", "get", "pop", "setdefault"}
            )
        )
        for target in targets
        for left in names(target, ast.Store) | ({root_name(target)} - {None})
        for right in loaded_names(value)
    ]
    uncertain_bindings = [
        ([node.target], node.iter)
        for node in ast.walk(tree)
        if isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(node.iter, (ast.List, ast.Tuple))
    ] + [
        ([item.optional_vars], item.context_expr)
        for node in ast.walk(tree)
        if isinstance(node, (ast.With, ast.AsyncWith))
        for item in node.items
        if item.optional_vars
    ]
    uncertain_alias_pairs = [
        {left, right}
        for targets, value in uncertain_bindings
        for target in targets
        for left in names(target, ast.Store) | ({root_name(target)} - {None})
        for right in loaded_names(value)
    ]
    generator_functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(isinstance(child, (ast.Yield, ast.YieldFrom)) for child in ast.walk(node))
    }
    one_shot_names = {
        left
        for targets, value in assignments
        if isinstance(value, ast.GeneratorExp)
        or (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in generator_functions | {"enumerate", "filter", "iter", "map", "zip"}
        )
        for target in targets
        for left in names(target, ast.Store)
    }
    pure_blocked, builder_blocked = (
        (ast.Await, ast.Lambda, ast.NamedExpr, ast.Yield, ast.YieldFrom),
        (
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
        ),
    )

    def has_callback_options(node: ast.AST) -> bool:
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
                    or (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id in pure_modules
                        and node.func.attr in {"dumps", "loads"}
                        and any(
                            keyword.arg
                            in {
                                None,
                                "cls",
                                "default",
                                "object_hook",
                                "object_pairs_hook",
                                "parse_constant",
                                "parse_float",
                                "parse_int",
                            }
                            for keyword in node.keywords
                        )
                    )
                )
            )
        )

    def allowed_call(node: ast.AST, mutable: set[str] = frozenset()) -> bool:
        return not isinstance(node, ast.Call) or (
            (isinstance(node.func, ast.Name) and node.func.id in pure_functions and not has_callback_options(node))
            or (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in pure_modules
                and node.func.attr in {"dumps", "loads"}
                and not has_callback_options(node)
            )
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in string_methods
                and (
                    isinstance(node.func.value, (ast.Constant, ast.JoinedStr))
                    or (
                        isinstance(node.func.value, ast.Call)
                        and isinstance(node.func.value.func, ast.Name)
                        and node.func.value.func.id in {"chr", "format", "str"}
                    )
                )
            )
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in collection_methods
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in mutable
            )
        )

    def is_pure(node: ast.AST) -> bool:
        return all(
            not isinstance(child, pure_blocked)
            and not (isinstance(child, ast.comprehension) and child.is_async)
            and allowed_call(child)
            for child in ast.walk(node)
        )

    def query_runs_first(statement: ast.stmt, query: ast.Call, loop: ast.For) -> bool:
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
                and is_owned_name(value.func.value.id, loop)
            )
        )

    def mutation_roots(node: ast.AST) -> set[str]:
        children = list(ast.walk(node))
        roots = {root_name(child.target) for child in children if isinstance(child, ast.AugAssign)}
        roots.update(
            root_name(child)
            for child in children
            if isinstance(child, (ast.Attribute, ast.Subscript)) and isinstance(child.ctx, ast.Store)
        )
        roots.update(
            root_name(child.func.value)
            for child in children
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr in collection_methods
        )
        return roots - {None}

    def is_owned_value(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Constant)
            or (
                isinstance(node, (ast.List, ast.Set, ast.Tuple))
                and all(is_owned_value(value) for value in node.elts)
            )
            or (
                isinstance(node, ast.Dict)
                and all(
                    key is not None and is_owned_value(key) and is_owned_value(value)
                    for key, value in zip(node.keys, node.values, strict=True)
                )
            )
            or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in pure_functions & {"dict", "list", "set", "str", "tuple"}
                and not node.args
                and not node.keywords
            )
        )

    def precedes(statement: ast.stmt, node: ast.AST) -> bool:
        return (parent := parents.get(node)) is not None and (
            any(
                statement in values[: values.index(node)]
                for _, values in ast.iter_fields(parent)
                if isinstance(values, list) and node in values
            )
            or precedes(statement, parent)
        )

    def is_owned_name(name: str, loop: ast.For) -> bool:
        return (
            name not in imported_names
            and sum(
                (isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store) and child.id == name)
                or (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child.name == name)
                or (isinstance(child, ast.ExceptHandler) and child.name == name)
                for child in ast.walk(tree)
            )
            == 1
            and any(
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == name
                and isinstance(statement.value, ast.List)
                and not statement.value.elts
                and precedes(statement, loop)
                for statement in ast.walk(tree)
            )
        )

    def enclosing_function(node: ast.AST) -> ast.AST | None:
        return (
            parent
            if isinstance((parent := parents.get(node)), (ast.FunctionDef, ast.AsyncFunctionDef))
            else enclosing_function(parent)
            if parent
            else None
        )

    def guarded_name(name: str, node: ast.AST) -> tuple[str, str]:
        scope = enclosing_function(node)
        is_local = any(
            scope is not None
            and table.get_type() == "function"
            and table.get_name() == scope.name
            and table.get_lineno() == scope.lineno
            and any(
                symbol.get_name() == name and (symbol.is_local() or symbol.is_parameter())
                for symbol in table.get_symbols()
            )
            for table in tables
        )
        if is_local:
            return f"{name!r} in locals()", f"locals()[{name!r}]"
        return (
            f"({name!r} in locals() or {name!r} in globals())",
            f"(locals() if {name!r} in locals() else globals())[{name!r}]",
        )

    def is_local_builder(node: ast.AST, roots: set[str]) -> bool:
        return all(
            not isinstance(child, builder_blocked)
            and allowed_call(child, roots)
            and not (
                isinstance(child, (ast.Attribute, ast.Subscript))
                and isinstance(child.ctx, ast.Store)
                and root_name(child) not in roots
            )
            and not (isinstance(child, ast.AugAssign) and not isinstance(child.target, ast.Name))
            and not (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr in collection_methods - {"append"}
            )
            and not (
                isinstance(child, (ast.For, ast.AsyncFor))
                and loaded_names(child.iter) & with_aliases(one_shot_names)
            )
            for child in ast.walk(node)
        )

    def with_aliases(values: set[str], pairs: list[set[str]] = alias_pairs + uncertain_alias_pairs) -> set[str]:
        expanded = set(values)
        while any(pair & expanded and not pair <= expanded for pair in pairs):
            for pair in pairs:
                if pair & expanded:
                    expanded.update(pair)
        return expanded

    replay_functions = set()
    unsafe_callbacks = with_aliases(
        (
            {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            | {
                left
                for targets, value in assignments
                if isinstance(value, ast.Lambda)
                for target in targets
                for left in names(target, ast.Store)
            }
        )
        - replay_functions,
        [
            {left, *loaded_names(value)}
            for targets, value in assignments
            for target in targets
            for left in names(target, ast.Store)
        ],
    )
    forbidden_query_parents = (
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

    def statement_queries(statement: ast.stmt) -> list[ast.Call] | None:
        queries: list[ast.Call] = []

        def visit(node: ast.AST, conditional: bool = False) -> bool:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "llm_query":
                if conditional or not is_query_call(node):
                    return False
                queries.append(node)
            child_conditional = conditional or isinstance(node, forbidden_query_parents)
            return all(visit(child, child_conditional) for child in ast.iter_child_nodes(node))

        return queries if visit(statement) else None

    def indent_source(source: str, depth: int = 1) -> str:
        return "\n".join("    " * depth + line for line in source.splitlines())

    def indent(node: ast.AST, depth: int = 1) -> str:
        return indent_source(ast.unparse(node), depth)

    def conditional_source(condition: str, true_body: str, false_body: str) -> str:
        return "\n".join(
            [
                f"if {condition}:",
                indent_source(true_body),
                "else:",
                indent_source(false_body),
            ]
        )

    def try_finally_source(body: str, cleanup: str) -> str:
        return "\n".join(["try:", indent_source(body), "finally:", indent_source(cleanup)])

    class QueryBatchTransformer(ast.NodeTransformer):
        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
            return node

        def visit_For(self, node: ast.For) -> ast.AST | list[ast.stmt]:
            nonlocal rewrite_count
            if (replacement := self._rewrite_for(node)) is None:
                return self.generic_visit(node)
            rewrite_count += 1
            return replacement

        def _rewrite_for(self, node: ast.For) -> list[ast.stmt] | None:
            generated_start = set(used_names)
            if (
                node.orelse
                or not is_pure(node.iter)
                or any(
                    isinstance(
                        child,
                        (
                            ast.Await,
                            ast.AsyncWith,
                            ast.Import,
                            ast.ImportFrom,
                            ast.Return,
                            ast.With,
                            ast.Yield,
                            ast.YieldFrom,
                        ),
                    )
                    or (
                        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and (
                            child.decorator_list
                            or child.returns
                            or child.args.defaults
                            or any(child.args.kw_defaults)
                            or any(
                                argument.annotation
                                for argument in [
                                    *child.args.posonlyargs,
                                    *child.args.args,
                                    child.args.vararg,
                                    *child.args.kwonlyargs,
                                    child.args.kwarg,
                                ]
                                if argument
                            )
                            or getattr(child, "type_params", [])
                        )
                    )
                    or (isinstance(child, ast.Lambda) and (child.args.defaults or any(child.args.kw_defaults)))
                    for child in ast.walk(node)
                )
            ):
                return None
            loop_names, parameter_aliases = (
                bound_names(node.target),
                with_aliases(
                    {
                        arg.arg
                        for function in [enclosing_function(node)]
                        if function is not None
                        for arg in ast.walk(function.args)
                        if isinstance(arg, ast.arg)
                    }
                ),
            )
            if (
                not loop_names
                or len(loop_names) != len(set(loop_names))
                or set(loop_names) & set().union(*(stored_names(statement) for statement in node.body))
            ):
                return None
            query_statements: dict[int, list[ast.Call]] = {}
            for index, statement in enumerate(node.body):
                queries = statement_queries(statement)
                if queries is None or len(queries) > 1:
                    return None
                if queries:
                    if not query_runs_first(statement, queries[0], node):
                        return None
                    query_statements[index] = queries
            if not query_statements:
                return None
            first_query, last_query = min(query_statements), max(query_statements)
            guards, prequery_prints = set(), {}
            for index, statement in enumerate(node.body):
                breaks = any(isinstance(child, ast.Break) for child in ast.walk(statement))
                continues = any(isinstance(child, ast.Continue) for child in ast.walk(statement))
                if not breaks and not continues:
                    continue
                simple_guard = (
                    index < first_query
                    and isinstance(statement, ast.If)
                    and not statement.orelse
                    and len(statement.body) == 1
                    and isinstance(statement.body[0], (ast.Break, ast.Continue))
                    and is_pure(statement.test)
                )
                if simple_guard:
                    guards.add(index)
                elif breaks or index <= last_query:
                    return None
            stores = [stored_names(statement) for statement in node.body]
            candidates = {
                index: (statement.targets[0].id, statement.value)
                for index, statement in enumerate(node.body)
                if isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and index not in query_statements
                and is_pure(statement.value)
            }
            selected, resolving = set(), set()

            def resolve(name: str, before: int) -> bool:
                for index in range(before - 1, -1, -1):
                    if name not in stores[index]:
                        continue
                    if name in mutation_roots(node.body[index]):
                        continue
                    if index not in candidates or index in resolving:
                        return False
                    if index in selected:
                        return True
                    resolving.add(index)
                    if not all(resolve(dependency, index) for dependency in loaded_names(candidates[index][1])):
                        return False
                    resolving.remove(index)
                    selected.add(index)
                    return True
                return name in loop_names or name not in set().union(*stores)

            dependencies = loaded_names(node.iter)
            prompt_inputs, guard_inputs = (
                [(index, query.args[0]) for index, queries in query_statements.items() for query in queries],
                [(index, node.body[index].test) for index in guards],
            )
            for index, expression in [*guard_inputs, *prompt_inputs]:
                expression_names = loaded_names(expression)
                dependencies.update(expression_names)
                if not is_pure(expression) or not all(resolve(name, index) for name in expression_names):
                    return None
            duplicated: set[int] = set()
            selected_by_name = {candidates[index][0]: index for index in selected}
            for index in range(first_query):
                roots = mutation_roots(node.body[index])
                if not roots:
                    continue
                initializers = {selected_by_name.get(root) for root in roots}
                bad_initializer = (
                    None in initializers
                    or any(
                        initializer >= index or not is_owned_value(candidates[initializer][1])
                        for initializer in initializers
                    )
                    or (
                        any(isinstance(child, ast.AugAssign) for child in ast.walk(node.body[index]))
                        and any(
                            not isinstance(candidates[initializer][1], ast.Constant) for initializer in initializers
                        )
                    )
                )
                if bad_initializer or not is_local_builder(node.body[index], roots):
                    return None
                duplicated.update({index, *initializers})
            for index in duplicated:
                if not all(resolve(name, index) for name in loaded_names(node.body[index])):
                    return None
            if guards and duplicated and max(guards) > min(duplicated):
                return None
            selected.difference_update(duplicated)
            selected_names = [candidates[index][0] for index in selected]
            if set(loop_names) & set(selected_names) or len(selected_names) != len(set(selected_names)):
                return None
            selected_positions = {candidates[index][0]: index for index in selected}
            if any(
                position > index and name in loaded_names(statement)
                for index, statement in enumerate(node.body)
                for name, position in selected_positions.items()
            ):
                return None
            dependencies.update(
                selected_names,
                *(loaded_names(candidates[index][1]) for index in selected),
                *(loaded_names(node.body[index]) for index in duplicated),
            )
            for index in range(last_query + 1):
                if index in selected | duplicated | guards | set(query_statements):
                    continue
                statement = node.body[index]
                is_print = (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Name)
                    and statement.value.func.id == "print"
                )
                print_values = (
                    [*statement.value.args, *(keyword.value for keyword in statement.value.keywords)]
                    if is_print
                    else []
                )
                safe_print_names = (
                    set(loop_names)
                    | loaded_names(node.iter)
                    | set(selected_names)
                    | set().union(*(mutation_roots(node.body[position]) for position in duplicated))
                    | pure_functions
                    | {"print"}
                )
                if (
                    stored_names(statement)
                    or not is_print
                    or (
                        index < first_query
                        and (
                            not all(is_pure(value) for value in print_values)
                            or loaded_names(statement) - safe_print_names
                            or any(
                                isinstance(value, ast.Starred)
                                or any(
                                    isinstance(child, ast.GeneratorExp)
                                    or (
                                        isinstance(child, ast.Call)
                                        and isinstance(child.func, ast.Name)
                                        and child.func.id in {"enumerate", "zip"}
                                    )
                                    for child in ast.walk(value)
                                )
                                for value in statement.value.args
                            )
                            or statement.value.keywords
                        )
                    )
                ):
                    return None
                prequery_prints.update(
                    {index: ", ".join(ast.unparse(value) for value in print_values)}
                    if index < first_query and print_values
                    else {}
                )
            mutations, runtime_mutations, callback_roots, receiver_roots = set(), set(), set(), set()
            for index, statement in enumerate(node.body):
                if index in selected | duplicated:
                    continue
                statement_mutations = mutation_roots(statement)
                mutations.update(stored_names(statement), statement_mutations)
                runtime_mutations.update(statement_mutations)
                if index > last_query:
                    prior_stores = set().union(
                        *(
                            names(target, ast.Store)
                            for position in range(index)
                            for target in (
                                node.body[position].targets
                                if isinstance(node.body[position], ast.Assign)
                                else [node.body[position].target]
                                if isinstance(node.body[position], (ast.AnnAssign, ast.AugAssign))
                                else []
                            )
                        )
                    )
                    receiver_roots.update(
                        (
                            loaded_names(statement)
                            | (names(statement, ast.Load) & names(statement, ast.Store))
                            | statement_mutations
                        )
                        - set(loop_names)
                        - prior_stores
                    )
                for call in (
                    child
                    for child in ast.walk(statement)
                    if isinstance(child, ast.Call) and not is_query_call(child)
                ):
                    known = allowed_call(call, mutation_roots(statement)) or (
                        isinstance(call.func, ast.Name) and call.func.id == "print"
                    )
                    if not known and index <= last_query:
                        return None
                    if known:
                        continue
                    if (
                        has_callback_options(call)
                        or (isinstance(call.func, ast.Name) and call.func.id not in replay_functions)
                        or any(
                            with_aliases(loaded_names(value)) & (unsafe_callbacks | parameter_aliases)
                            for value in [*call.args, *(keyword.value for keyword in call.keywords)]
                        )
                    ):
                        return None
                    argument_values = [*call.args, *(keyword.value for keyword in call.keywords)]
                    argument_roots = [loaded_names(value) for value in argument_values]
                    call_receiver_roots = (
                        [loaded_names(call.func.value)] if isinstance(call.func, ast.Attribute) else []
                    )
                    mutations.update(*argument_roots, *call_receiver_roots)
                    runtime_mutations.update(*argument_roots, *call_receiver_roots)
                    callback_values = [
                        *argument_values,
                        *([call.func] if not isinstance(call.func, (ast.Name, ast.Attribute)) else []),
                    ]
                    callback_roots.update(*(loaded_names(value) for value in callback_values))
                    receiver_roots.update(*call_receiver_roots)
            dependency_aliases, mutation_aliases = with_aliases(dependencies), with_aliases(mutations)
            if (
                set(loop_names) & mutations
                or dependency_aliases & mutation_aliases
                or any(pair & dependency_aliases and pair & mutation_aliases for pair in uncertain_alias_pairs)
            ):
                return None
            runtime_aliases, runtime_callbacks, runtime_receivers = (
                [
                    (left, right)
                    for left in dependency_aliases - pure_functions - {"llm_query", "print"}
                    for right in with_aliases(runtime_mutations) - pure_functions - {"llm_query", "print"}
                    if left != right
                ],
                with_aliases(callback_roots) - pure_functions - replay_functions - {"llm_query", "print"},
                with_aliases(receiver_roots | (dependencies & pure_modules))
                - pure_functions
                - replay_functions
                - {"llm_query", "print"},
            )
            selected_order, target_source, replay_target = (
                sorted(selected),
                ast.unparse(node.target),
                ", ".join(loop_names),
            )
            temp_names = (
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
            (
                frames,
                prompts,
                responses,
                frame,
                frame_index,
                final_target,
                final_values,
                gather_error,
                gather_position,
                gather_failed,
            ) = (fresh_name(f"__dspy_{name}") for name in temp_names)
            frame_error = fresh_name("__dspy_frame_error")
            gather = [
                "try:",
                f"    for {target_source} in {ast.unparse(node.iter)}:",
                f"        {final_target} = [{', '.join(loop_names)}]",
                f"        {frame} = [{', '.join([*loop_names, *(['None'] * len(selected_order))])}]",
                "        try:",
            ]
            for index, statement in enumerate(node.body):
                if index in selected | duplicated | guards | set(query_statements) | set(prequery_prints):
                    gather.append(
                        f"            {gather_position} = {index}"
                        + (f"\n            ({prequery_prints[index]},)" if index in prequery_prints else "")
                    )
                if index in selected | duplicated:
                    gather.append(indent(statement, 3))
                if index in selected:
                    offset = len(loop_names) + selected_order.index(index)
                    gather.extend(
                        [
                            f"            {frame}[{offset}] = {candidates[index][0]}",
                            f"            {final_values}[{index}] = {candidates[index][0]}",
                        ]
                    )
                elif index in guards:
                    gather.append(indent(statement, 3))
                for query in query_statements.get(index, []):
                    gather.append(f"            {prompts}.append({ast.unparse(query.args[0])})")
            gather.extend(
                [
                    "        except Exception:",
                    f"            {gather_failed} = True",
                    f"            {frames}.append({frame})",
                    "            raise",
                    f"        {frames}.append({frame})",
                    f"except Exception as {frame_error}:",
                    f"    {gather_error} = {frame_error}",
                    f"{responses} = __dspy_llm_query_batched({prompts})",
                    f"for {frame_index}, {frame} in enumerate({frames}):",
                ]
            )
            gather.append(
                f"    {replay_target} = {', '.join(f'{frame}[{index}]' for index in range(len(loop_names)))}"
            )
            original_loop = ast.unparse(node)
            query_index = 0

            class ResponseReplacer(ast.NodeTransformer):
                def visit_Call(self, call: ast.Call) -> ast.AST:
                    nonlocal query_index
                    if not is_query_call(call):
                        return self.generic_visit(call)
                    expression = ast.parse(
                        f"__dspy_replay_llm_query({responses}[{frame_index} * {len(query_statements)} + {query_index}])",
                        mode="eval",
                    ).body
                    query_index += 1
                    return ast.copy_location(expression, call)

            replayed_body = [ResponseReplacer().visit(statement) for statement in node.body]
            if query_index != len(query_statements):
                return None
            gather.append(f"    if {gather_failed} and {frame_index} == len({frames}) - 1:")
            for index, statement in enumerate(replayed_body):
                comparison = ">" if index in query_statements else ">="
                if index in query_statements:
                    gather.extend(
                        [f"        if {gather_position} == {index}:", f"            raise {gather_error}"]
                    )
                gather.append(f"        if {gather_position} {comparison} {index}:")
                gather.append(indent(statement, 3))
            gather.extend([f"        raise {gather_error}", "    else:"])
            for index, statement in enumerate(replayed_body):
                gather.append(
                    f"        {candidates[index][0]} = {frame}[{len(loop_names) + selected_order.index(index)}]"
                    if index in selected
                    else indent(statement, 2)
                )
            gather.extend(
                [
                    f"if {final_target} is not None:",
                    f"    {replay_target} = {', '.join(f'{final_target}[{index}]' for index in range(len(loop_names)))}",
                ]
            )
            for index in selected_order:
                gather.extend(
                    [f"if {index} in {final_values}:", f"    {candidates[index][0]} = {final_values}[{index}]"]
                )
            gather.extend([f"if {gather_error} is not None:", f"    raise {gather_error}"])
            initializers = [
                f"{frames} = []",
                f"{prompts} = []",
                f"{responses} = None",
                f"{final_target} = None",
                f"{final_values} = {{}}",
                f"{gather_error} = {gather_position} = None",
                f"{gather_failed} = False",
                f"{frame} = {frame_index} = None",
            ]
            cleanup_names = [
                frames,
                prompts,
                responses,
                frame,
                frame_index,
                final_target,
                final_values,
                gather_error,
                gather_position,
                gather_failed,
            ]
            compiled = "\n".join(
                [
                    *initializers,
                    try_finally_source("\n".join(gather), f"del {', '.join(cleanup_names)}"),
                ]
            )
            if runtime_aliases or runtime_callbacks or runtime_receivers:
                alias_ids = fresh_name("__dspy_alias_ids")
                checks = []
                for left, right in runtime_aliases:
                    left_exists, left_value = guarded_name(left, node)
                    right_exists, right_value = guarded_name(right, node)
                    checks.append(
                        f"({left_exists} and {right_exists} and "
                        f"{alias_ids}({left_value})[0] & {alias_ids}({right_value})[0])"
                    )
                for name in runtime_callbacks:
                    exists, value = guarded_name(name, node)
                    checks.append(f"({exists} and ({alias_ids}({value})[1] or not {alias_ids}({value})[2]))")
                for name in runtime_receivers:
                    exists, value = guarded_name(name, node)
                    checks.append(f"({exists} and not {alias_ids}({value})[2])")

                alias_helper = "\n".join(
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
                guarded_body = conditional_source(" or ".join(checks), original_loop, compiled)
                compiled = "\n".join(
                    [
                        alias_helper,
                        try_finally_source(guarded_body, f"del {alias_ids}"),
                    ]
                )
            system, gettrace, getprofile = (
                fresh_name(f"__dspy_{name}") for name in ("sys", "gettrace", "getprofile")
            )
            trusted_instrumentation = " and ".join(
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
            instrumentation_guard = (
                f"not ({trusted_instrumentation}) or {gettrace}() is not None or {getprofile}() is not None"
            )
            instrumented_body = conditional_source(instrumentation_guard, original_loop, compiled)
            instrumentation_setup = "\n".join(
                [
                    f"{system} = __import__('sys')",
                    f"{gettrace}, {getprofile} = {system}.gettrace, {system}.getprofile",
                ]
            )
            compiled = "\n".join(
                [
                    instrumentation_setup,
                    try_finally_source(instrumented_body, f"del {system}, {gettrace}, {getprofile}"),
                ]
            )

            collision_checks = [f"{name!r} in globals()" for name in sorted(used_names - generated_start)]
            compiled = conditional_source(" or ".join(collision_checks), original_loop, compiled)
            return ast.parse(compiled).body

    transformed = QueryBatchTransformer().visit(tree)
    if not rewrite_count:
        return code, 0
    return ast.unparse(ast.fix_missing_locations(transformed)), rewrite_count
