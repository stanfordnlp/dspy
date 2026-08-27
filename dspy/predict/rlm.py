"""
Recursive Language Model (RLM) module for DSPy.

RLMs are an inference strategy where LLMs treat long contexts as part of an external
environment rather than feeding them directly to the model. The LLM writes Python code
to programmatically examine, decompose, and recursively call sub-LLMs over snippets.

Reference: "Recursive Language Models" (Zhang, Kraska, Khattab, 2025)
"""

from __future__ import annotations

import ast
import base64
import contextvars
import functools
import inspect
import keyword
import logging
import symtable
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator

import pydantic

import dspy
from dspy.adapters.types.tool import Tool
from dspy.adapters.utils import parse_value, translate_field_type
from dspy.primitives.code_interpreter import (
    SIMPLE_TYPES,
    CodeExecutionError,
    CodeInterpreter,
    FinalOutput,
    _create_interpreter,
    _validate_interpreter,
    _validate_interpreter_factory,
)
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.primitives.repl_types import REPLEntry, REPLHistory, REPLVariable
from dspy.primitives.sandbox_serializable import SandboxSerializable, build_repl_variable
from dspy.signatures.signature import ensure_signature
from dspy.utils.annotation import experimental
from dspy.utils.exceptions import format_error_for_lm

if TYPE_CHECKING:

    from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)

# TODO: Optimize this prompt across a diverse benchmark

ACTION_INSTRUCTIONS_TEMPLATE = """You are tasked with producing the following outputs given the inputs {inputs}:
{output_fields}

You have access to a Python REPL environment. Write Python code and it will be executed. You will see the output, then write more code based on what you learned. This is an iterative process.
{interpreter_rules}
Available:
- Variables: {inputs} (your input data)
- `llm_query(prompt)` queries one sub-LLM (~500K char capacity); `llm_query_batched(prompts)` queries independent prompts concurrently and preserves input order
- `print()` - ALWAYS print to see results
- `SUBMIT({final_output_names})` - submit final output when done

IMPORTANT: This is ITERATIVE. Each code block you write will execute, you'll see the output, then you decide what to do next. Do NOT try to solve everything in one step.

1. EXPLORE FIRST - Look at your data before processing it. Print samples, check types/lengths, understand the structure.
2. ITERATE - Write small code snippets, observe outputs, then decide next steps. State persists between iterations.
3. VERIFY BEFORE SUBMITTING - If results seem wrong (zeros, empty, unexpected), reconsider your approach.
4. USE llm_query FREQUENTLY FOR SMALL SEMANTIC EXPLORATIONS - String matching finds WHERE things are; focused llm_query calls understand WHAT things mean.
5. MINIMIZE RETYPING (INPUTS & OUTPUTS) - When values are long, precise, or error-prone (IDs, numbers, code, quotes), re-access them via variables and parse/compute in code instead of retyping. Use small, targeted prints to sanity-check, but avoid manual copying when variables can carry the exact value.
6. SUBMIT ONLY AFTER SEEING OUTPUTS - SUBMIT ends the current run immediately. If you need to inspect printed output, run it in one step, review the result, then call SUBMIT in a later step.

You have max {max_llm_calls} sub-LLM calls. When done, call SUBMIT() with your output."""

_PYTHON_FENCE_LANGS = {"python", "py", "python3", "py3", ""}


def _strip_code_fences(code: str) -> str:
    """Extract Python code from markdown fences, or return as-is if no fences."""
    code = code.strip()
    if "```" not in code:
        return code

    # Strip outer decorative fence pairs (e.g. ```\n```python\n...\n```\n```)
    lines = code.splitlines()
    while len(lines) >= 2 and lines[0].strip() == "```" and lines[-1].strip() == "```":
        lines.pop(0)
        lines.pop()
    code = "\n".join(lines).strip()
    if "```" not in code:
        return code

    # Find the first opening fence (skip any text before it)
    fence_start = code.find("```")
    lang_line, separator, remainder = code[fence_start + 3:].partition("\n")
    if not separator:
        return code

    # Accept python-labeled fences or bare ``` fences; reject explicit non-Python tags
    lang = (lang_line.strip().split(maxsplit=1)[0] if lang_line.strip() else "").lower()
    if lang not in _PYTHON_FENCE_LANGS:
        raise SyntaxError(f"Expected Python code but got ```{lang} fence. Write Python code, not {lang}.")

    # Find closing fence
    block_end = remainder.find("```")
    if block_end == -1:
        return remainder.strip()

    return remainder[:block_end].strip()


@experimental
class RLM(Module):
    """Recursive Language Model module.

    Uses a sandboxed REPL to let the LLM programmatically explore large contexts
    through code execution. The LLM writes Python code to examine data, call
    sub-LLMs for semantic analysis, and build up answers iteratively.

    The default interpreter is PythonInterpreter (Deno/Pyodide/WASM), but
    ``interpreter_factory`` can create another CodeInterpreter implementation,
    such as an adapter for a remote sandbox. RLM updates the interpreter's
    mutable ``tools`` dictionary with invocation-scoped tools before execution.
    A caller-owned interpreter may be reused sequentially with the same RLM
    instance, but must not be shared by overlapping invocations.

    Examples:
        ```python
        # Basic usage
        rlm = dspy.RLM("context, query -> output", max_iters=10)
        result = rlm(context="...very long text...", query="What is the magic number?")
        print(result.output)
        ```
    """

    def __init__(
        self,
        signature: type[Signature] | str,
        max_iters: int = 20,
        max_llm_calls: int = 50,
        max_output_chars: int = 10_000,
        verbose: bool = False,
        tools: list[Callable] | None = None,
        sub_lm: dspy.LM | None = None,
        interpreter_factory: Callable[[], CodeInterpreter] = PythonInterpreter,
    ):
        """
        Args:
            signature: Defines inputs and outputs. String like "context, query -> answer"
                      or a Signature class.
            max_iters: Maximum REPL interaction iterations.
            max_llm_calls: Maximum sub-LLM calls (llm_query/llm_query_batched) per execution.
            max_output_chars: Maximum characters to include from REPL output.
            verbose: Whether to log detailed execution info.
            tools: List of tool functions or dspy.Tool objects callable from interpreter code.
                  Built-in tools: llm_query(prompt), llm_query_batched(prompts).
            sub_lm: LM for llm_query/llm_query_batched. Defaults to dspy.settings.lm.
                   Allows using a different (e.g., cheaper) model for sub-queries.
            interpreter_factory: Zero-argument callable that creates an interpreter for each forward pass. The
                callable may be invoked concurrently, and DSPy shuts down each interpreter it returns. RLM updates
                the returned interpreter's mutable ``tools`` dictionary before execution. The callable may expose
                an ``execution_instructions`` string describing its runtime for the action prompt.
        """
        super().__init__()
        _validate_interpreter_factory(interpreter_factory)
        if isinstance(signature, str) and any(field.split(":", 1)[0].strip().startswith("__dspy_") for field in signature.split("->", 1)[0].split(",")):
            raise ValueError("Input fields conflict with built-in sandbox functions")
        self.signature = ensure_signature(signature)
        self.max_iters = max_iters
        self.max_llm_calls = max_llm_calls
        self.max_output_chars = max_output_chars
        self.verbose = verbose
        self.sub_lm = sub_lm
        self._interpreter_factory = interpreter_factory
        self._user_tools = self._normalize_tools(tools)
        self._validate_namespace(self._user_tools)

        # Build the action and extract signatures
        action_sig, extract_sig = self._build_signatures()
        self.generate_action = dspy.Predict(action_sig)
        self.extract = dspy.Predict(extract_sig)

    # =========================================================================
    # Tool Creation and Validation
    # =========================================================================

    # Names owned by RLM rather than the user-provided signature or tools.
    _RESERVED_SANDBOX_NAMES = frozenset({"llm_query", "llm_query_batched", "__dspy_llm_query_batched", "__dspy_replay_llm_query", "SUBMIT", "print"})
    _RESERVED_RESULT_NAMES = frozenset({"trajectory", "final_reasoning"})

    @staticmethod
    def _compile_llm_query_loops(code: str) -> tuple[str, int]:
        """Split independent query loops into prompt-gather and ordered replay stages."""
        try:
            tree = ast.parse(code)
            tables = [symtable.symtable(code, "<rlm>", "exec")]
        except SyntaxError:
            return code, 0
        pure_functions = set("bool chr dict enumerate float format int len list max min range repr set sorted str sum tuple zip".split())
        string_methods, collection_methods = frozenset("format join lower lstrip replace rstrip strip upper".split()), frozenset("add append extend insert setdefault update".split())
        pure_modules = {alias.asname or "json" for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names if alias.name == "json"}
        for table in tables:
            tables.extend(table.get_children())
        symbols = [symbol for table in tables for symbol in table.get_symbols()]
        used_names = {symbol.get_name() for symbol in symbols}
        shadowed_names = {symbol.get_name() for symbol in symbols if symbol.is_assigned() or symbol.is_parameter()}
        imported_names = {symbol.get_name() for symbol in symbols if symbol.is_imported()}
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        if {"llm_query", "llm_query_batched", "__dspy_llm_query_batched", "__dspy_replay_llm_query", "print"} & (shadowed_names | imported_names) or any(isinstance(node, (ast.ClassDef, ast.Delete, ast.Match)) or (isinstance(node, ast.Import) and any(alias.name.split(".", 1)[0] in {"builtins", "gc", "importlib", "inspect", "operator"} for alias in node.names)) or (isinstance(node, ast.ImportFrom) and (node.module is None or node.module.split(".", 1)[0] in {"builtins", "gc", "importlib", "inspect", "operator"} or any(alias.name == "*" or alias.name in {"_getframe", "modules"} for alias in node.names))) or (isinstance(node, ast.Attribute) and (node.attr in {"__builtins__", "__code__", "__defaults__", "__delattr__", "__dict__", "__getattribute__", "__globals__", "__import__", "__kwdefaults__", "__self__", "__setattr__", "__subclasses__", "_getframe", "attrgetter", "currentframe", "delattr", "eval", "exec", "f_builtins", "f_globals", "f_locals", "get_referrers", "getattr", "getattr_static", "getmembers", "globals", "import_module", "locals", "methodcaller", "modules", "setattr", "vars"} or (isinstance(node.value, ast.Name) and node.value.id in pure_functions | {"print"}))) or (isinstance(node, ast.Name) and (node.id in {"__builtins__", "__import__", "attrgetter", "delattr", "eval", "exec", "getattr", "methodcaller", "setattr", "type"} or node.id.startswith("__dspy_") or (node.id in {"globals", "locals", "vars"} and (node.id in shadowed_names | imported_names or not (isinstance((call := parents.get(node)), ast.Call) and not call.args and not call.keywords and isinstance((comparison := parents.get(call)), ast.Compare) and len(comparison.ops) == 1 and isinstance(comparison.ops[0], (ast.In, ast.NotIn)) and comparison.comparators == [call] and isinstance(comparison.left, ast.Constant) and type(comparison.left.value) is str))))) or (isinstance(node, ast.Constant) and node.value in {"__builtins__", "__code__", "__defaults__", "__delattr__", "__dict__", "__getattribute__", "__globals__", "__import__", "__kwdefaults__", "__self__", "__setattr__", "__subclasses__", "f_builtins", "f_globals", "f_locals"}) for node in ast.walk(tree)):
            return code, 0
        pure_functions.difference_update(shadowed_names | imported_names)
        pure_modules.difference_update(shadowed_names | {alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names} | {alias.asname or alias.name.split(".", 1)[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names if alias.name != "json"})
        rewrite_count = 0
        def fresh_name(prefix: str) -> str:
            name, suffix = prefix, 0
            while name in used_names:
                suffix += 1
                name = f"{prefix}_{suffix}"
            used_names.add(name)
            return name
        def is_query_call(node: ast.AST) -> bool: return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "llm_query" and len(node.args) == 1 and not node.keywords
        def bound_names(node: ast.AST) -> list[str] | None:
            if isinstance(node, ast.Name):
                return [node.id]
            return [value.id for value in node.elts] if isinstance(node, (ast.Tuple, ast.List)) and all(isinstance(value, ast.Name) for value in node.elts) else None
        def names(node: ast.AST, context: type[ast.expr_context]) -> set[str]: return {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, context)}
        def stored_names(node: ast.AST) -> set[str]: return names(node, ast.Store) | {child.name for child in ast.walk(node) if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) or (isinstance(child, ast.ExceptHandler) and child.name)}
        def loaded_names(node: ast.AST) -> set[str]: return names(node, ast.Load) - names(node, ast.Store) | {load.id for expression in ast.walk(node) if isinstance(expression, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)) for load in ast.walk(node) if isinstance(load, ast.Name) and isinstance(load.ctx, ast.Load) and load.id in set().union(*(names(generator.target, ast.Store) for generator in expression.generators)) and (load not in set(ast.walk(expression)) or any(load in set(ast.walk(generator.iter)) for generator in expression.generators))}
        def root_name(node: ast.AST) -> str | None:
            while isinstance(node, (ast.Attribute, ast.Subscript)):
                node = node.value
            return node.id if isinstance(node, ast.Name) else None
        assignments = [(node.targets, node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign)] + [([node.target], node.value) for node in ast.walk(tree) if isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None]
        alias_pairs = [{left, right} for targets, value in assignments if isinstance(value, (ast.Name, ast.Attribute, ast.Subscript, ast.Tuple, ast.List, ast.BinOp, ast.BoolOp, ast.IfExp, ast.Dict)) or (isinstance(value, ast.Call) and not is_query_call(value) and (not isinstance(value.func, ast.Name) or value.func.id not in {"bool", "chr", "float", "format", "int", "len", "range", "repr", "str", "sum"}) and (not isinstance(value.func, ast.Attribute) or value.func.attr in {"copy", "get", "pop", "setdefault"})) for target in targets for left in names(target, ast.Store) | ({root_name(target)} - {None}) for right in loaded_names(value)]
        uncertain_bindings = [([node.target], node.iter) for node in ast.walk(tree) if isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(node.iter, (ast.List, ast.Tuple))] + [([item.optional_vars], item.context_expr) for node in ast.walk(tree) if isinstance(node, (ast.With, ast.AsyncWith)) for item in node.items if item.optional_vars]
        uncertain_alias_pairs = [{left, right} for targets, value in uncertain_bindings for target in targets for left in names(target, ast.Store) | ({root_name(target)} - {None}) for right in loaded_names(value)]
        generator_functions = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(isinstance(child, (ast.Yield, ast.YieldFrom)) for child in ast.walk(node))}
        one_shot_names = {left for targets, value in assignments if isinstance(value, ast.GeneratorExp) or (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in generator_functions | {"enumerate", "filter", "iter", "map", "zip"}) for target in targets for left in names(target, ast.Store)}
        pure_blocked, builder_blocked = (ast.Await, ast.Lambda, ast.NamedExpr, ast.Yield, ast.YieldFrom), (ast.Await, ast.Lambda, ast.NamedExpr, ast.Yield, ast.YieldFrom, ast.Break, ast.Continue, ast.Delete, ast.Global, ast.Nonlocal, ast.Raise, ast.Return, ast.Try, ast.While, ast.With)
        def has_callback_options(node: ast.AST) -> bool: return isinstance(node, ast.Call) and ((isinstance(node.func, ast.Name) and node.func.id in {"max", "min", "sorted"} and any(keyword.arg in {None, "key"} for keyword in node.keywords)) or (isinstance(node.func, ast.Attribute) and ((node.func.attr == "sort" and any(keyword.arg in {None, "key"} for keyword in node.keywords)) or (isinstance(node.func.value, ast.Name) and node.func.value.id in pure_modules and node.func.attr in {"dumps", "loads"} and any(keyword.arg in {None, "cls", "default", "object_hook", "object_pairs_hook", "parse_constant", "parse_float", "parse_int"} for keyword in node.keywords)))))
        def allowed_call(node: ast.AST, mutable: set[str] = frozenset()) -> bool:
            return not isinstance(node, ast.Call) or (
                (isinstance(node.func, ast.Name) and node.func.id in pure_functions and not has_callback_options(node))
                or (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id in pure_modules and node.func.attr in {"dumps", "loads"} and not has_callback_options(node))
                or (isinstance(node.func, ast.Attribute) and node.func.attr in string_methods and (isinstance(node.func.value, (ast.Constant, ast.JoinedStr)) or (isinstance(node.func.value, ast.Call) and isinstance(node.func.value.func, ast.Name) and node.func.value.func.id in {"chr", "format", "str"})))
                or (isinstance(node.func, ast.Attribute) and node.func.attr in collection_methods and isinstance(node.func.value, ast.Name) and node.func.value.id in mutable))
        def is_pure(node: ast.AST) -> bool: return all(not isinstance(child, pure_blocked) and not (isinstance(child, ast.comprehension) and child.is_async) and allowed_call(child) for child in ast.walk(node))
        def query_runs_first(statement: ast.stmt, query: ast.Call, loop: ast.For) -> bool: return (isinstance(statement, ast.Expr) and statement.value is query) or (isinstance(statement, ast.Assign) and statement.value is query and all(isinstance(target, ast.Name) for target in statement.targets)) or (isinstance((value := getattr(statement, "value", None)), ast.Call) and isinstance(value.func, ast.Attribute) and isinstance(value.func.value, ast.Name) and value.func.attr == "append" and value.args == [query] and not value.keywords and is_owned_name(value.func.value.id, loop))
        def mutation_roots(node: ast.AST) -> set[str]:
            children = list(ast.walk(node))
            roots = {root_name(child.target) for child in children if isinstance(child, ast.AugAssign)}
            roots.update(root_name(child) for child in children if isinstance(child, (ast.Attribute, ast.Subscript)) and isinstance(child.ctx, ast.Store))
            roots.update(root_name(child.func.value) for child in children if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr in collection_methods)
            return roots - {None}
        def is_owned_value(node: ast.AST) -> bool: return isinstance(node, ast.Constant) or (isinstance(node, (ast.List, ast.Set, ast.Tuple)) and all(is_owned_value(value) for value in node.elts)) or (isinstance(node, ast.Dict) and all(key is not None and is_owned_value(key) and is_owned_value(value) for key, value in zip(node.keys, node.values, strict=True))) or (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in pure_functions & {"dict", "list", "set", "str", "tuple"} and not node.args and not node.keywords)
        def precedes(statement: ast.stmt, node: ast.AST) -> bool: return (parent := parents.get(node)) is not None and (any(statement in values[:values.index(node)] for _, values in ast.iter_fields(parent) if isinstance(values, list) and node in values) or precedes(statement, parent))
        def is_owned_name(name: str, loop: ast.For) -> bool: return name not in imported_names and sum((isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store) and child.id == name) or (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child.name == name) or (isinstance(child, ast.ExceptHandler) and child.name == name) for child in ast.walk(tree)) == 1 and any(isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == name and isinstance(statement.value, ast.List) and not statement.value.elts and precedes(statement, loop) for statement in ast.walk(tree))
        def enclosing_function(node: ast.AST) -> ast.AST | None: return parent if isinstance((parent := parents.get(node)), (ast.FunctionDef, ast.AsyncFunctionDef)) else enclosing_function(parent) if parent else None
        def guarded_name(name: str, node: ast.AST) -> tuple[str, str]: return (lambda local: (f"{name!r} in locals()" if local else f"({name!r} in locals() or {name!r} in globals())", f"locals()[{name!r}]" if local else f"(locals() if {name!r} in locals() else globals())[{name!r}]"))(any(scope is not None and table.get_type() == "function" and table.get_name() == scope.name and table.get_lineno() == scope.lineno and any(symbol.get_name() == name and (symbol.is_local() or symbol.is_parameter()) for symbol in table.get_symbols()) for scope in [enclosing_function(node)] for table in tables))
        def is_local_builder(node: ast.AST, roots: set[str]) -> bool:
            return all(not isinstance(child, builder_blocked) and allowed_call(child, roots) and not (isinstance(child, (ast.Attribute, ast.Subscript)) and isinstance(child.ctx, ast.Store) and root_name(child) not in roots) and not (isinstance(child, ast.AugAssign) and not isinstance(child.target, ast.Name)) and not (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr in collection_methods - {"append"}) and not (isinstance(child, (ast.For, ast.AsyncFor)) and loaded_names(child.iter) & with_aliases(one_shot_names)) for child in ast.walk(node))
        def with_aliases(values: set[str], pairs: list[set[str]] = alias_pairs + uncertain_alias_pairs) -> set[str]:
            expanded = set(values)
            while any(pair & expanded and not pair <= expanded for pair in pairs):
                for pair in pairs:
                    if pair & expanded:
                        expanded.update(pair)
            return expanded
        replay_functions = set()
        unsafe_callbacks = with_aliases(({node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))} | {left for targets, value in assignments if isinstance(value, ast.Lambda) for target in targets for left in names(target, ast.Store)}) - replay_functions, [{left, *loaded_names(value)} for targets, value in assignments for target in targets for left in names(target, ast.Store)])
        forbidden_query_parents = (ast.AsyncFor, ast.AsyncFunctionDef, ast.BoolOp, ast.ClassDef, ast.For, ast.FunctionDef, ast.If, ast.IfExp, ast.Lambda, ast.Match, ast.Try, ast.While, ast.With, ast.comprehension)
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
        def indent(node: ast.AST, depth: int = 1) -> str: return "\n".join("    " * depth + line for line in ast.unparse(node).splitlines())
        class QueryBatchTransformer(ast.NodeTransformer):
            def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST: return node
            def visit_For(self, node: ast.For) -> ast.AST | list[ast.stmt]:
                nonlocal rewrite_count
                if (replacement := self._rewrite_for(node)) is None:
                    return self.generic_visit(node)
                rewrite_count += 1
                return replacement
            def _rewrite_for(self, node: ast.For) -> list[ast.stmt] | None:
                generated_start = set(used_names)
                if node.orelse or not is_pure(node.iter) or any(isinstance(child, (ast.Await, ast.AsyncWith, ast.Import, ast.ImportFrom, ast.Return, ast.With, ast.Yield, ast.YieldFrom)) or (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and (child.decorator_list or child.returns or child.args.defaults or any(child.args.kw_defaults) or any(argument.annotation for argument in [*child.args.posonlyargs, *child.args.args, child.args.vararg, *child.args.kwonlyargs, child.args.kwarg] if argument) or getattr(child, "type_params", []))) or (isinstance(child, ast.Lambda) and (child.args.defaults or any(child.args.kw_defaults))) for child in ast.walk(node)):
                    return None
                loop_names, parameter_aliases = bound_names(node.target), with_aliases({arg.arg for function in [enclosing_function(node)] if function is not None for arg in ast.walk(function.args) if isinstance(arg, ast.arg)})
                if not loop_names or len(loop_names) != len(set(loop_names)) or set(loop_names) & set().union(*(stored_names(statement) for statement in node.body)):
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
                    simple_guard = index < first_query and isinstance(statement, ast.If) and not statement.orelse and len(statement.body) == 1 and isinstance(statement.body[0], (ast.Break, ast.Continue)) and is_pure(statement.test)
                    if simple_guard:
                        guards.add(index)
                    elif breaks or index <= last_query:
                        return None
                stores = [stored_names(statement) for statement in node.body]
                candidates = {index: (statement.targets[0].id, statement.value) for index, statement in enumerate(node.body) if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name) and index not in query_statements and is_pure(statement.value)}
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
                prompt_inputs, guard_inputs = [(index, query.args[0]) for index, queries in query_statements.items() for query in queries], [(index, node.body[index].test) for index in guards]
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
                    bad_initializer = None in initializers or any(initializer >= index or not is_owned_value(candidates[initializer][1]) for initializer in initializers) or (any(isinstance(child, ast.AugAssign) for child in ast.walk(node.body[index])) and any(not isinstance(candidates[initializer][1], ast.Constant) for initializer in initializers))
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
                if any(position > index and name in loaded_names(statement) for index, statement in enumerate(node.body) for name, position in selected_positions.items()):
                    return None
                dependencies.update(selected_names, *(loaded_names(candidates[index][1]) for index in selected), *(loaded_names(node.body[index]) for index in duplicated))
                for index in range(last_query + 1):
                    if index in selected | duplicated | guards | set(query_statements):
                        continue
                    statement = node.body[index]
                    is_print = isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name) and statement.value.func.id == "print"
                    print_values = [*statement.value.args, *(keyword.value for keyword in statement.value.keywords)] if is_print else []
                    safe_print_names = set(loop_names) | loaded_names(node.iter) | set(selected_names) | set().union(*(mutation_roots(node.body[position]) for position in duplicated)) | pure_functions | {"print"}
                    if stored_names(statement) or not is_print or (index < first_query and (not all(is_pure(value) for value in print_values) or loaded_names(statement) - safe_print_names or any(isinstance(value, ast.Starred) or any(isinstance(child, ast.GeneratorExp) or (isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id in {"enumerate", "zip"}) for child in ast.walk(value)) for value in statement.value.args) or statement.value.keywords)):
                        return None
                    prequery_prints.update({index: ", ".join(ast.unparse(value) for value in print_values)} if index < first_query and print_values else {})
                mutations, runtime_mutations, callback_roots, receiver_roots = set(), set(), set(), set()
                for index, statement in enumerate(node.body):
                    if index in selected | duplicated:
                        continue
                    (mutations.update(stored_names(statement), mutation_roots(statement)), runtime_mutations.update(mutation_roots(statement)), receiver_roots.update((loaded_names(statement) | (names(statement, ast.Load) & names(statement, ast.Store)) | mutation_roots(statement)) - set(loop_names) - set().union(*(names(target, ast.Store) for position in range(index) for target in (node.body[position].targets if isinstance(node.body[position], ast.Assign) else [node.body[position].target] if isinstance(node.body[position], (ast.AnnAssign, ast.AugAssign)) else []))) if index > last_query else set()))
                    for call in (child for child in ast.walk(statement) if isinstance(child, ast.Call) and not is_query_call(child)):
                        known = allowed_call(call, mutation_roots(statement)) or (isinstance(call.func, ast.Name) and call.func.id == "print")
                        if not known and index <= last_query:
                            return None
                        if known:
                            continue
                        if has_callback_options(call) or (isinstance(call.func, ast.Name) and call.func.id not in replay_functions) or any(with_aliases(loaded_names(value)) & (unsafe_callbacks | parameter_aliases) for value in [*call.args, *(keyword.value for keyword in call.keywords)]):
                            return None
                        (mutations.update(*(loaded_names(value) for value in [*call.args, *(keyword.value for keyword in call.keywords)]), *([loaded_names(call.func.value)] if isinstance(call.func, ast.Attribute) else [])), runtime_mutations.update(*(loaded_names(value) for value in [*call.args, *(keyword.value for keyword in call.keywords)]), *([loaded_names(call.func.value)] if isinstance(call.func, ast.Attribute) else [])), callback_roots.update(*(loaded_names(value) for value in [*call.args, *(keyword.value for keyword in call.keywords), *([call.func] if not isinstance(call.func, (ast.Name, ast.Attribute)) else [])])), receiver_roots.update(*([loaded_names(call.func.value)] if isinstance(call.func, ast.Attribute) else [])))
                dependency_aliases, mutation_aliases = with_aliases(dependencies), with_aliases(mutations)
                if set(loop_names) & mutations or dependency_aliases & mutation_aliases or any(pair & dependency_aliases and pair & mutation_aliases for pair in uncertain_alias_pairs):
                    return None
                runtime_aliases, runtime_callbacks, runtime_receivers = [(left, right) for left in dependency_aliases - pure_functions - {"llm_query", "print"} for right in with_aliases(runtime_mutations) - pure_functions - {"llm_query", "print"} if left != right], with_aliases(callback_roots) - pure_functions - replay_functions - {"llm_query", "print"}, with_aliases(receiver_roots | (dependencies & pure_modules)) - pure_functions - replay_functions - {"llm_query", "print"}
                selected_order, target_source, replay_target = sorted(selected), ast.unparse(node.target), ", ".join(loop_names)
                temp_names = ("frames", "prompts", "responses", "frame", "frame_index", "final_target", "final_values", "gather_error", "gather_position", "gather_failed")
                frames, prompts, responses, frame, frame_index, final_target, final_values, gather_error, gather_position, gather_failed = (fresh_name(f"__dspy_{name}") for name in temp_names)
                frame_error = fresh_name("__dspy_frame_error")
                gather = ["try:", f"    for {target_source} in {ast.unparse(node.iter)}:", f"        {final_target} = [{', '.join(loop_names)}]", f"        {frame} = [{', '.join([*loop_names, *(['None'] * len(selected_order))])}]", "        try:"]
                for index, statement in enumerate(node.body):
                    if index in selected | duplicated | guards | set(query_statements) | set(prequery_prints):
                        gather.append(f"            {gather_position} = {index}" + (f"\n            ({prequery_prints[index]},)" if index in prequery_prints else ""))
                    if index in selected | duplicated:
                        gather.append(indent(statement, 3))
                    if index in selected:
                        offset = len(loop_names) + selected_order.index(index)
                        gather.extend([f"            {frame}[{offset}] = {candidates[index][0]}", f"            {final_values}[{index}] = {candidates[index][0]}"])
                    elif index in guards:
                        gather.append(indent(statement, 3))
                    for query in query_statements.get(index, []):
                        gather.append(f"            {prompts}.append({ast.unparse(query.args[0])})")
                gather.extend(["        except Exception:", f"            {gather_failed} = True", f"            {frames}.append({frame})", "            raise", f"        {frames}.append({frame})", f"except Exception as {frame_error}:", f"    {gather_error} = {frame_error}", f"{responses} = __dspy_llm_query_batched({prompts})", f"for {frame_index}, {frame} in enumerate({frames}):"])
                gather.append(f"    {replay_target} = {', '.join(f'{frame}[{index}]' for index in range(len(loop_names)))}")
                original_loop = ast.unparse(node)
                query_index = 0
                class ResponseReplacer(ast.NodeTransformer):
                    def visit_Call(self, call: ast.Call) -> ast.AST:
                        nonlocal query_index
                        if not is_query_call(call):
                            return self.generic_visit(call)
                        expression = ast.parse(f"__dspy_replay_llm_query({responses}[{frame_index} * {len(query_statements)} + {query_index}])", mode="eval").body
                        query_index += 1
                        return ast.copy_location(expression, call)
                replayed_body = [ResponseReplacer().visit(statement) for statement in node.body]
                if query_index != len(query_statements):
                    return None
                gather.append(f"    if {gather_failed} and {frame_index} == len({frames}) - 1:")
                for index, statement in enumerate(replayed_body):
                    comparison = ">" if index in query_statements else ">="
                    if index in query_statements:
                        gather.extend([f"        if {gather_position} == {index}:", f"            raise {gather_error}"])
                    gather.append(f"        if {gather_position} {comparison} {index}:")
                    gather.append(indent(statement, 3))
                gather.extend([f"        raise {gather_error}", "    else:"])
                for index, statement in enumerate(replayed_body):
                    gather.append(f"        {candidates[index][0]} = {frame}[{len(loop_names) + selected_order.index(index)}]" if index in selected else indent(statement, 2))
                gather.extend([f"if {final_target} is not None:", f"    {replay_target} = {', '.join(f'{final_target}[{index}]' for index in range(len(loop_names)))}"])
                for index in selected_order:
                    gather.extend([f"if {index} in {final_values}:", f"    {candidates[index][0]} = {final_values}[{index}]"])
                gather.extend([f"if {gather_error} is not None:", f"    raise {gather_error}"])
                initializers = [f"{frames} = []", f"{prompts} = []", f"{responses} = None", f"{final_target} = None", f"{final_values} = {{}}", f"{gather_error} = {gather_position} = None", f"{gather_failed} = False", f"{frame} = {frame_index} = None"]
                staged = "\n".join("    " + line for line in "\n".join(gather).splitlines())
                cleanup = f"    del {frames}, {prompts}, {responses}, {frame}, {frame_index}, {final_target}, {final_values}, {gather_error}, {gather_position}, {gather_failed}"
                compiled = "\n".join([*initializers, "try:", staged, "finally:", cleanup])
                if runtime_aliases or runtime_callbacks or runtime_receivers:
                    alias_ids = fresh_name("__dspy_alias_ids")
                    checks = " or ".join([f"({guarded_name(left, node)[0]} and {guarded_name(right, node)[0]} and {alias_ids}({guarded_name(left, node)[1]})[0] & {alias_ids}({guarded_name(right, node)[1]})[0])" for left, right in runtime_aliases] + [f"({guarded_name(name, node)[0]} and ({alias_ids}({guarded_name(name, node)[1]})[1] or not {alias_ids}({guarded_name(name, node)[1]})[2]))" for name in runtime_callbacks] + [f"({guarded_name(name, node)[0]} and not {alias_ids}({guarded_name(name, node)[1]})[2])" for name in runtime_receivers])
                    compiled = f"def {alias_ids}(value):\n    stack, seen, mutable, callback, native = [value], set(), set(), False, True\n    while stack:\n        value = stack.pop()\n        if id(value) in seen:\n            continue\n        seen.add(id(value))\n        callback = callback or callable(value)\n        if type(value) in (dict, list, tuple, set):\n            if type(value) is not tuple:\n                mutable.add(id(value))\n            stack.extend([*value.keys(), *value.values()] if type(value) is dict else value)\n        elif type(value) not in (str, int, float, bool, type(None)):\n            native = False\n    return mutable, callback, native\ntry:\n" + "\n".join(f"    {line}" for line in (f"if {checks}:\n" + "\n".join(f"    {line}" for line in original_loop.splitlines()) + "\nelse:\n" + "\n".join(f"    {line}" for line in compiled.splitlines())).splitlines()) + f"\nfinally:\n    del {alias_ids}"
                system, gettrace, getprofile = (fresh_name(f"__dspy_{name}") for name in ("sys", "gettrace", "getprofile"))
                trusted_instrumentation = f"type({gettrace}) is type(len) and {gettrace}.__self__ is {system} and {gettrace}.__module__ == 'sys' and {gettrace}.__name__ == 'gettrace' and type({getprofile}) is type(len) and {getprofile}.__self__ is {system} and {getprofile}.__module__ == 'sys' and {getprofile}.__name__ == 'getprofile'"
                compiled = f"{system} = __import__('sys')\n{gettrace}, {getprofile} = {system}.gettrace, {system}.getprofile\ntry:\n" + "\n".join(f"    {line}" for line in (f"if not ({trusted_instrumentation}) or {gettrace}() is not None or {getprofile}() is not None:\n" + "\n".join(f"    {line}" for line in original_loop.splitlines()) + "\nelse:\n" + "\n".join(f"    {line}" for line in compiled.splitlines())).splitlines()) + f"\nfinally:\n    del {system}, {gettrace}, {getprofile}"
                compiled = f"if {' or '.join(f'{name!r} in globals()' for name in sorted(used_names - generated_start))}:\n" + "\n".join(f"    {line}" for line in original_loop.splitlines()) + "\nelse:\n" + "\n".join(f"    {line}" for line in compiled.splitlines())
                return ast.parse(compiled).body
        transformed = QueryBatchTransformer().visit(tree)
        if not rewrite_count:
            return code, 0
        return ast.unparse(ast.fix_missing_locations(transformed)), rewrite_count

    def _normalize_tools(self, tools: list[Callable] | None) -> dict[str, Tool]:
        """Normalize tools list to a dict of Tool objects keyed by name."""
        if not tools:
            return {}

        if isinstance(tools, dict):
            raise TypeError(
                "tools must be a list, not a dict. "
                "Change tools={'name': func} to tools=[func] "
                "(tool names are inferred from function names, or use dspy.Tool(func, name='custom_name'))"
            )

        def to_tool(func: Callable | Tool) -> Tool:
            if isinstance(func, Tool):
                return func
            if not callable(func):
                raise TypeError(f"Tool {func!r} must be callable, got {type(func).__name__}")
            return Tool(func)

        normalized = {}
        for value in tools:
            tool = to_tool(value)
            if tool.name in normalized:
                raise ValueError(f"Duplicate tool name '{tool.name}'")
            normalized[tool.name] = tool
        return normalized

    def _validate_namespace(self, tools: dict[str, Tool]) -> None:
        """Validate names owned by the RLM result and sandbox APIs."""
        for name in tools:
            if not name.isidentifier() or keyword.iskeyword(name):
                raise ValueError(f"Invalid tool name '{name}': must be a valid Python identifier and not a keyword")
            if name in self._RESERVED_SANDBOX_NAMES or name.startswith("__dspy_"):
                raise ValueError(f"Tool name '{name}' conflicts with built-in sandbox function")

        input_names = set(self.signature.input_fields)
        reserved_inputs = sorted(name for name in input_names if name in self._RESERVED_SANDBOX_NAMES or name.startswith("__dspy_"))
        if reserved_inputs:
            raise ValueError(f"Input fields conflict with built-in sandbox functions: {reserved_inputs}")

        tool_inputs = sorted(input_names & tools.keys())
        if tool_inputs:
            raise ValueError(f"Input fields conflict with user tools: {tool_inputs}")

        reserved_outputs = sorted(set(self.signature.output_fields) & self._RESERVED_RESULT_NAMES)
        if reserved_outputs:
            raise ValueError(f"Output fields conflict with RLM result metadata: {reserved_outputs}")

    def _format_tool_docs(self, tools: dict[str, Tool]) -> str:
        """Format user-provided tools for inclusion in instructions."""
        if not tools:
            return ""

        lines = ["\nAdditional tools available (use these instead of standard library equivalents):"]
        for tool in tools.values():
            # Build signature string from Tool's args
            params = []
            for arg_name, arg_schema in (tool.args or {}).items():
                arg_type = arg_schema.get("type", "Any")
                params.append(f"{arg_name}: {arg_type}")
            params_str = ", ".join(params)
            sig_str = f"{tool.name}({params_str})"

            # Get description with newlines escaped
            desc = (tool.desc or "No description").replace("\n", "  ")
            lines.append(f"- `{sig_str}` - {desc}")

        return "\n".join(lines)

    def _make_llm_tools(self, max_workers: int = 8) -> dict[str, Callable]:
        """Create llm_query and llm_query_batched tools with a fresh call counter."""
        state = {"call_count": 0}
        errors = []
        lock = threading.Lock()
        lm = self.sub_lm

        def _check_and_increment(n: int = 1) -> None:
            with lock:
                if state["call_count"] + n > self.max_llm_calls:
                    raise RuntimeError(
                        f"LLM call limit exceeded: {state['call_count']} + {n} > {self.max_llm_calls}. "
                        f"Use Python code for aggregation instead of making more LLM calls."
                    )
                state["call_count"] += n

        def _error_outcome(error: Exception) -> dict[str, int]:
            with lock:
                errors.append(error)
                return {"error": len(errors) - 1}

        def _query_lm(prompt: str) -> str:
            target_lm = lm if lm is not None else dspy.settings.lm
            if target_lm is None:
                raise dspy.LMNotConfiguredError(
                    "No LM configured. Use dspy.configure(lm=...) or pass sub_lm to RLM."
                )
            response = target_lm(prompt)
            if isinstance(response, dspy.LMResponse):
                text = response.text
            elif isinstance(response, list) and response:
                first_output = response[0]
                text = first_output.get("text") if isinstance(first_output, dict) else first_output
            else:
                raise TypeError(
                    "Sub-LM must return dspy.LMResponse or a non-empty list of text outputs, "
                    f"got {type(response).__name__}."
                )

            if not isinstance(text, str):
                raise TypeError(f"Sub-LM response must contain text, got {type(text).__name__}.")
            return text

        def llm_query(prompt: str) -> str:
            """Query the LLM with a prompt string."""
            if not prompt:
                raise ValueError("prompt cannot be empty")
            _check_and_increment(1)
            return _query_lm(prompt)

        def __dspy_llm_query_batched(prompts: list[str]) -> list[dict[str, str | int]]:
            """Run the scalar-call prefix allowed by validation and budget, retaining ordered failures for replay."""
            outcomes = []
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for prompt in prompts:
                    if not prompt:
                        outcomes.append(_error_outcome(ValueError("prompt cannot be empty")))
                        break
                    try:
                        _check_and_increment(1)
                    except Exception as error:
                        outcomes.append(_error_outcome(error))
                        break
                    outcomes.append({})
                    futures.append((len(outcomes) - 1, executor.submit(contextvars.copy_context().run, _query_lm, prompt)))
                for index, future in futures:
                    try:
                        outcomes[index] = {"value": future.result()}
                    except Exception as error:
                        outcomes[index] = _error_outcome(error)
            return outcomes

        def __dspy_replay_llm_query(outcome: dict) -> str:
            if "error" in outcome:
                raise errors[outcome["error"]]
            return outcome["value"]

        def llm_query_batched(prompts: list[str]) -> list[str]:
            """Query prompts concurrently with the same validation and errors as llm_query."""
            return [__dspy_replay_llm_query(outcome) for outcome in __dspy_llm_query_batched(prompts)]

        return {"llm_query": llm_query, "llm_query_batched": llm_query_batched, "__dspy_llm_query_batched": __dspy_llm_query_batched, "__dspy_replay_llm_query": __dspy_replay_llm_query}

    @property
    def tools(self) -> dict[str, Tool]:
        """User-provided tools (excludes internal llm_query/llm_query_batched)."""
        return dict(self._user_tools)

    # =========================================================================
    # Signature Building
    # =========================================================================

    def _build_signatures(self) -> tuple[Signature, Signature]:
        """Build the action and extract signatures from templates."""
        inputs_str = ", ".join(f"`{n}`" for n in self.signature.input_fields)

        # Simple names for SUBMIT() examples
        final_output_names = ", ".join(self.signature.output_fields.keys())

        output_fields = "\n".join(
            f"- {translate_field_type(n, f)}"
            for n, f in self.signature.output_fields.items()
        )

        # Include original signature instructions (docstring) if present
        task_instructions = f"{self.signature.instructions}\n\n" if self.signature.instructions else ""

        # Format tool documentation for user-provided tools
        tool_docs = self._format_tool_docs(self._user_tools)

        execution_instructions = getattr(self._interpreter_factory, "execution_instructions", "")
        if not isinstance(execution_instructions, str):
            raise TypeError("interpreter_factory.execution_instructions must be a string")
        interpreter_rules = f"\nExecution environment:\n{execution_instructions}\n" if execution_instructions else ""

        action_sig = (
            dspy.Signature({}, task_instructions + ACTION_INSTRUCTIONS_TEMPLATE.format(
                inputs=inputs_str, final_output_names=final_output_names, output_fields=output_fields,
                max_llm_calls=self.max_llm_calls, interpreter_rules=interpreter_rules,
            ) + tool_docs)
            .append("variables_info", dspy.InputField(desc="Metadata about the variables available in the REPL"), type_=str)
            .append("repl_history", dspy.InputField(desc="Previous REPL code executions and their outputs"), type_=REPLHistory)
            .append("iteration", dspy.InputField(desc="Current iteration number (1-indexed) out of max_iters"), type_=str)
            .append("reasoning", dspy.OutputField(desc="Think step-by-step: what do you know? What remains? Plan your next action."), type_=str)
            .append("code", dspy.OutputField(desc="Python code to execute. Use markdown code block format: ```python\\n<code>\\n```"), type_=str)
        )

        # Extract signature: includes the original signature's output fields and task instructions.
        extract_instructions = """Based on the REPL trajectory, extract the final outputs now.

            Review your trajectory to see what information you gathered and what values you computed, then provide the final outputs."""

        # Prepend original task instructions to extract instructions so the LLM knows what task to extract for
        extended_task_instructions = ""
        if task_instructions:
            extended_task_instructions = "The trajectory was generated with the following objective: \n" + task_instructions + "\n"
        full_extract_instructions = extended_task_instructions + extract_instructions

        extract_sig = dspy.Signature(
            {**self.signature.output_fields},
            full_extract_instructions,
        )
        extract_sig = extract_sig.prepend("repl_history", dspy.InputField(desc="Your REPL interactions so far"), type_=REPLHistory)
        extract_sig = extract_sig.prepend("variables_info", dspy.InputField(desc="Metadata about the variables available in the REPL"), type_=str)

        return action_sig, extract_sig

    # =========================================================================
    # Input/Output Processing
    # =========================================================================

    def _get_output_fields_info(self) -> list[dict]:
        """Get output field info for sandbox registration."""
        fields = []
        for name, field in self.signature.output_fields.items():
            annotation = getattr(field, "annotation", str)
            field_info = {"name": name}
            # Only include type for simple types that work in function signatures
            # Complex types like Literal, Union, etc. are not included
            if annotation in SIMPLE_TYPES:
                field_info["type"] = annotation.__name__
            fields.append(field_info)
        return fields

    def _build_variables(self, **input_args: Any) -> list[REPLVariable]:
        """Build REPLVariable list from input arguments with field metadata."""
        variables = []
        for name, value in input_args.items():
            field_info = self.signature.input_fields.get(name)
            if isinstance(value, SandboxSerializable):
                var = build_repl_variable(value, name, field_info=field_info)
            else:
                var = REPLVariable.from_value(name, value, field_info=field_info)
            variables.append(var)
        return variables

    def _format_output(self, output: str) -> str:
        if not output:
            return "(no output - did you forget to print?)"
        return output

    def _validate_inputs(self, input_args: dict[str, Any]) -> None:
        """Validate call-time arguments against the signature's input namespace."""
        if "interpreter" in input_args and "interpreter" not in self.signature.input_fields:
            raise TypeError(
                "To use a caller-owned interpreter, pass it as the first positional argument when calling the module."
            )
        input_names = set(self.signature.input_fields)
        unexpected = set(input_args) - input_names
        if unexpected:
            raise ValueError(f"Unexpected inputs not declared in the signature: {sorted(unexpected)}")

        missing = input_names - set(input_args)
        if missing:
            raise ValueError(f"Missing required inputs: {sorted(missing)}")

    def _prepare_serializable_vars(
        self, input_args: dict[str, Any], repl: CodeInterpreter,
    ) -> dict[str, Any]:
        """Inject SandboxSerializable values into the interpreter.

        For each SandboxSerializable value in input_args, serializes it and
        executes setup + assignment code in the interpreter. Returns the
        remaining non-serializable args (for per-iteration use).
        """
        repl.start()
        regular_args = {}
        for name, value in input_args.items():
            if not isinstance(value, SandboxSerializable):
                regular_args[name] = value
                continue

            payload = value.to_sandbox()
            setup = value.sandbox_setup()
            raw_var_name = f"_raw_{name}"
            assignment = value.sandbox_assignment(name, raw_var_name)
            code_lines = []
            payload_vars: dict[str, str] = {}
            if isinstance(payload, bytes):
                try:
                    payload_vars[raw_var_name] = payload.decode("utf-8")
                except UnicodeDecodeError:
                    encoded_var_name = f"{raw_var_name}_base64"
                    payload_vars[encoded_var_name] = base64.b64encode(payload).decode("ascii")
                    code_lines.extend([
                        "import base64",
                        f"{raw_var_name} = base64.b64decode({encoded_var_name})",
                    ])
            else:
                payload_vars[raw_var_name] = str(payload)

            if setup:
                code_lines.append(setup)
            code_lines.append(assignment)
            repl.execute("\n".join(code_lines), variables=payload_vars)

        return regular_args

    # =========================================================================
    # CodeInterpreter Lifecycle
    # =========================================================================

    def _make_interpreter_tool(self, tool: Tool) -> Callable:
        """Preserve function metadata while routing execution through Tool."""
        if inspect.iscoroutinefunction(tool.func) or inspect.iscoroutinefunction(getattr(tool.func, "__call__", None)):
            async def invoke(**kwargs):
                return await tool.acall(**kwargs)
        else:
            def invoke(**kwargs):
                return tool(**kwargs)

        functools.update_wrapper(invoke, tool.func)
        invoke.__signature__ = inspect.signature(tool.func)
        return invoke

    def _prepare_execution_tools(self) -> dict[str, Callable]:
        """Create fresh LLM tools and merge with user-provided tools."""
        execution_tools = self._make_llm_tools()
        execution_tools.update({name: self._make_interpreter_tool(tool) for name, tool in self._user_tools.items()})
        return execution_tools

    def _inject_execution_context(self, interpreter: CodeInterpreter, execution_tools: dict[str, Callable]) -> None:
        """Inject execution tools and output fields into an interpreter.

        This ensures llm_query, llm_query_batched, and typed FINAL signatures are available,
        even for user-provided interpreters. Each forward() call gets fresh tools with a
        fresh call counter, so we must inject on every execution.
        """
        interpreter.tools.update(execution_tools)
        if hasattr(interpreter, "output_fields"):
            interpreter.output_fields = self._get_output_fields_info()
        # Reset registration flag to force re-registration with fresh tools
        if hasattr(interpreter, "_tools_registered"):
            interpreter._tools_registered = False

    @contextmanager
    def _interpreter_context(
        self,
        execution_tools: dict[str, Callable],
        interpreter: CodeInterpreter | None,
    ) -> Iterator[CodeInterpreter]:
        """Yield a caller-owned interpreter or manage a factory-created one."""
        if interpreter is not None:
            _validate_interpreter(interpreter)
            self._inject_execution_context(interpreter, execution_tools)
            yield interpreter
            return

        interpreter = _create_interpreter(self._interpreter_factory)
        try:
            self._inject_execution_context(interpreter, execution_tools)
            yield interpreter
        finally:
            interpreter.shutdown()

    # =========================================================================
    # Execution Core
    # =========================================================================

    def _extract_fallback(
        self,
        variables: list[REPLVariable],
        history: REPLHistory,
        output_field_names: list[str],
    ) -> Prediction:
        """Use extract module to get final output when max iterations reached."""
        logger.warning("RLM reached max iterations, using extract to get final output")

        variables_info = [variable.format() for variable in variables]
        extract_pred = self.extract(
            variables_info=variables_info,
            repl_history=history,
        )

        return Prediction(
            trajectory=[e.model_dump() for e in history],
            final_reasoning="Extract forced final output",
            **{name: getattr(extract_pred, name) for name in output_field_names},
        )

    def _process_final_output(
        self,
        result: FinalOutput,
        output_field_names: list[str],
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Validate and parse FinalOutput. Returns (parsed_outputs, None) or (None, error)."""
        raw_output = result.output

        # Validate raw_output is a dict
        if not isinstance(raw_output, dict):
            return None, f"[Error] FINAL returned {type(raw_output).__name__}, expected dict with fields: {output_field_names}"

        # Validate all required output fields are present
        missing = set(output_field_names) - set(raw_output.keys())
        if missing:
            return None, f"[Error] Missing output fields: {sorted(missing)}. Use SUBMIT({', '.join(output_field_names)})"

        # Parse and validate each output field
        parsed_outputs = {}
        type_errors = []
        for name in output_field_names:
            field = self.signature.output_fields[name]
            annotation = getattr(field, "annotation", str)
            try:
                parsed_outputs[name] = parse_value(raw_output[name], annotation)
            except (ValueError, pydantic.ValidationError) as e:
                type_errors.append(
                    f"{name}: expected {annotation.__name__ if hasattr(annotation, '__name__') else annotation}, "
                    f"got {type(raw_output[name]).__name__}: {e}"
                )

        if type_errors:
            return None, "[Type Error] " + "; ".join(type_errors)

        return parsed_outputs, None

    def _process_execution_result(
        self,
        pred: Prediction,
        code: str,
        result: Any,
        history: REPLHistory,
        output_field_names: list[str],
    ) -> Prediction | REPLHistory:
        """Process interpreter result, returning Prediction if final, else updated history.

        This shared helper reduces duplication between sync and async execution paths.

        Args:
            pred: The prediction containing reasoning and code attributes
            code: Code to record in history (already stripped when possible)
            result: Result from interpreter.execute() - FinalOutput, list, str, or error string
            history: Current REPL history
            output_field_names: List of expected output field names

        Returns:
            Prediction if FINAL was called successfully, else updated REPLHistory
        """
        # Handle error strings from caught exceptions
        if isinstance(result, str) and result.startswith("[Error]"):
            output = self._format_output(result)
            return history.append(reasoning=pred.reasoning, code=code, output=output)

        # Handle FINAL output
        if isinstance(result, FinalOutput):
            parsed_outputs, error = self._process_final_output(result, output_field_names)

            if error:
                return history.append(reasoning=pred.reasoning, code=code, output=error)

            final_history = history.append(
                reasoning=pred.reasoning, code=code, output=f"FINAL: {parsed_outputs}"
            )
            return Prediction(
                **parsed_outputs,
                trajectory=[e.model_dump() for e in final_history],
                final_reasoning=pred.reasoning,
            )

        # Format non-final result as output
        if isinstance(result, list):
            output = "\n".join(map(str, result))
        else:
            output = str(result) if result else ""

        output = self._format_output(output)
        if self.verbose:
            logger.info(REPLEntry.format_output(output, self.max_output_chars))
        return history.append(reasoning=pred.reasoning, code=code, output=output)

    def _execute_code(
        self,
        repl: CodeInterpreter,
        code: str,
        input_args: dict[str, Any],
    ) -> Any:
        """Execute code in the interpreter, returning the result or an error string."""
        execution_code, _ = self._compile_llm_query_loops(code)
        try:
            return repl.execute(execution_code, variables=dict(input_args))
        except (CodeExecutionError, SyntaxError) as e:
            return f"[Error] {format_error_for_lm(e)}"

    def _execute_iteration(
        self,
        repl: CodeInterpreter,
        variables: list[REPLVariable],
        history: REPLHistory,
        iteration: int,
        input_args: dict[str, Any],
        output_field_names: list[str],
    ) -> Prediction | REPLHistory:
        """Execute one iteration. Returns Prediction if done, else updated REPLHistory."""
        variables_info = [variable.format() for variable in variables]
        action = self.generate_action(
            variables_info=variables_info,
            repl_history=history,
            iteration=f"{iteration + 1}/{self.max_iters}",
        )
        if self.verbose:
            logger.info(
                f"RLM iteration {iteration + 1}/{self.max_iters}\n"
                f"Reasoning: {action.reasoning}\nCode:\n{action.code}"
            )

        try:
            code = _strip_code_fences(action.code)
        except SyntaxError as e:
            code = action.code
            result = f"[Error] {format_error_for_lm(e)}"
            return self._process_execution_result(action, code, result, history, output_field_names)
        result = self._execute_code(repl, code, input_args)
        return self._process_execution_result(action, code, result, history, output_field_names)

    # =========================================================================
    # Public Interface
    # =========================================================================

    def forward(self, interpreter: CodeInterpreter | None = None, /, **input_args) -> Prediction:
        """Execute RLM to produce outputs from the given inputs.

        Args:
            interpreter: Optional caller-owned interpreter, passed positionally. RLM injects invocation tools and
                output metadata into it but does not shut it down. Reuse is supported only for sequential calls to
                this RLM instance.
            **input_args: Input values matching the signature's input fields.

        Returns:
            Prediction with output field(s) from the signature and 'trajectory' for debugging

        Raises:
            ValueError: If required input fields are missing
            CodeInterpreterError: If interpreter setup, process, or protocol fails
        """
        self._validate_inputs(input_args)

        output_field_names = list(self.signature.output_fields.keys())
        execution_tools = self._prepare_execution_tools()
        variables = self._build_variables(**input_args)

        with self._interpreter_context(execution_tools, interpreter) as repl:
            regular_args = self._prepare_serializable_vars(input_args, repl)
            history: REPLHistory = REPLHistory(max_output_chars=self.max_output_chars)

            for iteration in range(self.max_iters):
                result: Prediction | REPLHistory = self._execute_iteration(
                    repl, variables, history, iteration, regular_args, output_field_names
                )
                if isinstance(result, Prediction):
                    return result
                history = result

            # Max iterations reached - use extract fallback
            return self._extract_fallback(variables, history, output_field_names)

    async def _aextract_fallback(
        self,
        variables: list[REPLVariable],
        history: REPLHistory,
        output_field_names: list[str],
    ) -> Prediction:
        """Async version: Use extract module when max iterations reached."""
        logger.warning("RLM reached max iterations, using extract to get final output")

        variables_info = [variable.format() for variable in variables]
        extract_pred = await self.extract.acall(
            variables_info=variables_info,
            repl_history=history,
        )

        return Prediction(
            trajectory=[e.model_dump() for e in history],
            final_reasoning="Extract forced final output",
            **{name: getattr(extract_pred, name) for name in output_field_names},
        )

    async def _aexecute_iteration(
        self,
        repl: CodeInterpreter,
        variables: list[REPLVariable],
        history: REPLHistory,
        iteration: int,
        input_args: dict[str, Any],
        output_field_names: list[str],
    ) -> Prediction | REPLHistory:
        """Async version: Execute one iteration."""
        variables_info = [variable.format() for variable in variables]
        pred = await self.generate_action.acall(
            variables_info=variables_info,
            repl_history=history,
            iteration=f"{iteration + 1}/{self.max_iters}",
        )
        if self.verbose:
            logger.info(
                f"RLM iteration {iteration + 1}/{self.max_iters}\n"
                f"Reasoning: {pred.reasoning}\nCode:\n{pred.code}"
            )

        try:
            code = _strip_code_fences(pred.code)
        except SyntaxError as e:
            code = pred.code
            result = f"[Error] {format_error_for_lm(e)}"
            return self._process_execution_result(pred, code, result, history, output_field_names)
        result = self._execute_code(repl, code, input_args)
        return self._process_execution_result(pred, code, result, history, output_field_names)

    async def aforward(self, interpreter: CodeInterpreter | None = None, /, **input_args) -> Prediction:
        """Async version of forward(). Execute RLM to produce outputs.

        Args:
            interpreter: Optional caller-owned interpreter, passed positionally. RLM injects invocation tools and
                output metadata into it but does not shut it down. Reuse is supported only for sequential calls to
                this RLM instance.
            **input_args: Input values matching the signature's input fields.

        Returns:
            Prediction with output field(s) from the signature and 'trajectory' for debugging

        Raises:
            ValueError: If required input fields are missing
            CodeInterpreterError: If interpreter setup, process, or protocol fails
        """
        self._validate_inputs(input_args)

        output_field_names = list(self.signature.output_fields.keys())
        execution_tools = self._prepare_execution_tools()
        variables = self._build_variables(**input_args)

        with self._interpreter_context(execution_tools, interpreter) as repl:
            regular_args = self._prepare_serializable_vars(input_args, repl)
            history = REPLHistory(max_output_chars=self.max_output_chars)

            for iteration in range(self.max_iters):
                result = await self._aexecute_iteration(
                    repl, variables, history, iteration, regular_args, output_field_names
                )
                if isinstance(result, Prediction):
                    return result
                history = result

            # Max iterations reached - use extract fallback
            return await self._aextract_fallback(variables, history, output_field_names)
