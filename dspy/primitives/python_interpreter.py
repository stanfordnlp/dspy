# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import ast
import builtins
import difflib
import importlib
import re
import typing
from collections.abc import Mapping
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)


class InterpreterError(ValueError):
    r"""An error raised when the interpreter cannot evaluate a Python
    expression, due to syntax error or unsupported operations.
    """

    pass


class PythonInterpreter:
    r"""A customized python interpreter to control the execution of
    LLM-generated codes. The interpreter makes sure the code can only execute
    functions given in action space and import white list. It also supports
    fuzzy variable matching to receive uncertain input variable name.

    .. highlight:: none

    This class is adapted from the Camel adaptation https://github.com/camel-ai/
    camel/blob/9a9d71874944e9736c55cdaed3df469a8becec05/camel/utils/python_interpreter.py 
    which adapts from the hugging face implementation `python_interpreter.py 
    <https://github.com/huggingface/transformers/blob/8f093fb799246f7dd9104ff44728da0c53a9f67a
    /src/transformers/tools/python_interpreter.py>`_. The original license applies::

        Copyright 2023 The HuggingFace Inc. team. All rights reserved.

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
        implied. See the License for the specific language governing
        permissions and limitations under the License.

    Camel's modifications:
    "We have modified the original code to suit our requirements. We have
    encapsulated the original functions within a class and saved the
    interpreter state after execution. We have added support for "import"
    statements, "for" statements, and several binary and unary operators. We
    have added import white list to keep `import` statement safe. Additionally,
    we have modified the variable matching logic and introduced the
    :obj:`fuzz_state` for fuzzy matching."

    DSPy's modifications:
    "We expanded upon the Camel libraries modifications by adding additional 
    support for "Mapping" statements, "conditional" operators, and including
    the "CodePrompt" and "TextPrompt" classes for code execution.
    

    Modifications copyright (C) 2023 CAMEL-AI.org

    Args:
        action_space (Dict[str, Any]): A dictionary that maps action names to
            their corresponding functions or objects. The interpreter can only
            execute functions that are either directly listed in this
            dictionary or are member functions of objects listed in this
            dictionary. The concept of :obj:`action_space` is derived from
            EmbodiedAgent, representing the actions that an agent is capable of
            performing.
        import_white_list (Optional[List[str]], optional): A list that stores
            the Python modules or functions that can be imported in the code.
            All submodules and functions of the modules listed in this list are
            importable. Any other import statements will be rejected. The
            module and its submodule or function name are separated by a period
            (:obj:`.`). (default: :obj:`None`)
    """

    def __init__(self, action_space: Dict[str, Any],
                 import_white_list: Optional[List[str]] = None) -> None:
        self.action_space = action_space
        self.state = self.action_space.copy()
        self.fuzz_state: Dict[str, Any] = {}
        self.import_white_list = import_white_list or []

    def execute(self, code: str, state: Optional[Dict[str, Any]] = None,
                fuzz_state: Optional[Dict[str, Any]] = None,
                keep_state: bool = True) -> Any:
        r""" Execute the input python codes in a security environment.

        Args:
            code (str): Generated python code to be executed.
            state (Optional[Dict[str, Any]], optional): External variables that
                may be used in the generated code. (default: :obj:`None`)
            fuzz_state (Optional[Dict[str, Any]], optional): External variables
                that do not have certain variable names. The interpreter will
                use fuzzy matching to access these variables. For example, if
                :obj:`fuzz_state` has a variable :obj:`image`, the generated
                code can use :obj:`input_image` to access it. (default:
                :obj:`None`)
            keep_state (bool, optional):  If :obj:`True`, :obj:`state` and
                :obj:`fuzz_state` will be kept for later execution. Otherwise,
                they will be cleared. (default: :obj:`True`)

        Returns:
            Any: The value of the last statement (excluding "import") in the
                code. For this interpreter, the value of an expression is its
                value, the value of an "assign" statement is the assigned
                value, and the value of an "if" and "for" block statement is
                the value of the last statement in the block.
        """
        if state is not None:
            self.state.update(state)
        if fuzz_state is not None:
            self.fuzz_state.update(fuzz_state)

        try:
            expression = ast.parse(code)
        except SyntaxError as e:
            error_line = code.splitlines()[e.lineno - 1]
            raise InterpreterError(f"Syntax error in code at line {e.lineno}: {error_line}\nError: {e}")

        result = None
        for idx, node in enumerate(expression.body):
            try:
                line_result = self._execute_ast(node)
            except InterpreterError as e:
                if not keep_state:
                    self.clear_state()
                msg = (f"Evaluation of the code stopped at node {idx}. "
                       f"See:\n{e}")
                # More information can be provided by `ast.unparse()`,
                # which is new in python 3.9.
                raise InterpreterError(msg)
            if line_result is not None:
                result = line_result

        if not keep_state:
            self.clear_state()

        return result
    
    def clear_state(self) -> None:
        r"""Initialize :obj:`state` and :obj:`fuzz_state`"""
        self.state = self.action_space.copy()
        self.fuzz_state = {}
    
    # ast.Index is deprecated after python 3.9, which cannot pass type check,
    # but is still necessary for older versions.
    @typing.no_type_check
    def _execute_ast(self, expression: ast.AST) -> Any:
        if isinstance(expression, ast.Assign):
            # Assignment -> evaluate the assignment which should
            # update the state. We return the variable assigned as it may
            # be used to determine the final result.
            return self._execute_assign(expression)
        elif isinstance(expression, ast.Attribute):
            value = self._execute_ast(expression.value)
            return getattr(value, expression.attr)
        elif isinstance(expression, ast.AugAssign):
            return self._execute_augassign(expression)
        elif isinstance(expression, ast.BinOp):
            # Binary Operator -> return the result value
            return self._execute_binop(expression)
        elif isinstance(expression, ast.Call):
            # Function call -> return the value of the function call
            return self._execute_call(expression)
        elif isinstance(expression, ast.Compare):
            return self._execute_condition(expression)
        elif isinstance(expression, ast.Constant):
            # Constant -> just return the value
            return expression.value
        elif isinstance(expression, ast.Dict):
            # Dict -> evaluate all keys and values
            result: Dict = {}
            for k, v in zip(expression.keys, expression.values):
                if k is not None:
                    result[self._execute_ast(k)] = self._execute_ast(v)
                else:
                    result.update(self._execute_ast(v))
            return result
        elif isinstance(expression, ast.Expr):
            # Expression -> evaluate the content
            return self._execute_ast(expression.value)
        elif isinstance(expression, ast.For):
            return self._execute_for(expression)
        elif isinstance(expression, ast.FormattedValue):
            # Formatted value (part of f-string) -> evaluate the content
            # and return
            return self._execute_ast(expression.value)
        elif isinstance(expression, ast.FunctionDef):
            self.state[expression.name] = expression
            return None
        elif isinstance(expression, ast.If):
            # If -> execute the right branch
            return self._execute_if(expression)
        elif isinstance(expression, ast.Import):
            # Import -> add imported names in self.state and return None.
            self._execute_import(expression)
            return None
        elif isinstance(expression, ast.ImportFrom):
            self._execute_import_from(expression)
            return None
        elif hasattr(ast, "Index") and isinstance(expression, ast.Index):
            # cannot pass type check
            return self._execute_ast(expression.value)
        elif isinstance(expression, ast.JoinedStr):
            return "".join(
                [str(self._execute_ast(v)) for v in expression.values])
        elif isinstance(expression, ast.List):
            # List -> evaluate all elements
            return [self._execute_ast(elt) for elt in expression.elts]
        elif isinstance(expression, ast.Name):
            # Name -> pick up the value in the state
            return self._execute_name(expression)
        elif isinstance(expression, ast.Return):
            return self._execute_ast(expression.value)
        elif isinstance(expression, ast.Subscript):
            # Subscript -> return the value of the indexing
            return self._execute_subscript(expression)
        elif isinstance(expression, ast.Tuple):
            return tuple([self._execute_ast(elt) for elt in expression.elts])
        elif isinstance(expression, ast.UnaryOp):
            # Binary Operator -> return the result value
            return self._execute_unaryop(expression)
        else:
            # For now we refuse anything else. Let's add things as we need
            # them.
            raise InterpreterError(
                f"{expression.__class__.__name__} is not supported.")

    def _execute_assign(self, assign: ast.Assign) -> Any:
        targets = assign.targets
        result = self._execute_ast(assign.value)

        for target in targets:
            self._assign(target, result)
        return result

    def _assign(self, target: ast.expr, value: Any):
        if isinstance(target, ast.Name):
            self.state[target.id] = value
        elif isinstance(target, ast.Tuple):
            if not isinstance(value, tuple):
                raise InterpreterError(f"Expected type tuple, but got"
                                       f"{value.__class__.__name__} instead.")
            if len(target.elts) != len(value):
                raise InterpreterError(
                    f"Expected {len(target.elts)} values but got"
                    f" {len(value)}.")
            for t, v in zip(target.elts, value):
                self.state[self._execute_ast(t)] = v
        else:
            raise InterpreterError(f"Unsupported variable type. Expected "
                                   f"ast.Name or ast.Tuple, got "
                                   f"{target.__class__.__name__} instead.")

    def _execute_call(self, call: ast.Call) -> Any:
        callable_func = self._execute_ast(call.func)

        args = [self._execute_ast(arg) for arg in call.args]
        kwargs = {
            keyword.arg: self._execute_ast(keyword.value)
            for keyword in call.keywords
        }
        if isinstance(callable_func, ast.FunctionDef):
            old_state = self.state.copy()
            for param_name, arg_value in zip([param.arg for param in callable_func.args.args], args):
                self.state[param_name] = arg_value
            result = None
            for stmt in callable_func.body:
                result = self._execute_ast(stmt)
                if isinstance(stmt, ast.Return):
                    break
            self.state = old_state
            return result
        return callable_func(*args, **kwargs)

    def _execute_augassign(self, augassign: ast.AugAssign):
        current_value = self.state[augassign.target.id]
        increment_value = self._execute_ast(augassign.value)
        if not (isinstance(current_value, (int, float)) and isinstance(increment_value, (int, float))):
            raise InterpreterError(f"Invalid types for augmented assignment: {type(current_value)}, {type(increment_value)}")
        if isinstance(augassign.op, ast.Add):
            new_value = current_value + increment_value
        elif isinstance(augassign.op, ast.Sub):
            new_value = current_value - increment_value
        elif isinstance(augassign.op, ast.Mult):
            new_value = current_value * increment_value
        elif isinstance(augassign.op, ast.Div):
            new_value = current_value / increment_value
        #TODO - any other augassign operators that are missing
        else:
            raise InterpreterError(f"Augmented assignment operator {augassign.op} is not supported")
        self._assign(augassign.target, new_value)
        return new_value

    def _execute_subscript(self, subscript: ast.Subscript):
        index = self._execute_ast(subscript.slice)
        value = self._execute_ast(subscript.value)
        if not isinstance(subscript.ctx, ast.Load):
            raise InterpreterError(
                f"{subscript.ctx.__class__.__name__} is not supported for "
                "subscript.")
        if isinstance(value, (list, tuple)):
            return value[int(index)]
        if index in value:
            return value[index]
        if isinstance(index, str) and isinstance(value, Mapping):
            close_matches = difflib.get_close_matches(index,
                                                      list(value.keys()))
            if len(close_matches) > 0:
                return value[close_matches[0]]

        raise InterpreterError(f"Could not index {value} with '{index}'.")

    def _execute_name(self, name: ast.Name):
        if name.id in dir(builtins):
          return getattr(builtins, name.id)
        if isinstance(name.ctx, ast.Store):
            return name.id
        elif isinstance(name.ctx, ast.Load):
            return self._get_value_from_state(name.id)
        else:
            raise InterpreterError(f"{name.ctx} is not supported.")

    def _execute_condition(self, condition):
        if isinstance(condition, ast.BoolOp):
            if isinstance(condition.op, ast.And):
                results = [self._execute_ast(value) for value in condition.values]
                return all(results)
            elif isinstance(condition.op, ast.Or):
                results = [self._execute_ast(value) for value in condition.values]
                return any(results)
            else: #TODO - add any other BoolOps missing
                raise InterpreterError(f"Boolean operator {condition.op} is not supported")
        elif isinstance(condition, ast.Compare):
            if len(condition.ops) > 1:
                raise InterpreterError("Cannot evaluate conditions with multiple operators")
        if len(condition.ops) > 1:
            raise InterpreterError(
                "Cannot evaluate conditions with multiple operators")
        left = self._execute_ast(condition.left)
        comparator = condition.ops[0]
        right = self._execute_ast(condition.comparators[0])
        if isinstance(comparator, ast.Eq):
            return left == right
        elif isinstance(comparator, ast.NotEq):
            return left != right
        elif isinstance(comparator, ast.Lt):
            return left < right
        elif isinstance(comparator, ast.LtE):
            return left <= right
        elif isinstance(comparator, ast.Gt):
            return left > right
        elif isinstance(comparator, ast.GtE):
            return left >= right
        elif isinstance(comparator, ast.Is):
            return left is right
        elif isinstance(comparator, ast.IsNot):
            return left is not right
        elif isinstance(comparator, ast.In):
            return left in right
        elif isinstance(comparator, ast.NotIn):
            return left not in right
        else:
            raise InterpreterError("Unsupported condition type")

    def _execute_if(self, if_statement: ast.If):
        result = None
        if self._execute_condition(if_statement.test):
            for line in if_statement.body:
                line_result = self._execute_ast(line)
                if line_result is not None:
                    result = line_result
        else:
            for line in if_statement.orelse:
                line_result = self._execute_ast(line)
                if line_result is not None:
                    result = line_result
        return result

    def _execute_for(self, for_statement: ast.For):
        result = None
        for value in self._execute_ast(for_statement.iter):
            self._assign(for_statement.target, value)
            for line in for_statement.body:
                line_result = self._execute_ast(line)
                if line_result is not None:
                    result = line_result

        return result

    def _execute_import(self, import_module: ast.Import) -> None:
        for module in import_module.names:
            self._validate_import(module.name)
            alias = module.asname or module.name
            self.state[alias] = importlib.import_module(module.name)

    def _execute_import_from(self, import_from: ast.ImportFrom):
        if import_from.module is None:
            raise InterpreterError("\"from . import\" is not supported.")
        for import_name in import_from.names:
            full_name = import_from.module + f".{import_name.name}"
            self._validate_import(full_name)
            imported_module = importlib.import_module(import_from.module)
            alias = import_name.asname or import_name.name
            self.state[alias] = getattr(imported_module, import_name.name)

    def _validate_import(self, full_name: str):
        tmp_name = ""
        found_name = False
        for name in full_name.split("."):
            tmp_name += name if tmp_name == "" else f".{name}"
            if tmp_name in self.import_white_list:
                found_name = True
                return

        if not found_name:
            raise InterpreterError(f"It is not permitted to import modules "
                                   f"than module white list (try to import "
                                   f"{full_name}).")

    def _execute_binop(self, binop: ast.BinOp):
        left = self._execute_ast(binop.left)
        operator = binop.op
        right = self._execute_ast(binop.right)

        if isinstance(operator, ast.Add):
            return left + right
        elif isinstance(operator, ast.Sub):
            return left - right
        elif isinstance(operator, ast.Mult):
            return left * right
        elif isinstance(operator, ast.Div):
            return left / right
        elif isinstance(operator, ast.FloorDiv):
            return left // right
        elif isinstance(operator, ast.Mod):
            return left % right
        elif isinstance(operator, ast.Pow):
            return left**right
        elif isinstance(operator, ast.LShift):
            return left << right
        elif isinstance(operator, ast.RShift):
            return left >> right
        elif isinstance(operator, ast.MatMult):
            return left @ right
        else:
            raise InterpreterError(f"Operator not supported: {operator}")

    def _execute_unaryop(self, unaryop: ast.UnaryOp):
        operand = self._execute_ast(unaryop.operand)
        operator = unaryop.op

        if isinstance(operator, ast.UAdd):
            return +operand
        elif isinstance(operator, ast.USub):
            return -operand
        elif isinstance(operator, ast.Not):
            return not operand
        else:
            raise InterpreterError(f"Operator not supported: {operator}")

    def _get_value_from_state(self, key: str) -> Any:
        if key in self.state:
            return self.state[key]
        elif key in self.fuzz_state:
            return self.fuzz_state[key]
        else:
            raise InterpreterError(f"The variable `{key}` is not defined.")

class TextPrompt(str):
    r"""A class that represents a text prompt. The :obj:`TextPrompt` class
    extends the built-in :obj:`str` class to provide a property for retrieving
    the set of keywords in the prompt.

    Attributes:
        key_words (set): A set of strings representing the keywords in the
            prompt.
    """

    @property
    def key_words(self) -> Set[str]:
        """Returns a set of strings representing the keywords in the prompt."""
        # Regex to find format placeholders within the string, excluding escaped braces
        pattern = re.compile(r"\{([^{}]+)\}")
        found = pattern.findall(self)
        return set(found)

    def format(self, *args: Any, **kwargs: Any) -> 'TextPrompt':
        r"""Overrides the built-in :obj:`str.format` method to allow for
        default values in the format string. This is used to allow formatting
        the partial string.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            TextPrompt: A new :obj:`TextPrompt` object with the format string
                replaced with the formatted string.
        """
        default_kwargs = {key: '{' + f'{key}' + '}' for key in self.key_words}
        default_kwargs.update(kwargs)
        return TextPrompt(super().format(*args, **default_kwargs))

class CodePrompt(TextPrompt):
    r"""A class that represents a code prompt. It extends the :obj:`TextPrompt`
    class with a :obj:`code_type` property.

    Attributes:
        code_type (str, optional): The type of code. Defaults to None.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> 'CodePrompt':
        r"""Creates a new instance of the :obj:`CodePrompt` class.

        Args:
            *args (Any): Positional arguments.
            **kwargs (Any): Keyword arguments.

        Returns:
            CodePrompt: The created :obj:`CodePrompt` instance.
        """
        code_type = kwargs.pop('code_type', None)
        instance = super().__new__(cls, *args, **kwargs)
        instance._code_type = code_type
        return instance

    @property
    def code_type(self) -> Optional[str]:
        r"""Returns the type of code.

        Returns:
            Optional[str]: The type of code.
        """
        return self._code_type

    def set_code_type(self, code_type: str) -> None:
        r"""Sets the type of code.

        Args:
            code_type (str): The type of code.
        """
        self._code_type = code_type

    def execute(
        self, interpreter: Optional[PythonInterpreter] = None,
        user_variable: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, PythonInterpreter]:
        r"""Executes the code string by a given python interpreter.

        Args:
            interpreter (PythonInterpreter, optional): interpreter to be used
                during code execution. (default: :obj:`None`)
            user_variable (Optional[Dict[str, Any]]): variables that can be
                used in the code, which applying fuzzy matching, such as images
                or documents. (default: :obj:`None`)

        Returns:
            Tuple[Any, PythonInterpreter]: A tuple containing the execution
                result and the used interpreter. The execution result
                represents the value of the last statement (excluding "import")
                in the code. This value could potentially be the desired result
                of the LLM-generated code.
    """
        # NOTE: Only supports Python code for now.
        if not interpreter:
            interpreter = PythonInterpreter(action_space=globals())
        execution_res = interpreter.execute(self, fuzz_state=user_variable,
                                            keep_state=True)
        return execution_res, interpreter
