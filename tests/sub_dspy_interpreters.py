"""Reference SUB_DSPY worker backend for tests.

dspy-interpreters' ``SubprocessInterpreter`` honoring the sub-dspy contract: it declares the
capability and, on first use, defines ``dspy_interpreter_factory`` inside the worker as a class that
does the same for its own children, so recursion stays native all the way down.
"""

from typing import Any

from dspy_interpreters import SubprocessInterpreter

from dspy.primitives.code_interpreter import SUB_DSPY_FACTORY_NAME, InterpreterCapability

_FACTORY_SOURCE_VAR = "_dspy_factory_source"

# Executed inside a worker (with its own text injected as _FACTORY_SOURCE_VAR); the class it defines
# re-executes that text in each child it starts.
NESTED_FACTORY_SOURCE = f"""
from dspy_interpreters import SubprocessInterpreter as _SubprocessInterpreter
from dspy.primitives.code_interpreter import InterpreterCapability as _InterpreterCapability


class {SUB_DSPY_FACTORY_NAME}(_SubprocessInterpreter):
    capabilities = _InterpreterCapability.SUB_DSPY
    _contract_installed = False

    def execute(self, code, variables=None):
        if not self._contract_installed:
            super().execute({_FACTORY_SOURCE_VAR}, variables={{"{_FACTORY_SOURCE_VAR}": {_FACTORY_SOURCE_VAR}}})
            self._contract_installed = True
        return super().execute(code, variables)
"""


class SubDspySubprocessInterpreter(SubprocessInterpreter):
    capabilities = InterpreterCapability.SUB_DSPY
    _contract_installed = False

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        if not self._contract_installed:
            super().execute(NESTED_FACTORY_SOURCE, variables={_FACTORY_SOURCE_VAR: NESTED_FACTORY_SOURCE})
            self._contract_installed = True
        return super().execute(code, variables)
