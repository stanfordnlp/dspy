from dspy.primitives.base_module import BaseModule
from dspy.primitives.example import Example
from dspy.primitives.interpreter import FinalAnswerResult, Interpreter, InterpreterError
from dspy.primitives.local_interpreter import PythonInterpreter
from dspy.primitives.module import Module
from dspy.primitives.prediction import Completions, Prediction

__all__ = [
    "BaseModule",
    "Completions",
    "Example",
    "FinalAnswerResult",
    "Interpreter",
    "InterpreterError",
    "Module",
    "Prediction",
    "PythonInterpreter",
]
