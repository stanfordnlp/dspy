import copy
import inspect

from dspy.primitives.module import BaseModule
from dspy.primitives.assertions import *
import re


class ProgramMeta(type):
    """
    Metaclass for the Program class.
    """
    pass
    # def __call__(cls, *args, **kwargs):
    #     obj = super(ProgramMeta, cls).__call__(*args, **kwargs)

    #     if issubclass(cls, Program) and not getattr(obj, "_program_init_called", False):
    #         obj._base_init()
    #         obj._program_init_called = True
    #     return obj


class Module(BaseModule, metaclass=ProgramMeta):
    """
    The Module class represents a module in the DSPy framework.
    It provides methods for managing predictors and handling calls.
    """

    def _base_init(self):
        """
        Initialize the base attributes of the Module instance.
        """
        self._compiled = False

    def __init__(self):
        """
        Initialize a new instance of the Module class.
        """
        self._compiled = False

    def __call__(self, *args, **kwargs):
        """
        Call the Module instance.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the forward method.
        """
        return self.forward(*args, **kwargs)

    def named_predictors(self):
        """
        Get the named predictors of the Module instance.

        Returns:
            list: A list of tuples, where each tuple contains the name of a predictor and the predictor itself.
        """
        from dspy.predict.predict import Predict

        named_parameters = self.named_parameters()
        return [
            (name, param)
            for name, param in named_parameters
            if isinstance(param, Predict)
        ]

    def predictors(self):
        """
        Get the predictors of the Module instance.

        Returns:
            list: A list of predictors.
        """
        return [param for _, param in self.named_predictors()]

    def __repr__(self):
        """
        Get a string representation of the Module instance.

        Returns:
            str: A string representation of the Module instance.
        """
        s = []

        for name, param in self.named_predictors():
            s.append(f"{name} = {param}")

        return "\n".join(s)

    def map_named_predictors(self, func):
        """
        Apply a function to all named predictors of the Module instance.
    
        Args:
            func (function): The function to apply.
    
        Returns:
            Module: The Module instance itself.
        """
        for name, predictor in self.named_predictors():
            set_attribute_by_name(self, name, func(predictor))
        return self

    # def __deepcopy__(self, memo):
    #     # memo is a dict of id's to copies already made during the current call
    #     # Check if the object is already copied
    #     if id(self) in memo:
    #         return memo[id(self)]

    #     print(f"Deep copying {self.__class__.__name__}...")

    #     new_copy = copy.copy(self)
    #     memo[id(self)] = new_copy

    #     for k, v in self.__dict__.items():
    #         print(f"Copying attribute {k} of type {type(v)}...")
    #         setattr(new_copy, k, copy.deepcopy(v, memo))
    #         print("Done")

    #     return new_copy


# FIXME(Shangyint): This may cause some problems for nested patterns.
def set_attribute_by_name(obj, name, value):
    # Regular expressions for different patterns
    module_pattern = re.compile(r"^([^.]+)\.(.+)$")
    list_pattern = re.compile(r"^([^\[]+)\[([0-9]+)\]$")
    dict_pattern = re.compile(r"^([^\[]+)\['([^']+)'\]$")

    # Match for module.attribute pattern
    module_match = module_pattern.match(name)
    if module_match:
    # Regular expressions for different patterns
    module_pattern = re.compile(r"^([^.]+)\.(.+)$")
    list_pattern = re.compile(r"^([^\[]+)\[([0-9]+)\]$")
    dict_pattern = re.compile(r"^([^\[]+)\['([^']+)'\]$")

    # Match for module.attribute pattern
    module_match = module_pattern.match(name)
    if module_match:
        module_name, sub_name = module_match.groups()
        sub_obj = getattr(obj, module_name)
        set_attribute_by_name(sub_obj, sub_name, value)
        return

    # Match for list[index] pattern
    list_match = list_pattern.match(name)
    if list_match:
        list_name, index = list_match.groups()
        getattr(obj, list_name)[int(index)] = value
        return

    # Match for dict['key'] pattern
    dict_match = dict_pattern.match(name)
    if dict_match:
        dict_name, key = dict_match.groups()
        getattr(obj, dict_name)[key] = value
        return

    # Default case for simple attributes
    setattr(obj, name, value)


Program = Module
