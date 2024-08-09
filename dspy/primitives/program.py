from __future__ import annotations  # Required for type hints of the class itself, see Module.is_structurally_equivalent
from typing import Optional

import magicattr

from dsp.modules.lm import LM
from dspy.primitives.assertions import *
from dspy.primitives.module import BaseModule


class ProgramMeta(type):
    pass
    # def __call__(cls, *args, **kwargs):
    #     obj = super(ProgramMeta, cls).__call__(*args, **kwargs)

    #     if issubclass(cls, Program) and not getattr(obj, "_program_init_called", False):
    #         obj._base_init()
    #         obj._program_init_called = True
    #     return obj


# TODO: Replace print statements with logging
class Module(BaseModule, metaclass=ProgramMeta):
    def _base_init(self):
        self._compiled = False

    def __init__(self):
        self._compiled = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _is_structurally_equivalent(self, other: Module) -> bool:
        """ Check if two modules are structurally equivalent by comparing the names and structure of their predictors.
        
        Args:
            other: The module to compare with.
        
        Returns:
            bool: `True` if the modules are structurally equivalent, `False` otherwise.
        """
        # Check if the two modules are instances of the Program class
        if not isinstance(other, self.__class__):
            return False

        # Check if the two modules have the same number of predictors
        if len(self.predictors()) != len(other.predictors()):
            return False

        # Check if the two modules have structurally equivalent predictors
        for (name1, pred1), (name2, pred2) in zip(self.named_predictors(), other.named_predictors()):
            if name1 != name2 or not pred1._is_structurally_equivalent(pred2):
                return False

        return True

    def _has_shared_predictor(self, other: Module) -> bool:
        """ Check if two modules share a predictor that points to the same Python object.
        
        Args:
            other: The module to compare with.
        
        Returns:
            bool: `True` if the modules share a predictor, `False` otherwise.
        """
        for pred1, pred2 in zip(self.predictors(), other.predictors()):
            if id(pred1) == id(pred2):
                return True
        return False

    def _is_all_predictor_lms_set(self) -> bool:
        """ Check if all predictors in the module have their LMs set.
        
        Returns:
            bool: `True` if all predictors have their LMs set, `False` otherwise.
        """
        return all(predictor.lm is not None for predictor in self.predictors())

    def _is_all_predictor_lms_unset(self) -> bool:
        """ Check if all predictors in the module have their LMs unset.
        
        Returns:
            bool: `True` if all predictors have their LMs unset, `False` otherwise.
        """
        return all(predictor.lm is None for predictor in self.predictors())

    def _set_all_predictor_lms(self, lm: LM):
        """ Set the LM of all predictors in the module.
        
        Args:
            lm: The LM to set.
        """
        for predictor in self.predictors():
            predictor.lm = lm

    def _unset_all_predictor_lms(self):
        """ Unset the LM of all predictors in the module. """
        for predictor in self.predictors():
            predictor.lm = None

    def _print_lm_information(self):
        print(f"The LM set in dspy.settings.lm is {dspy.settings.lm}")
        print("Looping through all predictors in the program:")
        for name, predictor in self.named_predictors():
            print(f"    Predictor {name} is set to LM {predictor.lm}")

    def _assert_lm_consistency(self) -> Optional[AssertionError]:
        """ Check if the module satisfies the LM consistency property.

        Ensures that (1) all predictors in the module have their LMs set when `dspy.settings.lm` is `None`, and, (2)
        none of the module predictors have their LMs set (to a same or different LM) when `dspy.settings.lm` is set.

        Raises:
            AssertionError: If the module does not satisfy the LM consistency property.
        """
        err_msg = None
        if dspy.settings.lm is None and not self._is_all_predictor_lms_unset():
            err_msg = "LM consistency property violated: LM is not set in a module predictor when dspy.settings.lm is None."
        elif dspy.settings.lm is not None and not self._is_all_predictor_lms_set():
            err_msg = "LM consistency property violated: LM is set in a module predictor when dspy.settings.lm is set."
        
        if err_msg is not None:
            self._print_lm_information()
            raise AssertionError(err_msg)

    def named_predictors(self):
        from dspy.predict.predict import Predict

        return [(name, param) for name, param in self.named_parameters() if isinstance(param, Predict)]

    def predictors(self):
        return [param for _, param in self.named_predictors()]

    def __repr__(self):
        s = []

        for name, param in self.named_predictors():
            s.append(f"{name} = {param}")

        return "\n".join(s)

    def map_named_predictors(self, func):
        """Applies a function to all named predictors."""
        for name, predictor in self.named_predictors():
            set_attribute_by_name(self, name, func(predictor))
        return self

    def activate_assertions(self, handler=backtrack_handler, **handler_args):
        """
        Activates assertions for the module.
        The default handler is the backtrack_handler.
        """
        assert_transform_module(self, handler, **handler_args)
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


def set_attribute_by_name(obj, name, value):
    magicattr.set(obj, name, value)

Program = Module
