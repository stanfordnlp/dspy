from __future__ import annotations  # Required for type hints of the class itself, see Module._is_structurally_equivalent
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

    def _assert_no_shared_predictor(self, other: Module) -> Optional[AssertionError]:
        """ Assert that the module does not share any predictors with another module.

        Args:
            other: The module to compare with.

        Raises:
            AssertionError: If the two modules have any predictor that points to the same Python object.
        """
        # Get the names and IDs of the predictors in the two modules
        self_id_to_name = {id(predictor): name for name, predictor in self.named_predictors()}
        other_id_to_name = {id(predictor): name for name, predictor in other.named_predictors()}

        # Find the shared predictors
        shared_predictors = set(self_id_to_name.keys()) & set(other_id_to_name.keys())
        if shared_predictors:
            shared_predictor_self_names = ", ".join(self_id_to_name[id] for id in shared_predictors)
            err_msg = f"This module shares the following predictor(s) with the other module: {shared_predictor_self_names}"
            raise AssertionError(err_msg)

    def _assert_structural_equivalency(self, other: Module) -> Optional[AssertionError]:
        """ Assert that the module is structurally equivalent to another module.
        
        Args:
            other: The module to compare with.

        Raises:
            AssertionError: If the two modules are not structurally equivalent.
        """
        # Assert that the other object is an instance of the same class
        err_msg = f"Modules must be instances of the same class for structural equivalency: '{self.__class__.__name__}' != '{other.__class__.__name__}'"
        assert isinstance(other, self.__class__), err_msg

        # Assert that the two modules have the same number of predictors
        err_msg = f"Structurally equivalent modules must have the same number of predictors. The number of predictors for this module does not match that of the other module: {len(self.predictors())} != {len(other.predictors())}"
        assert len(self.predictors()) == len(other.predictors()), err_msg

        # Assert that the two modules have structurally equivalent predictors sharing the same names
        for ind, ((self_name, self_pred), (other_name, other_pred)) in enumerate(zip(self.named_predictors(), other.named_predictors())):
            err_msg = f"Module predictor names must match at corresponding indices for structural equivalency. The predictor names for this module and the other module do not match at index {ind}: {self_name} != {other_name}"
            assert self_name == other_name, err_msg

            self_pred._assert_structural_equivalency(other_pred)

    def _are_all_predictor_lms_set(self) -> bool:
        """ Check if all predictors in the module have their LMs set.
        
        Returns:
            bool: `True` if all predictors have their LMs set, `False` otherwise.
        """
        return all(predictor.lm is not None for predictor in self.predictors())

    def _are_all_predictor_lms_unset(self) -> bool:
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

    def _get_lm_info_str(self):
        """ Get a string representation of the LM information of the module. """
        info = f"The LM set in 'dspy.settings.lm is' an instance of '{dspy.settings.lm.__class__.__name__}'\n"
        info += "Looping through all the predictors in the program:\n"
        for name, predictor in self.named_predictors():
            info += f"    Predictor {name} is an instance of '{predictor.lm.__class__.__name__}'\n"
        return info

    def _assert_lm_consistency(self) -> Optional[AssertionError]:
        """ Check if the module satisfies the LM consistency property.

        Ensures either that (1) all predictors in the module have their LMs set when `dspy.settings.lm` is `None`, or,
        (2) none of the module predictors have their LMs set (to a same or different LM) when `dspy.settings.lm` is set.

        Raises:
            AssertionError: If the module does not satisfy the LM consistency property.
        """
        err_msg = None
        if dspy.settings.lm is None and not self._are_all_predictor_lms_set():
            err_msg = "LM consistency violated: Expected all predictors' LMs to be set when `dspy.settings.lm` is set to 'NoneType', but some predictors have 'NoneType' LMs."
        elif dspy.settings.lm is not None and not self._are_all_predictor_lms_unset():
            err_msg = "LM consistency violated: Expected all predictors' LMs to be 'NoneType' when `dspy.settings.lm` is set, but some predictors have LMs set."
        
        if err_msg is not None:
            lm_info_str = self._get_lm_info_str()
            err_msg = f"{err_msg}\n\n{lm_info_str}"
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
