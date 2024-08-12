from __future__ import annotations  # Used for self type hints
from typing import Optional

import magicattr

from dsp.modules.lm import LM
from dspy.primitives.assertions import *
from dspy.primitives.module import BaseModule


#-------------------------------------------------------------------------------
#    Templates for the user-facing strings used by this module
#-------------------------------------------------------------------------------
# TODO: It might be good to control all user-facing strings from a central place

_ERR_MSG_SHARED_PREDICTORS = """This module shares the following predictor(s) \
with the other module: {pred_names}"""

_ERR_MSG_CLASS = """The class of this object does not match the class of the \
other object: '{sname}' != '{oname}'"""

_ERR_MSG_NUM_PREDICTORS = """Structurally equivalent modules must have the \
the number of predictors. The number of predictors for this module does not \
match that of the other module: {snum} != {onum}"""

_ERR_MSG_PREDICTOR_NAMES = """Module predictor names must match at \
corresponding indices for structural equivalency. The predictor names for this \
module and the other module do not match at index {ind}: {sname} != {oname}"""

_ERR_MSG_UNSET_DSPY_LM = """LM consistency violated: Expected all predictors' \
LMs to be set when dspy.settings.lm is set to 'NoneType', but some \
predictors have 'NoneType' LMs.

{lm_info}"""

_ERR_MSG_SET_DSPY_LM = """LM consistency violated: Expected all predictors' \
LMs to be 'NoneType' when dspy.settings.lm is set, but some predictors have \
LMs set.

{lm_info}"""

_INFO_LM_MODULE = """The LM set in dspy.settings.lm is an instance of '{lname}'
LMs set for the predictors in the module are:{pred_lm_info}"""

_INFO_LM_PRED = """
    LM for {pname} is an instance of '{lname}'"""

#-------------------------------------------------------------------------------
#    Classes
#-------------------------------------------------------------------------------


class ProgramMeta(type):
    pass
    # def __call__(cls, *args, **kwargs):
    #     obj = super(ProgramMeta, cls).__call__(*args, **kwargs)

    #     if issubclass(cls, Program) and not getattr(obj, "_program_init_called", False):
    #         obj._base_init()
    #         obj._program_init_called = True
    #     return obj


class Module(BaseModule, metaclass=ProgramMeta):
    def _base_init(self):
        self._compiled = False

    def __init__(self):
        self._compiled = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _assert_no_shared_predictor(
            self,
            other: Module
        ) -> Optional[AssertionError]:
        """Assert the module shares no predictor with another module.

        Args:
            other: The module to compare with.

        Raises:
            AssertionError: If the two modules share predictors.
        """
        # Get the names and IDs of the predictors in the two modules
        self_id_to_name = {id(p): n for n, p in self.named_predictors()}
        other_id_to_name = {id(p): n for n, p in other.named_predictors()}

        # Find the shared predictors
        shared_ids = set(self_id_to_name.keys()) & set(other_id_to_name.keys())
        if shared_ids:
            pred_names = ", ".join(self_id_to_name[id] for id in shared_ids)
            err_msg = _ERR_MSG_SHARED_PREDICTORS.format(pred_names=pred_names)
            raise AssertionError(err_msg)

    def _assert_structural_equivalency(
            self,
            other: Module
        ) -> Optional[AssertionError]:
        """Assert that the module is structurally equivalent to another module.
        
        Args:
            other: The module to compare with.

        Raises:
            AssertionError: If the modules are not structurally equivalent.
        """
        # Assert that the other object is an instance of the same class
        sname = self.__class__.__name__
        oname = other.__class__.__name__
        err_msg = _ERR_MSG_CLASS.format(sname=sname, oname=oname)
        assert isinstance(other, self.__class__), err_msg

        # Assert that the two modules have the same number of predictors
        snum = len(self.predictors())
        onum = len(other.predictors())
        err_msg = _ERR_MSG_NUM_PREDICTORS.format(snum=snum, onum=onum)
        assert snum == onum, err_msg

        # Assert that the two modules have structurally equivalent predictors
        # sharing the same names
        pzip = zip(self.named_predictors(), other.named_predictors())
        for ind, ((sname, spred), (oname, opred)) in enumerate(pzip):
            err_msg = _ERR_MSG_PREDICTOR_NAMES.format(
                ind=ind, sname=sname, oname=oname
            )
            assert sname == oname, err_msg

            spred._assert_structural_equivalency(opred)

    def _are_all_predictor_lms_set(self) -> bool:
        """Check if all predictors in the module have their LMs set.
        
        Returns:
            bool: True if all predictors have their LMs set, False otherwise.
        """
        return all(pred.lm is not None for pred in self.predictors())

    def _are_all_predictor_lms_unset(self) -> bool:
        """Check if all predictors in the module have their LMs unset.
        
        Returns:
            bool: True if all predictors have their LMs unset, False otherwise.
        """
        return all(pred.lm is None for pred in self.predictors())

    def _set_all_predictor_lms(
            self,
            lm: LM
        ):
        """Set the LMs of all predictors in the module.
        
        Args:
            lm: The LM to set.
        """
        for pred in self.predictors():
            pred.lm = lm

    def _unset_all_predictor_lms(self):
        """Unset the LMs of all predictors in the module."""
        for pred in self.predictors():
            pred.lm = None

    def _get_lm_info_str(self):
        """Get an informational string about the LMs affecting this module."""
        pred_lm_info = ""
        for pname, pred in self.named_predictors():
            lname = pred.lm.__class__.__name__
            pred_lm_info += _INFO_LM_PRED.format(pname=pname, lname=lname)

        lname = dspy.settings.lm.__class__.__name__
        info = _INFO_LM_MODULE.format(lname=lname, pred_lm_info=pred_lm_info)
        return info

    def _assert_lm_consistency(self) -> Optional[AssertionError]:
        """Assert that the module satisfies the LM consistency property.

        Ensures either that (1) all predictors in the module have their LMs set
        when dspy.settings.lm is NoneType, or, (2) none of the module predictors
        have their LMs set when dspy.settings.lm is set.

        Raises:
            AssertionError: If the LM consistency property is violated.
        """
        err_msg = None
        global_lm = dspy.settings.lm
        if global_lm is None and not self._are_all_predictor_lms_set():
            err_msg = _ERR_MSG_UNSET_DSPY_LM
        elif global_lm is not None and not self._are_all_predictor_lms_unset():
            err_msg = _ERR_MSG_SET_DSPY_LM
        
        if err_msg is not None:
            lm_info = self._get_lm_info_str()
            err_msg = err_msg.format(lm_info=lm_info)
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
