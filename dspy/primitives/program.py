from __future__ import annotations  # Used for self type hints
from typing import Optional

import magicattr

from dsp.modules.lm import LM
import dspy
from dspy.primitives.assertions import *
from dspy.primitives.module import BaseModule


#-------------------------------------------------------------------------------
#    Templates for the user-facing strings used by this module
#-------------------------------------------------------------------------------

_ERR_MSG_NUM_PREDICTORS = """Structurally equivalent programs must have the \
the number of predictors. The number of predictors for the two modules do not \
match: {num1} != {num2}"""

_ERR_MSG_PREDICTOR_NAMES = """Program predictor names must match at \
corresponding indices for structural equivalency. The predictor names for the \
programs do not match at index {ind}: '{name1}' != '{name2}'"""

_ERR_MSG_SHARED_PREDICTORS = """The programs share the following predictor(s) \
with each other: {pred_names}"""

_ERR_MSG_UNSET_DSPY_LM = """LM consistency violated: Expected all predictors’ \
LMs to be set when dspy.settings.lm is set to ‘NoneType’, but some \
predictors have 'NoneType' LMs.

{lm_info}"""

_ERR_MSG_SET_DSPY_LM = """LM consistency violated: Expected all predictors' \
LMs to be 'NoneType' when 'dspy.settings.lm' is set, but some predictors have \
LMs set.

{lm_info}"""

_ERR_MSG_PROGRAM_LM = """Setting or getting the LM of a program is an \
experimental feature. Please enable the 'dspy.settings.experimental' flag to \
use these feature."""

_INFO_LM_PROGRAM = """The LM set in 'dspy.settings.lm' is an instance of \
'{lname}' LMs set for the predictors in the module are:{pred_lm_info}"""

_INFO_LM_PRED = """
    LM for {pname} is an instance of ’{lname}’"""


#-------------------------------------------------------------------------------
#    Helper functions
#-------------------------------------------------------------------------------

def assert_structural_equivalency_for_programs(
        program1: object,
        program2: object,
    ) -> Optional[AssertionError]:
    """Assert that the program is structurally equivalent to another program.

    Args:
        program1: The program to compare with.
        program2: The program to compare with.

    Raises:
        AssertionError: If the programs are not structurally equivalent.
    """
    # TODO: The following import can be removed if "Program" class can be moved
    # to its own module, since "Module" is the base class for both the “Program”
    # and "Predict" classes.
    # Function level import to prevent issues with circular imports.
    from dspy.predict.predict import (
        assert_structural_equivalency_for_predictors
    )

    # Assert that the objects are predictors
    assert isinstance(program1, Program)
    assert isinstance(program2, Program)

    # Assert that the two programs have the same number of predictors
    num1 = len(program1.predictors())
    num2 = len(program2.predictors())
    err_msg = _ERR_MSG_NUM_PREDICTORS.format(num1=num1, num2=num2)
    assert num1 == num2, err_msg

    # Assert that the two programs have structurally equivalent predictors
    # sharing the same names
    pzip = zip(program1.named_predictors(), program2.named_predictors())
    for ind, ((name1, pred1), (name2, pred2)) in enumerate(pzip):
        # Check for predictor name match
        err_msg = _ERR_MSG_PREDICTOR_NAMES.format(
            ind=ind, name1=name1, name2=name2
        )
        assert name1 == name2, err_msg

        # Check for structural equivalency of the predictors
        assert_structural_equivalency_for_predictors(pred1, pred2)


def assert_no_shared_predictor_for_programs(
        program1: Program,
        program2: Program,
    ) -> Optional[AssertionError]:
    """Assert that the programs shares no predictor with each other.

    Args:
        program1: The program to compare with.
        program2: The program to compare with.

    Raises:
        AssertionError: If the two programs share predictors.
    """
    # Get the names and IDs of the predictors in the two programs
    id_to_name1 = {id(p): n for n, p in program1.named_predictors()}
    id_to_name2 = {id(p): n for n, p in program2.named_predictors()}

    # Find the shared predictors
    shared_ids = set(id_to_name1.keys()) & set(id_to_name2.keys())
    if shared_ids:
        pred_names = ", ".join(id_to_name1[id] for id in shared_ids)
        err_msg = _ERR_MSG_SHARED_PREDICTORS.format(pred_names=pred_names)
        raise AssertionError(err_msg)


def are_all_predictor_lms_set_for_program(program: Program) -> bool:
    """Check if all predictors in the program have their LMs set.

    Returns:
        bool: `True` if all predictors have their LMs set, `False` otherwise.
    """
    return all(pred.lm is not None for pred in program.predictors())


def are_all_predictor_lms_unset_for_program(program: Program) -> bool:
    """Check if all predictors in the program have their LMs unset.

    Returns:
        bool: True if all predictors have their LMs unset, False otherwise.
    """
    return all(pred.lm is None for pred in program.predictors())


def get_lm_info_str_for_program(program: Program):
    """Get an informational string about the LMs affecting this program."""
    # The informational string to be returned
    pred_lm_info = ""

    # Get the LM information for each predictor in the program
    for pname, pred in program.named_predictors():
        lname = pred.lm.__class__.__name__
        pred_lm_info += _INFO_LM_PRED.format(pname=pname, lname=lname)

    # Get the LM information from the global setting; augment the information
    # string
    lname = dspy.settings.lm.__class__.__name__
    info = _INFO_LM_PROGRAM.format(lname=lname, pred_lm_info=pred_lm_info)

    return info


def assert_lm_consistency_for_program(
        program: Program
    ) -> Optional[AssertionError]:
    """Assert that the program satisfies the LM consistency property.

    Ensure either that (1) all predictors in the module have their LMs set when
    `dspy.settings.lm` is `NoneType`, or, (2) none of the program predictors
    have their LMs set when `dspy.settings.lm` is set.

    Raises:
        AssertionError: If the LM consistency property is violated.
    """
    # The error message to be raised if the LM consistency property is violated
    err_msg = None

    # Check if the LM consistency property is violated
    global_flag = dspy.settings.lm is not None
    if not global_flag and not are_all_predictor_lms_set_for_program(program):
        err_msg = _ERR_MSG_UNSET_DSPY_LM
    elif global_flag and not are_all_predictor_lms_unset_for_program(program):
        err_msg = _ERR_MSG_SET_DSPY_LM

    # Raise an error if the LM consistency property is violated
    if err_msg is not None:
        lm_info = get_lm_info_str_for_program(program)
        err_msg = err_msg.format(lm_info=lm_info)
        raise AssertionError(err_msg)


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

    def set_lm(self, lm: LM) -> Optional[AssertionError]:
        """Set the LM for all predictors in the module.

        This is an experimental method, which requires dsp.settings.experimental
        to be enabled.

        Args:
            lm: The LM to set for all predictors.

        Raises:
            AssertionError: If the dspy experimental setting is not enabled.
        """
        # Check if the experimental setting is enabled
        err_msg = _ERR_MSG_PROGRAM_LM
        assert dspy.settings.experimental, err_msg

        # Set the LM for all predictors
        for _, pred in self.named_predictors():
            pred.lm = lm

    def named_predictors(self):
        from dspy.predict.predict import Predict

        return [(name, param) for name, param in self.named_parameters() if isinstance(param, Predict)]

    def predictors(self):
        return [param for _, param in self.named_predictors()]
    
    def set_lm(self, lm):
        import dspy
        assert dspy.settings.experimental, "Setting the lm is an experimental feature."

        for _, param in self.named_predictors():
            param.lm = lm

    def get_lm(self):
        import dspy
        assert dspy.settings.experimental, "Getting the lm is an experimental feature."

        all_used_lms = [param.lm for _, param in self.named_predictors()]

        if len(set(all_used_lms)) == 1:
            return all_used_lms[0]
        
        raise ValueError("Multiple LMs are being used in the module.")

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
