try:
    import math_equivalence
except ImportError:
    raise ImportError("MATH's metric requires `pip install git+https://github.com/hendrycks/math.git`")


def is_equiv(golden, pred):
    return math_equivalence.is_equiv(golden, pred)


def is_equiv_dspy(example, pred, trace=None):
    return is_equiv(example.answer, pred.answer)
