class Parameter:
    """Vestigial marker used by module traversal, state dumps, and deepcopy.

    Not a public concept: the optimizer-facing predictor is ``dspy.Predict``,
    and custom predictors subclass it. Kept only because ``named_parameters()``
    and ``BaseModule.deepcopy`` key off it.
    """
