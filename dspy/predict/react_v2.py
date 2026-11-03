import warnings

from dspy.predict.react import ReAct


class ReActV2(ReAct):
    """Deprecated compatibility name for :class:`dspy.ReAct`.

    .. deprecated:: 3.5
        Use ``dspy.ReAct`` instead. This alias will be removed in DSPy 3.6.
    """

    def __init__(self, signature, tools, max_iters=20):
        warnings.warn(
            "The dspy.ReActV2 implementation has now replaced dspy.ReAct. "
            "You can migrate directly from dspy.ReActV2 to dspy.ReAct with no behavior changes. "
            "The dspy.ReActV2 name is deprecated and will be removed in DSPy 3.6.",
            DeprecationWarning,
            stacklevel=3,  # Skip ProgramMeta.__call__ as well as this constructor.
        )
        super().__init__(signature, tools, max_iters=max_iters)
