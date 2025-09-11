from dspy.adapters import History
from dspy.primitives import Prediction
from dspy.primitives.module import Module
from dspy.signatures import InputField


class Stateful(Module):
    """Adds automatic conversation history management to any DSPy module.

    Example:
        >>> qa = dspy.Predict("question -> answer")
        >>> stateful_qa = dspy.Stateful(qa)
        >>> response1 = stateful_qa(question="What's Python?")
        >>> response2 = stateful_qa(question="Is it fast?")  # Has context from previous turn

        >>> # Works with any module type
        >>> stateful_cot = dspy.Stateful(dspy.ChainOfThought("question -> answer"))
    """

    def __init__(self, module):
        super().__init__()
        self.module = module.deepcopy()
        self._history = History(messages=[])

        for pred in self.module.predictors():
            pred.signature = pred.signature.prepend(
                name="history",
                field=InputField(),
                type_=History
            )

    def forward(self, **kwargs):
        kwargs["history"] = self._history
        res = self.module(**kwargs)

        # Build history entry
        turn = {k: v for k, v in kwargs.items() if k != "history"}
        if isinstance(res, Prediction):
            turn.update(dict(res))
        elif isinstance(res, dict):
            turn.update(res)
        else:
            turn["output"] = res

        self._history.messages.append(turn)
        return res

    def reset_history(self):
        self._history = History(messages=[])
