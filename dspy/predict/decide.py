"""Compatibility constructor for closed-set Predict instances."""

from dspy.adapters.decision_adapter import DecisionAdapter
from dspy.predict.predict import Predict
from dspy.utils.annotation import experimental


@experimental
def Decide(signature, *, client=None, callbacks=None):  # noqa: N802 - preserve the experimental constructor spelling
    """Return Predict with a DecisionAdapter and optional TypeSafe backend.

    Prefer ``dspy.Predict(signature, adapter=DecisionAdapter(), backend=client)``.
    This compatibility spelling is a factory, not a separate module type.
    """
    return Predict(signature, adapter=DecisionAdapter(), backend=client, callbacks=callbacks)
