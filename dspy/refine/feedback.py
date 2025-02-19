from typing import Union

from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature


class GenerateFeedback(Signature):
    """
    Based on each metric value and metric definition for the inputs-outputs pair, provide feedback the DSPy module
     along with submodules in order to improve the metric values at the retry. Only provide feedback for built-in
     classses, e.g., dspy.Predict, dspy.Module, dspy.ChainOfThought and so on. If an attribute is a list, make sure
     you look into every element. It's also possible that some components are not related to the certain score, we
     should skip generating feedback if it is the case.
    """

    metrics: list[str] = InputField(desc="The definition of each scoring criterion")
    metric_values: list[Union[int, float, bool]] = InputField(desc="The value of each metric, the higher the better")
    module_inputs: dict = InputField(desc="The inputs of the DSPy module")
    module_outputs: dict = InputField(desc="The outputs of the DSPy module")
    source_code: str = InputField(desc="The source code of the DSPy module")
    feedback: dict[str, list[str]] = OutputField(
        desc="Feedback for the DSPy module in general, along with feedback for each submodule in the DSPy model, only "
        "provide feedback for attributes in `__init__` method that is a built-in class of dspy. The key should be the "
        "attribute name, e.g., `self.cot` or `self.predict`. If the attribute is a list, write the key as "
        "`self.cots[0]`, `self.predicts[1]` and so on. The feedback should be "
        "a list of strings, corresponding to each score function in `metrics`."
    )
