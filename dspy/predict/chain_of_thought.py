from dsp import GPT3, settings
from dspy.signatures.field import OutputField
from dspy.primitives.program import Module
from dspy.predict.predict import Predict

# TODO: FIXME: Insert this right before the *first* output field. Also rewrite this to use the new signature system.

# Let's focus on the activated case. It's a predictor with the expanded signature.
# Now, when deactivated, it's a predictor with the original signature.
# When activate is None, though, we need the expanded one but during forward we need to pass the right signature.

"""
class ChainOfThought(dspy.Module):
    def __init__(self, signature):

        input_fields, output_fields = dspy.process_signature(signature)
        output_fields = dict(rationale=dspy.OutputField(prefix="Reasoning: Let's think step by step."), **output_fields)
        self.signature = dspy.Signature(input_fields, output_fields)
        
        self.predict = dspy.Predict(self.signature)
    
    def forward(self, **kwargs):
        return self.predict(**kwargs)

# How this should look like. But with also passing signature=simpler_signature to the predict module *if* deactivated.
"""


class ChainOfThought(Module):
    """
    The ChainOfThought class is a module that uses a Predict object to make predictions based on a given signature.
    It can be activated or deactivated, and it can have an extended signature that includes a rationale.

    Attributes:
        predict (Predict): The Predict object used for making predictions.
    """

    def __init__(self, signature, rationale_type=None, activated=True, **config):
        """
        The constructor for the ChainOfThought class.

        Parameters:
            signature (Signature): The signature used for making predictions.
            rationale_type (OutputField, optional): The rationale type for the extended signature. Defaults to None.
            activated (bool, optional): A flag indicating whether the module is activated. Defaults to True.
            **config: Arbitrary keyword arguments.
        """
        self.predict = Predict(signature, **config)
        self.predict.activated = activated
        self.predict.extended_signature = self._get_extended_signature(rationale_type)

    def forward(self, **kwargs):
        """
        The forward method for the ChainOfThought class.

        Parameters:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Predict: The result of the forward method of the Predict object.
        """
        signature = kwargs.pop("new_signature", None) or self._get_signature()
        return self.predict.forward(signature=signature, **kwargs)

    def dump_state(self):
        """
        The dump_state method for the ChainOfThought class.

        Returns:
            dict: A dictionary containing the state of the Predict object and the extended signature.
        """
        state = self.predict.dump_state()
        state["extended_signature_instructions"] = self.predict.extended_signature.instructions

        *_, last_key = self.signature.fields.keys()
        state["extended_signature_prefix"] = self.predict.extended_signature.fields[last_key].json_schema_extra['prefix']
        return state

    def load_state(self, state):
        """
        The load_state method for the ChainOfThought class.

        Parameters:
            state (dict): A dictionary containing the state of the Predict object and the extended signature.
        """
        self.predict.load_state(state)
        self._reconstruct_signature(state)

    def _get_extended_signature(self, rationale_type):
        """
        A private method to get the extended signature for the Predict object.

        Parameters:
            rationale_type (OutputField): The rationale type for the extended signature.

        Returns:
            Signature: The extended signature for the Predict object.
        """
        *_keys, last_key = self.predict.signature.output_fields.keys()
        rationale_type = rationale_type or OutputField(
            prefix="Reasoning: Let's think step by step in order to",
            desc="${produce the " + last_key + "}. We ...",
        )
        return self.predict.signature.prepend("rationale", rationale_type, type_=str)

    def _get_signature(self):
        """
        A private method to get the signature for the Predict object.

        Returns:
            Signature: The signature for the Predict object.
        """
        if self.predict.activated is True or (
                self.predict.activated is None and isinstance(settings.lm, GPT3)
        ):
            return self.predict.extended_signature
        else:
            return self.predict.signature

    def _reconstruct_signature(self, state):
        """
        A private method to reconstruct the signature for the Predict object.

        Parameters:
            state (dict): A dictionary containing the state of the Predict object and the extended signature.
        """
        if "extended_signature_instructions" in state:
            instructions = state["extended_signature_instructions"]
            self.predict.extended_signature.instructions = (
                self.predict.extended_signature.with_instructions(instructions))

        if "extended_signature_prefix" in state:
            prefix = state["extended_signature_prefix"]
            *_, last_key = self.predict.extended_signature.fields.keys()
            self.extended_signature = self.predict.extended_signature.with_updated_fields(last_key, prefix=prefix)

"""
TODO: In principle, we can update the field's prefix during forward too to fill any thing based on the input args.

IF the user didn't overwrite our default rationale_type.
"""
