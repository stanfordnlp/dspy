import dspy.dsp as dsp
import dspy
from dspy.signatures.signature import ensure_signature
from dspy.predict import Predict

class FieldwiseChainOfThought(Predict):
    def __init__(self, signature, activated=True, **config):
        super().__init__(signature, **config)
        self.activated = activated
        
        signature = ensure_signature(self.signature)
        
        # Build extended signature by prepending reasoning field before each output field
        extended_signature = signature
        output_fields = list(signature.output_fields.keys())
        
        # Insert reasoning fields in reverse order to maintain correct positions
        for i, field_name in enumerate(reversed(output_fields)):
            reasoning_field_name = f"{field_name}_reasoning"
            reasoning_field = dspy.OutputField(
                prefix=f"Reasoning for {field_name}: Let's think step by step in order to",
                desc="${produce the " + field_name + "}. We ...",
            )
            # Insert reasoning field right before the output field
            # Position: negative index from end, accounting for fields we haven't processed yet
            position = -(len(output_fields) - i)
            extended_signature = extended_signature.insert(
                position,
                reasoning_field_name,
                reasoning_field,
                type_=str
            )
        
        self.extended_signature = extended_signature

    def forward(self, **kwargs):
        new_signature = kwargs.pop("new_signature", None)
        
        if new_signature is None:
            if self.activated is True or (
                self.activated is None and isinstance(dsp.settings.lm, dsp.GPT3)
            ):
                signature = self.extended_signature
            else:
                signature = self.signature
        else:
            signature = new_signature
        
        return super().forward(signature=signature, **kwargs)

    def dump_state(self):
        state = super().dump_state()
        
        # Cache the extended signature instructions
        state["extended_signature_instructions"] = self.extended_signature.instructions
        
        # Cache prefixes for all reasoning fields
        state["reasoning_field_prefixes"] = {}
        output_fields = list(self.signature.output_fields.keys())
        for field_name in output_fields:
            reasoning_field_name = f"{field_name}_reasoning"
            if reasoning_field_name in self.extended_signature.fields:
                state["reasoning_field_prefixes"][reasoning_field_name] = (
                    self.extended_signature.fields[reasoning_field_name].json_schema_extra['prefix']
                )
        
        return state

    def load_state(self, state):
        super().load_state(state)
        
        # Reconstruct the signature
        if "extended_signature_instructions" in state:
            instructions = state["extended_signature_instructions"]
            self.extended_signature = self.extended_signature.with_instructions(instructions)
        
        if "reasoning_field_prefixes" in state:
            for field_name, prefix in state["reasoning_field_prefixes"].items():
                self.extended_signature = self.extended_signature.with_updated_fields(
                    field_name, prefix=prefix
                )