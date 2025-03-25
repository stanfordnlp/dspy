import json
from typing import Any, Dict, Type, Optional

from dspy.signatures.signature import Signature
from dspy.adapters.base import Adapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.utils import get_dspy_field_type
from dspy.signatures.signature import make_signature
from dspy.clients import LM
# from dspy.predict.predict import Predict


class SmallLMAdapter(Adapter):
    """
    A two-stage adapter that:
    1. Uses a simpler, more natural prompt for the main LM
    2. Uses a smaller LM with JSON adapter to extract structured data from the response
    """
    
    def __init__(self, extraction_model):
        self.extraction_model = extraction_model
        assert isinstance(self.extraction_model, LM)
        self.json_adapter = JSONAdapter()

    def format(self, signature: Signature, demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]:
        """Format a more natural prompt for the first stage"""
        messages = []

        # Create a natural description of the task
        task_description = self._create_task_description(signature)
        messages.append({"role": "system", "content": task_description})

        # Format demos in a natural way
        # print("len(demos)", len(demos))
        for demo in demos:
            # print("demo", demo)
            messages.extend(self._format_demo(signature, demo))

        # Format the current input
        messages.append({"role": "user", "content": self._format_input(signature, inputs)})

        return messages

    # This could probably be a method on the Signature class
    def _create_signature_with_text_input_and_outputs(
        self,
        original_signature: Type[Signature], 
        instructions: Optional[str] = None
    ) -> Type[Signature]:
        """Create a new signature containing a new 'text' input field plus all output fields.
        
        Args:
            original_signature: The original signature to extract output fields from
            instructions: Optional custom instructions for the new signature. If None, 
                        will generate default instructions.
        
        Returns:
            A new Signature type with a text input field and all output fields
        """
        # Create new fields dict starting with our new text input
        new_fields = {
            'text': (str, InputField())
        }
        
        # Add all output fields
        new_fields.update({
            name: (field.annotation, field)
            for name, field in original_signature.output_fields.items()
        })
        
        if instructions is None:
            outputs_str = ", ".join([f"`{field}`" for field in original_signature.output_fields])
            instructions = f"The input is a text that should contain all the necessary information to produce the fields {outputs_str}. \
                Your job is to extract the fields from the text verbatim. Do not repeat the name of the field in your response."
            
        return make_signature(new_fields, instructions)

    def parse(self, signature: Signature, completion: str) -> Dict[str, Any]:
        """
        Two-stage parsing:
        1. Get unstructured completion from main LM
        2. Use smaller LM with JSON adapter to extract structured data
        """
        # The signature is supposed to be "input -> {original output fields}"
        # Json is implicit with structured outputs and jsonadapter
        
        extractor_signature = self._create_signature_with_text_input_and_outputs(signature)

        import dspy
        extractor = dspy.Predict(extractor_signature)
        extractor.demos = [
            dspy.Example(
                text="""
<think>
Okay, let's see. I need to solve the equation 2x + 3 = 7 for x. Hmm, where do I start? Oh right, the goal is to get
x by itself on one side of the equation.

First, I should get rid of that 3 that's being added to 2x. To do that, I can subtract 3 from both sides of the 
equation. That way, the equation stays balanced. So, subtracting 3 from both sides gives me 2x = 7 - 3. Let me 
calculate that: 7 minus 3 is 4. So now the equation is 2x = 4.

Now, I need to solve for x. Since 2 is multiplied by x, I should divide both sides by 2 to isolate x. Dividing both
sides by 2 gives x = 4 / 2. Calculating that, 4 divided by 2 is 2. So x equals 2. Let me check if that makes sense.
Plugging x = 2 back into the original equation: 2*(2) + 3 = 4 + 3 = 7, which matches the right side. Yep, that 
works. So the solution is x = 2.
</think>

reasoning: To solve for x, first subtract 3 from both sides to isolate the term with x, resulting in 2x = 4. Then 
divide both sides by 2 to find x = 2.  
answer: 2""",
                reasoning="To solve for x, first subtract 3 from both sides to isolate the term with x, resulting in 2x = 4. Then divide both sides by 2 to find x = 2.",
                answer="2"
            ).with_inputs("text")
        ]
        
        try:
            # Call the smaller LM to extract JSON
            # import rich
            # rich.print(completion)

            with dspy.settings.context(adapter=self.json_adapter, lm=self.extraction_model):
                extracted_data = extractor(text=completion)
            # rich.print(extracted_data)
            # Validate the extracted data matches our signature
            # if not all(field in extracted_data for field in signature.output_fields):
            #     missing = set(signature.output_fields) - set(extracted_data)
            #     raise ValueError(f"Missing required fields in extracted data: {missing}")
                
            return extracted_data
            
        except Exception as e:
            raise ValueError(f"Failed to parse response: {str(e)}\nOriginal completion: {completion}")
    
    def _create_task_description(self, signature: Signature) -> str:
        """Create a natural description of the task based on the signature"""
        parts = []
        
        # Get field descriptions
        input_descs = [
            f"{name}: {field.json_schema_extra.get('desc', name)}"
            for name, field in signature.input_fields.items()
        ]
        output_descs = [
            f"{name}: {field.json_schema_extra.get('desc', name)}"
            for name, field in signature.output_fields.items()
        ]
        
        parts.append("You are a helpful assistant that can solve tasks based on user input.")
        parts.append(f"For each input, which includes: {', '.join(input_descs)}")
        parts.append(f"You should provide: {', '.join(output_descs)}")
        
        if signature.instructions:
            parts.append(f"\nSpecific instructions: {signature.instructions}")
            
        return "\n".join(parts)
    
    def _format_input(self, signature: Signature, values: Dict[str, Any]) -> str:
        """Format input in a natural way"""
        parts = []
        
        for name, field in signature.input_fields.items():
            if name in values:
                parts.append(f"{name}: {values[name]}")
                
        return "\n".join(parts)
    
    def _format_demo(self, signature: Signature, values: Dict[str, Any]) -> list[dict[str, str]]:
        """Format a demo example in a natural way"""
        messages = []
        
        # Format input
        if any(k in values for k in signature.input_fields):
            messages.append({
                "role": "user",
                "content": self._format_input(signature, values)
            })
        
        # Format output if present
        if any(k in values for k in signature.output_fields):
            output_parts = []
            for name, field in signature.output_fields.items():
                if name in values:
                    desc = field.json_schema_extra.get('desc', name)
                    output_parts.append(f"{desc}: {values[name]}")
            
            if output_parts:
                messages.append({
                    "role": "assistant",
                    "content": "\n".join(output_parts)
                })
                
        return messages
    
    def _create_extraction_prompt(self, signature: Signature, text: str) -> str:
        """Create a prompt for the extraction phase"""
        parts = []
        
        parts.append("Extract the following information from the text into JSON format:")
        for name, field in signature.output_fields.items():
            desc = field.json_schema_extra.get('desc', name)
            field_type = field.annotation.__name__ if hasattr(field.annotation, '__name__') else str(field.annotation)
            parts.append(f"- {name} ({field_type}): {desc}")
            
        parts.append("\nText to extract from:")
        parts.append(text)
        
        parts.append("\nProvide the output in valid JSON format with these exact field names.")
        
        return "\n".join(parts) 