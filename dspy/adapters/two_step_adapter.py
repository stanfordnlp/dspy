from typing import Any, Type

from dspy.signatures.signature import Signature
from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.utils import enumerate_fields
from dspy.signatures.field import InputField
from dspy.signatures.signature import make_signature
from dspy.clients import LM

class TwoStepAdapter(Adapter):
    """
    A two-stage adapter that:
        1. Uses a simpler, more natural prompt for the main LM
        2. Uses a smaller LM with chat adapter to extract structured data from the response of main LM
    This adapter uses a commong __call__ logic defined in base Adapter class.
    This class is particularly useful when interacting with reasoning models as the main LM since reasoning models
    are known to struggle with structured outputs.

    Example:
    ```
    import dspy
    class CoT(dspy.Module):
        def __init__(self):
            super().__init__()
            self.prog = dspy.ChainOfThought("question -> answer")

        def forward(self, question):
            return self.prog(question=question)
    lm = dspy.LM(model='openai/o3-mini', max_tokens=10000, temperature = 1.0)
    adapter = dspy.TwoStepAdapter(dspy.LM('openai/gpt-4o-mini'))
    dspy.configure(lm=lm, adapter=adapter)
    program = CoT()
    result = program("What is the capital of France?")
    print(result)
    ```
    """
    
    def __init__(self, extraction_model: LM):
        assert isinstance(extraction_model, LM), "extraction_model must be an instance of LM"
        self.extraction_model = extraction_model

    def format(self, signature: Type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Format a natural prompt for the first stage with the main LM.

        Args:
            signature: The signature of the original task
            demos: A list of demo examples
            inputs: The current input
        
        Returns:
            A list of messages to be passed to the main LM.
        """
        messages = []

        # Create a task description for the main LM
        task_description = self._create_task_description(signature)
        messages.append({"role": "system", "content": task_description})

        # Format demos in a natural way
        for demo in demos:
            messages.extend(self._format_demo(signature, demo))

        # Format the current input
        messages.append({"role": "user", "content": self._format_input(signature, inputs)})

        return messages

    def parse(self, signature: Signature, completion: str) -> dict[str, Any]:
        """
        Use a smaller LM (extraction_model) with chat adapter to extract structured data 
        from the raw completion text of the main LM.

        Args:
            signature: The signature of the original task
            completion: The completion from the main LM

        Returns:
            A dictionary containing the extracted structured data.
        """
        # The signature is supposed to be "text -> {original output fields}"
        extractor_signature = self._create_signature_with_text_input_and_outputs(signature)
        
        try:
            # Call the smaller LM to extract structured data from the raw completion text with ChatAdapter
            parsed_result = ChatAdapter()(lm=self.extraction_model, lm_kwargs={}, signature=extractor_signature, demos=[], inputs={ "text": completion })
            return parsed_result[0]

        except Exception as e:
            raise ValueError(f"Failed to parse response from the original completion: {completion}") from e
    
    def _create_task_description(self, signature: Signature) -> str:
        """Create a natural description of the task based on the signature"""
        parts = []
        
        parts.append("You are a helpful assistant that can solve tasks based on user input.")
        parts.append("Your input fields are:\n" + enumerate_fields(signature.input_fields))
        parts.append("Your final output fields are:\n" + enumerate_fields(signature.output_fields))
        parts.append("You should provide reasoning for your answers in details so that your answer can be understood by another agent")
        
        if signature.instructions:
            parts.append(f"Specific instructions: {signature.instructions}")
            
        return "\n".join(parts)
    
    def _format_input(self, signature: Signature, values: dict[str, Any]) -> str:
        parts = []
        
        for name in signature.input_fields.keys():
            if name in values:
                parts.append(f"{name}: {values[name]}")
                
        return "\n".join(parts)
    
    def _format_demo(self, signature: Signature, values: dict[str, Any]) -> list[dict[str, str]]:
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

    def _create_signature_with_text_input_and_outputs(
        self,
        original_signature: Type[Signature],        
    ) -> Type[Signature]:
        """Create a new signature containing a new 'text' input field and all output fields.
        
        Args:
            original_signature: The original signature to extract output fields from
        
        Returns:
            A new Signature type with a text input field and all output fields
        """
        # Create new fields dict with 'text' input field and all output fields
        new_fields = {
            'text': (str, InputField()),
            **{
                name: (field.annotation, field)
                for name, field in original_signature.output_fields.items()
            }
        }
        
        outputs_str = ", ".join([f"`{field}`" for field in original_signature.output_fields])
        instructions = f"The input is a text that should contain all the necessary information to produce the fields {outputs_str}. \
            Your job is to extract the fields from the text verbatim. Do not repeat the name of the field in your response."
            
        return make_signature(new_fields, instructions)
