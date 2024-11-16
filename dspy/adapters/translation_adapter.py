

from typing import Any, Dict, List, Optional
from dspy.signatures.signature import Signature, SignatureMeta
from dspy.adapters.base import *
from dspy.adapters.chat_adapter import *
import logging

class TranslationAdapter(Adapter):
    """
    A streamlined adapter for translation tasks that leverages ChatAdapter's core functionality
    while maintaining comprehensive translation instructions.
    """

    def __init__(
        self,
        target_lang: str,
        base_adapter: ChatAdapter,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize TranslationAdapter with required components.

        Args:
            target_lang: Target language for translations
            base_adapter: Underlying ChatAdapter for message handling
            logger: Optional custom logger
        """
        super().__init__()
        
        if not isinstance(target_lang, str) or not target_lang.strip():
            raise ValueError("target_lang must be a non-empty string")
            
        if not isinstance(base_adapter, ChatAdapter):
            raise TypeError("base_adapter must be an instance of ChatAdapter")

        self.target_lang = target_lang
        self.base_adapter = base_adapter
        self.logger = logger or logging.getLogger(__name__)

    def _get_translation_prompt(self, signature: SignatureMeta) -> str:
        """Generate comprehensive translation-specific system prompt."""
        parts = [
            f"""You are a professional translation assistant specializing in translations to {self.target_lang}.
            Your task is to provide accurate, culturally-appropriate translations that preserve the original meaning
            while adapting naturally to {self.target_lang} conventions.""",
            
            "\n### Core Translation Principles ###",
            "1. Accuracy and Fidelity:",
            "   - Preserve the complete meaning and intent of the original content",
            "   - Maintain all key information, context, and nuances",
            "   - Neither add nor remove information during translation",
            "   - Keep the original tone and style (formal, casual, technical, etc.)",
            
            "2. Natural Language Use:",
            f"   - Use natural, idiomatic expressions in {self.target_lang}",
            "   - Avoid literal translations that sound unnatural",
            "   - Maintain consistent style and terminology throughout",
            "   - Follow target language grammar and punctuation rules",
            
            "3. Cultural Adaptation:",
            "   - Adapt cultural references to resonate with the target audience",
            "   - Provide cultural context notes when necessary",
            "   - Use culturally appropriate expressions and metaphors",
            "   - Consider target region-specific language variations",
            
            "\n### Technical Guidelines ###",
            "1. Names and Identifiers:",
            "   - Keep person names in their original form",
            "   - Maintain brand names unless they have official translations",
            "   - Include original terms in parentheses where helpful",
            "   - Translate titles only when official translations exist",
            
            "2. Numbers and Formatting:",
            "   - Adapt number formats to target language conventions",
            "   - Convert units of measurement if culturally appropriate",
            "   - Maintain date/time formats per target language standards",
            "   - Preserve all formatting markers (bold, italic, lists, etc.)",
            
            "3. Technical Terms:",
            "   - Use standardized translations for technical terminology",
            "   - Maintain consistency in technical term usage",
            "   - Keep industry-specific jargon appropriate to the field",
            "   - Include original terms in parentheses for clarity if needed",
            
            "\n### Output Structure Requirements ###",
            "1. Field Formatting:",
            "   - Begin each field with: [[ ## fieldname ## ]]",
            "   - Keep field names in English - DO NOT translate them",
            "   - Include one blank line between fields",
            "   - Ensure proper field marker formatting",
            
            "2. Content Organization:",
            "   - Place translated content directly after each field marker",
            "   - Preserve original paragraph structure and formatting",
            "   - Maintain document hierarchy and organization",
            "   - Complete each field's content before starting the next",
            
            "\n### Required Output Fields ###",
            f"Include ALL of these fields in order:\n{', '.join(signature.output_fields)}",
            
            "\n### Quality Assurance ###",
            "1. Pre-submission Checklist:",
            "   - Verify all required fields are present and complete",
            "   - Check for consistent terminology throughout",
            "   - Confirm formatting and structure are preserved",
            "   - Ensure translations maintain original meaning and context",
            
            "2. Common Error Prevention:",
            "   - Double-check numbers and dates for accuracy",
            "   - Verify proper handling of technical terms",
            "   - Confirm cultural references are appropriately adapted",
            "   - Check for natural flow in target language",

            "### Language Enforcement ###",
            f"- ALL outputs MUST be in {self.target_lang}",
            "- Do not switch languages between responses",
            "- If you detect yourself using any other language, stop and restart in the target language",
        ]
        
        if signature.instructions:
            parts.extend([
                "\n### Task-Specific Instructions ###",
                signature.instructions,
                "\nFollow these additional instructions while maintaining all general translation requirements."
            ])
            
        return "\n".join(parts)

    def format(self, signature: Signature, demos: List[Dict[str, Any]], 
               inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format input for translation, leveraging ChatAdapter's formatting."""
        messages = []
        
        # Add translation-specific system message with original comprehensive instructions
        messages.append({
            "role": "system",
            "content": self._get_translation_prompt(signature)
        })
        
        # Use base adapter for demo and input formatting
        formatted_messages = self.base_adapter.format(signature, demos, inputs)
        
        # Skip base adapter's system message if present
        if formatted_messages and formatted_messages[0]["role"] == "system":
            formatted_messages = formatted_messages[1:]
            
        messages.extend(formatted_messages)
        return messages

    def parse(self, signature: Signature, completion: str, 
              _parse_values: bool = True) -> Dict[str, Any]:
        """Parse completion using ChatAdapter's robust parsing."""
        try:
            # Leverage ChatAdapter's parsing
            parsed_result = self.base_adapter.parse(
                signature, completion, _parse_values
            )
            return parsed_result
            
        except Exception as e:
            self.logger.error(f"Error parsing translation: {str(e)}")
            return {field: "" for field in signature.output_fields}

    def format_finetune_data(self, signature: Signature, demos: List[Dict[str, Any]], 
                            inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Format data for fine-tuning, delegating to ChatAdapter."""
        return self.base_adapter.format_finetune_data(
            signature, demos, inputs, outputs
        )