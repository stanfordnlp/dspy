from typing import Any, Dict, List, Optional
from dspy.signatures.signature import Signature, SignatureMeta
from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter, format_fields, FieldInfoWithName
import logging
from pydantic import TypeAdapter

class ChainedTranslationAdapter(Adapter):
    """
    A sophisticated adapter that implements a three-step translation process:
    1. Translate source language to English
    2. Process the query/task in English
    3. Translate results back to target language
    """

    def __init__(
        self,
        lm,  # The language model being used
        source_lang: str,
        target_lang: str,
        base_adapter: ChatAdapter,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ChainedTranslationAdapter with required components.

        Args:
            lm: Language model instance to use for translations
            source_lang: Source language of the input
            target_lang: Target language for final output
            base_adapter: Underlying ChatAdapter for message handling
            logger: Optional custom logger
        """
        super().__init__()
        
        if not isinstance(source_lang, str) or not source_lang.strip():
            raise ValueError("source_lang must be a non-empty string")
        if not isinstance(target_lang, str) or not target_lang.strip():
            raise ValueError("target_lang must be a non-empty string")
        if not isinstance(base_adapter, ChatAdapter):
            raise TypeError("base_adapter must be an instance of ChatAdapter")

        self.lm = lm
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.base_adapter = base_adapter
        self.logger = logger or logging.getLogger(__name__)

    def _translate_to_english(self, text: str) -> str:
        """Translate input text from source language to English."""
        messages = [
            {
                "role": "system",
                "content": f"""You are a professional translator from {self.source_lang} to English.
                Translate the following text to English, preserving all meaning and context.
                Maintain any special formatting, numbers, or technical terms.
                Provide ONLY the translation, no explanations or notes."""
            },
            {
                "role": "user",
                "content": text
            }
        ]
        
        response = self.lm(messages=messages)
        return response[0] if isinstance(response, list) else response

    def _translate_to_target(self, text: str) -> str:
        """Translate processed English text to target language."""
        messages = [
            {
                "role": "system",
                "content": f"""You are a professional translator from English to {self.target_lang}.
                Translate the following text to {self.target_lang}, preserving all meaning and context.
                Maintain any special formatting, numbers, or technical terms.
                Provide ONLY the translation, no explanations or notes."""
            },
            {
                "role": "user",
                "content": text
            }
        ]
        
        response = self.lm(messages=messages)
        return response[0] if isinstance(response, list) else response

    def _translate_dict_to_english(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Translate all string values in a dictionary from source language to English."""
        translated = {}
        for key, value in data.items():
            if isinstance(value, str) and value.strip():
                translated[key] = self._translate_to_english(value)
            else:
                translated[key] = value
        return translated

    def _translate_dict_to_target(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Translate all string values in a dictionary from English to target language."""
        translated = {}
        for key, value in data.items():
            if isinstance(value, str) and value.strip():
                translated[key] = self._translate_to_target(value)
            else:
                translated[key] = value
        return translated

    def format(self, signature: Signature, demos: List[Dict[str, Any]], 
               inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format input for processing, translating from source language to English first."""
        try:
            # Translate demos to English
            english_demos = []
            for demo in demos:
                english_demo = self._translate_dict_to_english(demo)
                english_demos.append(english_demo)

            # Translate inputs to English
            english_inputs = self._translate_dict_to_english(inputs)

            # Use base adapter to format the English content
            messages = self.base_adapter.format(signature, english_demos, english_inputs)

            # Add translation context to system message
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = messages[0]["content"] + f"\nNote: Final output should be in {self.target_lang}."

            return messages

        except Exception as e:
            self.logger.error(f"Error in format step: {str(e)}")
            raise

    def parse(self, signature: Signature, completion: str, 
              _parse_values: bool = True) -> Dict[str, Any]:
        """Parse completion and translate results to target language."""
        try:
            # First parse the English completion using base adapter
            parsed_result = self.base_adapter.parse(
                signature, completion, _parse_values=False  # Parse without value conversion first
            )

            # Translate the parsed results to target language
            translated_result = self._translate_dict_to_target(parsed_result)

            # Now parse values if needed
            if _parse_values:
                final_result = {}
                for field_name, field_value in translated_result.items():
                    if field_name in signature.output_fields:
                        try:
                            field_type = signature.output_fields[field_name].annotation
                            if field_type is str:
                                final_result[field_name] = field_value
                            else:
                                # For non-string types, we need to parse the value
                                final_result[field_name] = TypeAdapter(field_type).validate_python(field_value)
                        except Exception as e:
                            self.logger.error(f"Error parsing field {field_name}: {str(e)}")
                            final_result[field_name] = field_value
                return final_result
            
            return translated_result

        except Exception as e:
            self.logger.error(f"Error in parse step: {str(e)}")
            return {field: "" for field in signature.output_fields}

    def format_finetune_data(self, signature: Signature, demos: List[Dict[str, Any]], 
                            inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Format data for fine-tuning, handling translations appropriately."""
        # Translate all content to English for fine-tuning
        english_demos = [self._translate_dict_to_english(demo) for demo in demos]
        english_inputs = self._translate_dict_to_english(inputs)
        english_outputs = self._translate_dict_to_english(outputs)

        return self.base_adapter.format_finetune_data(
            signature, english_demos, english_inputs, english_outputs
        )