"""Exceptions for DSPy minimal implementation."""


class DSPyError(Exception):
    """Base exception for DSPy errors."""
    pass


class AdapterParseError(DSPyError):
    """Raised when an adapter fails to parse input or output."""
    
    def __init__(self, message=None, adapter_name=None, signature=None, lm_response=None, parsed_result=None):
        self.message = message
        self.adapter_name = adapter_name
        self.signature = signature
        self.lm_response = lm_response
        self.parsed_result = parsed_result
        
        # Build the error message
        if message:
            super().__init__(message)
        else:
            error_msg = f"Adapter {adapter_name or 'Unknown'} failed to parse"
            if signature:
                error_msg += f" signature {signature}"
            if lm_response:
                error_msg += f" from response: {lm_response[:100]}..."
            super().__init__(error_msg)


class SignatureError(DSPyError):
    """Raised when there's an issue with signatures."""
    pass


class ConfigurationError(DSPyError):
    """Raised when there's a configuration issue."""
    pass


class LMError(DSPyError):
    """Raised when there's an issue with the language model."""
    pass 