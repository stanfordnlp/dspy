from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseLM(ABC):
    """Base class for language models."""
    
    @abstractmethod
    def forward(self, prompt=None, messages=None, **kwargs):
        """Forward pass for the language model."""
        pass
        
    @abstractmethod
    async def aforward(self, prompt=None, messages=None, **kwargs):
        """Async forward pass for the language model."""
        pass
        
    def copy(self, **kwargs):
        """Create a copy of this LM with updated parameters."""
        # Simplified copy method
        return self.__class__(**{**self.__dict__, **kwargs}) 