"""
Abstract base classes for optimizer adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dspy import Example, Module


class OptimizerAdapter(ABC):
    """Abstract base class for optimizer adapters."""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def optimize(
        self,
        program: Module,
        train_set: list[Example], 
        val_set: list[Example],
        metric: Any,
        **kwargs
    ) -> Module:
        """Optimize the given program.
        
        Args:
            program: DSPy program to optimize.
            train_set: Training examples.
            val_set: Validation examples.
            metric: Evaluation metric.
            **kwargs: Additional optimizer-specific arguments.
            
        Returns:
            Optimized program.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the optimizer name."""
        pass
    
    def supports_gepa_metric(self) -> bool:
        """Return whether this optimizer supports GEPA-style metrics."""
        return False