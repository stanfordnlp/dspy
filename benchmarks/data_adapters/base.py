"""
Abstract base classes for dataset adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dspy import Example


class DatasetAdapter(ABC):
    """Abstract base class for dataset adapters."""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def load_dataset(self) -> tuple[list[Example], list[Example], list[Example]]:
        """Load and return train, validation, and test sets.
        
        Returns:
            Tuple of (train_set, val_set, test_set) with appropriate input fields.
        """
        pass
    
    @abstractmethod
    def get_metric(self) -> Any:
        """Return the evaluation metric for this dataset."""
        pass
    
    @abstractmethod
    def get_gepa_metric(self) -> Any:
        """Return the GEPA-compatible metric for this dataset."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the dataset name."""
        pass
    
    @property
    @abstractmethod
    def uses_context(self) -> bool:
        """Return whether this dataset uses context as input."""
        pass