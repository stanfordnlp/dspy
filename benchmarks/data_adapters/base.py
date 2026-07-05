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

    def get_input_fields(self, default: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        """Return input fields from config, falling back to an adapter default."""
        configured = self.config.get("input_fields")
        if configured is None:
            configured = self.config.get("input_keys")
        if configured is None:
            configured = default
        if isinstance(configured, str):
            configured = [configured]
        input_fields = tuple(configured)
        if not input_fields:
            raise ValueError(f"{self.name} requires at least one input field")
        return input_fields

    def apply_input_fields(self, examples: list[Example], input_fields: tuple[str, ...]) -> list[Example]:
        """Attach input fields to examples after validating field names exist."""
        missing_by_example = []
        for idx, ex in enumerate(examples[:10]):
            missing = [field for field in input_fields if field not in ex]
            if missing:
                missing_by_example.append((idx, missing))

        if missing_by_example:
            details = ", ".join([f"example {idx}: {missing}" for idx, missing in missing_by_example])
            raise ValueError(
                f"Configured input_fields {list(input_fields)} are not present in {self.name} examples: {details}"
            )

        return [ex.with_inputs(*input_fields) for ex in examples]
        
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
