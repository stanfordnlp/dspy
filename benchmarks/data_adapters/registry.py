"""
Registry for dataset adapters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Type

if TYPE_CHECKING:
    from .base import DatasetAdapter


class DatasetRegistry:
    """Registry for dataset adapters."""
    
    _adapters: Dict[str, Type[DatasetAdapter]] = {}
    
    @classmethod
    def register(cls, name: str, adapter_class: Type[DatasetAdapter]) -> None:
        """Register a dataset adapter.
        
        Args:
            name: Dataset name.
            adapter_class: Dataset adapter class.
        """
        cls._adapters[name] = adapter_class
    
    @classmethod
    def create_adapter(cls, name: str, config: Dict[str, Any]) -> DatasetAdapter:
        """Create a dataset adapter instance.
        
        Args:
            name: Dataset name.
            config: Configuration for the adapter.
            
        Returns:
            Dataset adapter instance.
            
        Raises:
            ValueError: If dataset name is not registered.
        """
        if name not in cls._adapters:
            available = ", ".join(cls._adapters.keys())
            raise ValueError(f"Unknown dataset: {name}. Available: {available}")
        
        adapter_class = cls._adapters[name]
        return adapter_class(config)
    
    @classmethod
    def list_datasets(cls) -> list[str]:
        """List available dataset names."""
        return list(cls._adapters.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a dataset is registered."""
        return name in cls._adapters


# Register built-in adapters
def _register_builtin_adapters() -> None:
    """Register built-in dataset adapters."""
    from data_adapters.hotpotqa import HotPotQAAdapter
    from data_adapters.aime import AIMEAdapter

    DatasetRegistry.register("hotpotqa", HotPotQAAdapter)
    DatasetRegistry.register("aime", AIMEAdapter)


# Auto-register on import
_register_builtin_adapters()