"""
Registry for optimizer adapters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Type

if TYPE_CHECKING:
    from .base import OptimizerAdapter


class OptimizerRegistry:
    """Registry for optimizer adapters."""
    
    _adapters: Dict[str, Type[OptimizerAdapter]] = {}
    
    @classmethod
    def register(cls, name: str, adapter_class: Type[OptimizerAdapter]) -> None:
        """Register an optimizer adapter.
        
        Args:
            name: Optimizer name.
            adapter_class: Optimizer adapter class.
        """
        cls._adapters[name] = adapter_class
    
    @classmethod
    def create_adapter(cls, name: str, config: Dict[str, Any]) -> OptimizerAdapter:
        """Create an optimizer adapter instance.
        
        Args:
            name: Optimizer name.
            config: Configuration for the adapter.
            
        Returns:
            Optimizer adapter instance.
            
        Raises:
            ValueError: If optimizer name is not registered.
        """
        if name not in cls._adapters:
            available = ", ".join(cls._adapters.keys())
            raise ValueError(f"Unknown optimizer: {name}. Available: {available}")
        
        adapter_class = cls._adapters[name]
        return adapter_class(config)
    
    @classmethod
    def list_optimizers(cls) -> list[str]:
        """List available optimizer names."""
        return list(cls._adapters.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an optimizer is registered."""
        return name in cls._adapters


# Register built-in adapters
def _register_builtin_adapters() -> None:
    """Register built-in optimizer adapters."""
    from optimizers.gepa import GepaAdapter
    from optimizers.mipro import MiproAdapter
    from optimizers.bootstrap import BootstrapAdapter
    from optimizers.copro import CoproAdapter
    from optimizers.baseline import BaselineAdapter
    from optimizers.sbo import SBOAdapter

    OptimizerRegistry.register("gepa", GepaAdapter)
    OptimizerRegistry.register("mipro", MiproAdapter)
    OptimizerRegistry.register("bootstrap", BootstrapAdapter)
    OptimizerRegistry.register("copro", CoproAdapter)
    OptimizerRegistry.register("baseline", BaselineAdapter)
    OptimizerRegistry.register("sbo", SBOAdapter)


# Auto-register on import
_register_builtin_adapters()