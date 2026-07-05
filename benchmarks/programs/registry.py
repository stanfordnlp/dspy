"""
Registry for program definitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Type

if TYPE_CHECKING:
    from .base import BaseProgram


class ProgramRegistry:
    """Registry for program definitions."""
    
    _programs: Dict[str, Type[BaseProgram]] = {}
    
    @classmethod
    def register(cls, name: str, program_class: Type[BaseProgram]) -> None:
        """Register a program class.
        
        Args:
            name: Program name.
            program_class: Program class.
        """
        cls._programs[name] = program_class
    
    @classmethod
    def create_program(cls, name: str, config: Dict[str, Any] = None) -> BaseProgram:
        """Create a program instance.
        
        Args:
            name: Program name.
            config: Configuration for the program (currently unused).
            
        Returns:
            Program instance.
            
        Raises:
            ValueError: If program name is not registered.
        """
        if name not in cls._programs:
            available = ", ".join(cls._programs.keys())
            raise ValueError(f"Unknown program: {name}. Available: {available}")
        
        program_class = cls._programs[name]
        return program_class()
    
    @classmethod
    def list_programs(cls) -> list[str]:
        """List available program names."""
        return list(cls._programs.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a program is registered."""
        return name in cls._programs


# Register built-in programs
def _register_builtin_programs() -> None:
    """Register built-in program classes."""
    from programs.qa import (
        NaiveQA,
        ReasoningFirstQA,
        ContextQA,
        ReasoningContextQA,
        MLflowBasePromptContextQA,
        MLflowBasePromptContextQAv2,
        SimpleContextQA,
        MathAnswerOnly,
        MathCoT,
        MathNaive,
    )

    ProgramRegistry.register("naive", NaiveQA)
    ProgramRegistry.register("reasoning", ReasoningFirstQA)
    ProgramRegistry.register("context", ContextQA)
    ProgramRegistry.register("reasoning_context", ReasoningContextQA)
    ProgramRegistry.register("mlflow_base_prompt", MLflowBasePromptContextQA)
    ProgramRegistry.register("mlflow_v2", MLflowBasePromptContextQAv2)
    ProgramRegistry.register("simple_context", SimpleContextQA)
    ProgramRegistry.register("math_answer_only", MathAnswerOnly)
    ProgramRegistry.register("math_cot", MathCoT)
    ProgramRegistry.register("math_naive", MathNaive)

# Auto-register on import
_register_builtin_programs()
