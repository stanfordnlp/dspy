"""Tool types for DSPy minimal implementation."""

from typing import Any, Dict, List, Optional


class Tool:
    """Represents a tool that can be called by the language model."""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.description = description
        self.parameters = parameters or {}
    
    def format_as_litellm_function_call(self) -> Dict[str, Any]:
        """Format the tool as a LiteLLM function call."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    def __call__(self, **kwargs) -> Any:
        """Execute the tool with the given arguments."""
        # This is a placeholder - actual tool execution would be implemented by subclasses
        raise NotImplementedError(f"Tool {self.name} execution not implemented")


class ToolCalls:
    """Represents a collection of tool calls."""
    
    def __init__(self, calls: List[Dict[str, Any]] = None):
        self.calls = calls or []
    
    @classmethod
    def from_dict_list(cls, calls: List[Dict[str, Any]]) -> "ToolCalls":
        """Create ToolCalls from a list of dictionaries."""
        return cls(calls)
    
    def add_call(self, name: str, args: Dict[str, Any]):
        """Add a tool call."""
        self.calls.append({"name": name, "args": args})
    
    def get_calls(self) -> List[Dict[str, Any]]:
        """Get all tool calls."""
        return self.calls
    
    def __len__(self) -> int:
        return len(self.calls)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.calls[index]
    
    def __iter__(self):
        return iter(self.calls)