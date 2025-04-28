from typing import Any, Dict, List, Optional, Tuple, Type
import json
import logging
import anyio
from dspy.primitives.tool import Tool

logger = logging.getLogger(__name__)

def map_json_schema_to_tool_args(schema: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Type], Dict[str, str]]:
    """Maps a JSON schema to tool arguments compatible with DSPy Tool.
    
    Args:
        schema: A JSON schema describing the tool's input parameters
        
    Returns:
        A tuple of (args, arg_types, arg_desc) dictionaries for DSPy Tool initialization
    """
    args, arg_types, arg_desc = {}, {}, {}
    if not schema or "properties" not in schema:
        return args, arg_types, arg_desc
        
    type_mapping = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}
    required = schema.get("required", [])
    
    for name, prop in schema["properties"].items():
        args[name] = prop
        arg_types[name] = type_mapping.get(prop.get("type", "string"), Any)
        arg_desc[name] = prop.get("description", "No description provided.")
        if name in required:
            arg_desc[name] += " (Required)"

    return args, arg_types, arg_desc

class MCPTool(Tool):
    """Wrapper for an MCP tool, compatible with DSPy agents.
    
    This class wraps a Model Context Protocol tool and makes it compatible with
    DSPy's ReAct and other agent frameworks. It handles the translation between
    DSPy's tool interface and MCP's JSON-RPC interface.
    """

    def __init__(self, tool_info: Any, session: Any):
        """Create a DSPy Tool from an MCP tool description.
        
        Args:
            tool_info: Tool information from MCP (object, dict, or JSON string)
            session: MCP client session for making tool calls
        """
        self.session = session
        self._raw_tool_info = tool_info

        name, desc, input_schema = self._extract_tool_info(tool_info)
        self.name = name
        args, arg_types, arg_desc = map_json_schema_to_tool_args(input_schema)

        super().__init__(
            func=self.call_tool_async,
            name=name,
            desc=desc,
            args=args,
            arg_types=arg_types,
            arg_desc=arg_desc
        )

    def _extract_tool_info(self, tool_info: Any) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """Extract name, description and input schema from tool info.
        
        Args:
            tool_info: Tool information in various formats (object, dict, JSON string)
            
        Returns:
            A tuple of (name, description, input_schema)
        """
        # Try object attributes
        if hasattr(tool_info, 'name') and hasattr(tool_info, 'description'):
            return (
                tool_info.name,
                tool_info.description,
                getattr(tool_info, 'inputSchema', None)
            )
    
        return str(tool_info), "No description available.", None

    async def call_tool_async(self, **kwargs: Any) -> Any:
        """Execute the MCP tool asynchronously.
        
        Args:
            **kwargs: Arguments to pass to the MCP tool
            
        Returns:
            Processed result from the tool execution
            
        Raises:
            RuntimeError: If there's an error executing the tool
        """
        try:
            logger.debug(f"Executing MCP tool {self.name} with args: {kwargs}")
            result = await self.session.call_tool(self.name, kwargs)
            logger.debug(f"MCP tool {self.name} returned: {result}")
            return self._process_result(result)
        except anyio.ClosedResourceError as e:
            # Special handling for closed resources (common during cleanup)
            logger.error(f"MCP resource closed during {self.name} execution: {str(e)}")
            raise RuntimeError(f"MCP connection closed while executing {self.name}")
        except Exception as e:
            logger.error(f"Error executing MCP tool {self.name}: {str(e)}")
            raise RuntimeError(f"Error executing tool {self.name}: {str(e)}")

    def _process_result(self, result: Any) -> Any:
        """Process the result from tool execution into a format suitable for agents.
        
        Args:
            result: Raw result from the MCP tool
            
        Returns:
            Processed result (typically as a string or structured data)
        """
        if result is None:
            return "Tool executed successfully but returned no content."
        
        # Handle content attribute
        if hasattr(result, 'content') and result.content:
            content = result.content
            if isinstance(content, list):
                try:
                    return "\n".join(str(getattr(item, 'text', item)) for item in content if item)
                except Exception:
                    pass
            return str(content)
            
        # Handle text attribute
        if hasattr(result, 'text') and result.text is not None:
            return result.text
            
        # Handle dictionary
        if isinstance(result, dict):
            for key in ("message", "output", "result", "text"):
                if key in result:
                    return str(result[key])
            return json.dumps(result, indent=2)
            
        return str(result)

class MCPTools:
    """Collection of tools from an MCP server, usable with DSPy agents.
    
    This class provides a container for multiple MCP tools and makes them
    available in a format compatible with DSPy's agent frameworks like ReAct.
    """
    
    def __init__(self, session: Any, tools_list: List[Any]):
        """Initialize the MCPTools collection.
        
        Args:
            session: MCP client session for making tool calls
            tools_list: List of tool descriptions from MCP
        """
        self.session = session
        self.tools = {MCPTool(tool, session).name: MCPTool(tool, session) for tool in tools_list}
        logger.info(f"Initialized MCPTools with {len(self.tools)} tools: {', '.join(self.tools.keys())}")
    
    def __getitem__(self, tool_name: str) -> MCPTool:
        """Get a tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            The requested MCPTool instance
            
        Raises:
            KeyError: If the tool is not found
        """
        if tool_name not in self.tools:
            raise KeyError(f"Tool '{tool_name}' not found in available MCP tools")
        return self.tools[tool_name]
    
    def get_tools(self) -> List[Tool]:
        """Get all tools as a list.
        
        Returns:
            List of all available MCPTool instances
        """
        return list(self.tools.values())
    
    def get_tool_names(self) -> List[str]:
        """Get names of all available tools.
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys())
    
    def __str__(self) -> str:
        """String representation showing available tools."""
        return f"MCPTools with {len(self.tools)} tools: {', '.join(self.tools.keys())}"
    
    def __repr__(self) -> str:
        """Detailed representation of the tools collection."""
        return f"MCPTools({len(self.tools)} tools: {list(self.tools.keys())})"

