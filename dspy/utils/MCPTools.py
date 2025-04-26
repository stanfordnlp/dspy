from typing import Any, Dict, List, Optional, Tuple, Type, Union
import json
from dspy.primitives.tool import Tool

def map_json_schema_to_tool_args(
    schema: Optional[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Type], Dict[str, str]]:
    """Maps a JSON schema to tool arguments compatible with DSPy Tool."""
    args, arg_types, arg_desc = {}, {}, {}

    if not schema or "properties" not in schema:
        return args, arg_types, arg_desc
        
    for name, prop in schema["properties"].items():
        args[name] = prop

        # Map JSON schema types to Python types
        type_mapping = {
            "string": str, "integer": int, "number": float, 
            "boolean": bool, "array": list, "object": dict
        }
        prop_type = prop.get("type", "string")
        arg_types[name] = type_mapping.get(prop_type, Any)

        # Description with required indicator
        arg_desc[name] = prop.get("description", "No description provided.")
        if name in schema.get("required", []):
            arg_desc[name] += " (Required)"

    return args, arg_types, arg_desc

class MCPTool(Tool):
    """Wrapper for an MCP tool, compatible with DSPy agents."""

    def __init__(self, tool_info: Any, session: Any):
        """Create a DSPy Tool from an MCP tool description.
        
        Args:
            tool_info: The tool information from MCP server
            session: The MCP client session
        """
        self.session = session
        self._raw_tool_info = tool_info

        name, desc, input_schema = self._extract_tool_info(tool_info)
        self.name = name  # Store name as instance attribute for use in call_tool_async
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
        """Extract name, description and input schema from tool info."""
        # Handle object with attributes first (most common case for MCP tools)
        try:
            name = getattr(tool_info, 'name', None)
            desc = getattr(tool_info, 'description', None)
            
            # Handle inputSchema or schema attribute
            input_schema = None
            if hasattr(tool_info, 'inputSchema'):
                input_schema = tool_info.inputSchema
            
            # If all attributes were found, return them
            if name and desc is not None:
                return name, desc, input_schema
        except (AttributeError, TypeError):
            pass
            
        # Handle dictionary format
        if isinstance(tool_info, dict):
            name = tool_info.get('name')
            desc = tool_info.get('description')
            input_schema = tool_info.get('inputSchema')
            
            if name and desc is not None:
                return name, desc, input_schema
        
        # Handle serialized JSON string
        if isinstance(tool_info, str):
            try:
                parsed = json.loads(tool_info)
                if isinstance(parsed, dict):
                    name = parsed.get('name')
                    desc = parsed.get('description')
                    input_schema = parsed.get('inputSchema')
                    
                    if name and desc is not None:
                        return name, desc, input_schema
            except json.JSONDecodeError:
                pass
                
        # Last resort fallback - use string representation as name
        return str(tool_info), "No description available.", None

    async def call_tool_async(self, **kwargs: Any) -> Any:
        """Execute the MCP tool."""
        try:
            # Pass the kwargs directly without nesting them
            result = await self.session.call_tool(self.name, kwargs)
            print(result)
            return self._process_result(result)
        except Exception as e:
            raise RuntimeError(f"Error executing tool {self.name}: {str(e)}")

    def _process_result(self, result: Any) -> Any:
        """Process the result from tool execution."""
        if result is None:
            return "Tool executed successfully but returned no content."
        
        if hasattr(result, 'content') and result.content:
            content = result.content
            if isinstance(content, list):
                try:
                    return "\n".join(str(getattr(item, 'text', item)) for item in content if item)
                except Exception:
                    pass
            return str(content)
            
        if hasattr(result, 'text') and result.text is not None:
            return result.text
            
        if isinstance(result, dict):
            for key in ("message", "output", "result", "text"):
                if key in result:
                    return str(result[key])
            return json.dumps(result, indent=2)
            
        return str(result)

class MCPTools:
    """Collection of tools from an MCP server, usable with DSPy agents."""
    
    def __init__(self, tools_list: List[Any], session: Any):
        """Initialize the MCPTools collection.
        
        Args:
            tools_list: List of tools from MCP server
            session: MCP client session
        """
        self.session = session
        self.tools = {}
        
        # Create MCPTool instances for each tool in the list
        for tool in tools_list:
            mcp_tool = MCPTool(tool, session)
            self.tools[mcp_tool.name] = mcp_tool
    
    def __getitem__(self, tool_name: str) -> MCPTool:
        """Get a tool by name."""
        if tool_name not in self.tools:
            raise KeyError(f"Tool '{tool_name}' not found in available MCP tools")
        return self.tools[tool_name]
    
    def get_tools(self) -> List[Tool]:
        """Get all tools as a list."""
        return list(self.tools.values())
    
    def get_tool_names(self) -> List[str]:
        """Get names of all available tools."""
        return list(self.tools.keys())
    
    def __str__(self) -> str:
        """String representation showing available tools."""
        return f"MCPTools with {len(self.tools)} tools: {', '.join(self.tools.keys())}"
    
    def __repr__(self) -> str:
        """Detailed representation of the tools collection."""
        return f"MCPTools({len(self.tools)} tools: {list(self.tools.keys())})"

