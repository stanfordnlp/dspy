"""
Enhanced resource management for Model Context Protocol (MCP) in DSPy.

This module provides robust utilities for managing MCP server connections,
including support for multiple concurrent sessions with proper resource handling,
and integrates MCP tools seamlessly into the DSPy framework.
"""

import asyncio
import json
import logging
import os
import shutil
import sys
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Import DSPy tools
from dspy.primitives.tool import Tool

#---------------------------------------------------------------------
# Constants and Configuration
#---------------------------------------------------------------------

# Default logging level - can be overridden
DEFAULT_LOG_LEVEL = logging.WARNING  # Less verbose default

#---------------------------------------------------------------------
# Logging Configuration
#---------------------------------------------------------------------

# Configure logger - initially null, configured by setup functions
logger = logging.getLogger("dspy.mcp")
logger.addHandler(logging.NullHandler())  # Prevent "no handler" warnings

def setup_logging(log_level=logging.INFO, log_to_file=False, log_dir=None):
    """Configure logging with optional file output and cleanup."""
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file:
        log_dir = Path(log_dir) if log_dir else Path.home() / "dspy_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean old log files
        clean_old_logs(log_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"dspy_mcp_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
        
    return root_logger

def clean_old_logs(log_dir, keep_last=10):
    """Clean old log files, keeping only the specified number of latest files."""
    log_files = list(log_dir.glob("dspy_mcp_*.log"))
    if len(log_files) > keep_last:
        log_files.sort(key=lambda f: f.stat().st_mtime)
        for f in log_files[:-keep_last]:
            try:
                f.unlink()
            except (PermissionError, OSError):
                pass

def disable_logging():
    """Completely disable all logging from the MCP module."""
    mcp_logger = logging.getLogger("dspy.mcp")
    for handler in mcp_logger.handlers[:]:
        mcp_logger.removeHandler(handler)
    mcp_logger.setLevel(logging.CRITICAL + 1)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.CRITICAL + 1)
    mcp_logger.addHandler(logging.NullHandler())
    return mcp_logger

# Default setup
setup_logging(log_level=DEFAULT_LOG_LEVEL)

#---------------------------------------------------------------------
# Schema Utilities
#---------------------------------------------------------------------

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

#---------------------------------------------------------------------
# Core Classes
#---------------------------------------------------------------------
class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []
        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    # Create a placeholder function with the tool's name
                    tool_func = lambda tool_name=tool.name, **kwargs: f"{tool_name}"
                    # Append the tool with the placeholder function
                    tools.append(
                        Tool(
                            func=tool_func,
                            name=tool.name,
                            desc=tool.description,
                            args=tool.inputSchema,
                        )
                    )

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments['kwargs'])

                return result

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")



class MCPTool(Tool):
    """Wrapper for an MCP tool, compatible with DSPy agents."""

    def __init__(self, manager: 'MCPServerManager', server_name: str, tool_info: Any):
        """Create a DSPy Tool from an MCP tool description."""
        self.manager = manager
        self.server_name = server_name
        self._raw_tool_info = tool_info

        name, desc, input_schema = self._extract_tool_info(tool_info)
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
        if isinstance(tool_info, str):
            try:
                parsed = json.loads(tool_info)
                if isinstance(parsed, dict):
                    return (
                        parsed.get('name', 'unknown_mcp_tool'),
                        parsed.get('description', 'No description available.'),
                        parsed.get('inputSchema', None)
                    )
            except json.JSONDecodeError:
                return tool_info, "No description available.", None
        
        # Handle dictionary
        if isinstance(tool_info, dict):
            return (
                tool_info.get('name', 'unknown_mcp_tool'),
                tool_info.get('description', 'No description available.'),
                tool_info.get('inputSchema', None)
            )
            
        # Handle object with attributes (original behavior)
        try:
            name = getattr(tool_info, 'name', 'unknown_mcp_tool')
            desc = getattr(tool_info, 'description', 'No description available.')
            input_schema = getattr(tool_info, 'inputSchema', None)
            return name, desc, input_schema
        except (AttributeError, TypeError):
            # Last resort fallback for unexpected formats
            return str(tool_info), "No description available.", None

    async def call_tool_async(self, **kwargs: Any) -> Any:
        """Execute the MCP tool."""
        server = self._get_server()
        try:
            result = await server.execute_tool(self.name, kwargs)
            return self._process_result(result)
        except Exception:
            raise

    def _get_server(self) -> Server:
        """Get and validate the server for this tool."""
        server = self.manager.servers.get(self.server_name)
        if not server:
            raise ValueError(f"Server '{self.server_name}' not found for tool '{self.name}'")
        if not server._is_initialized or not server.session:
            raise RuntimeError(f"Server '{self.server_name}' not ready for execution")
        return server

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


class MCPServerManager:
    """Manages multiple MCP server connections and DSPy-compatible tools."""
    def __init__(self ) -> None:
        self.servers = {}


    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load server configuration from a JSON file."""
        if not file_path or not isinstance(file_path, str):
            raise ValueError("Invalid file path provided.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, "r") as f:
                config = json.load(f)
            return config
        except json.JSONDecodeError:
            raise
        except Exception:
            raise

    async def initialize_servers(self, config: Dict[str, Any]) -> None:
        """Initialize all servers defined in the configuration."""
        if "mcpServers" not in config or not isinstance(config["mcpServers"], dict):
            raise ValueError("Configuration must contain an 'mcpServers' dictionary")
        
        server_configs = config["mcpServers"]
        if not server_configs:
            return

        # Create all server objects first
        for name, server_config in server_configs.items():
            if not isinstance(server_config, dict) or name in self.servers:
                continue
            self.servers[name] = Server(name, server_config)
        
        # Then initialize them sequentially
        for name, server in list(self.servers.items()):
            # try:
            await server.initialize()
            server._is_initialized = True
            # except Exception as e:
            #     logging.error(f"Failed to initialize server {name}: {e}")
            #     await self.cleanup()
            #     return
        
        logging.info(f"Initialized servers: {list(self.servers.keys())}")
        
    async def get_all_tools(self) -> List[MCPTool]:
        """Retrieve and wrap all available tools from all initialized servers."""
        all_mcp_tools = []
        
        if not self.servers:
            return all_mcp_tools
        
        # Collect tools server by server to avoid asyncio resource issues
        for server_name, server in self.servers.items():
            if not server._is_initialized:
                continue
                
            # Get tools directly, avoiding task creation which can lead to resource conflicts
            tools_list = await server.list_tools()
            logging.info(f"Tools from server {server_name}: {tools_list}")
            
            if not isinstance(tools_list, list):
                continue

                
            # Process tools from this server
            for tool_info in tools_list:
                mcp_tool_instance = MCPTool(self, server_name, tool_info)
                all_mcp_tools.append(mcp_tool_instance) 

        return all_mcp_tools

    async def cleanup(self) -> None:
        """Clean up resources for all managed serverconnections."""
        if not self.servers:
            return
        for name, server in self.servers.items():
            try:
                await server.cleanup()
            except Exception as e:
                logging.error(f"Error during cleanup of server {name}: {e}")

        self.servers = {}

    async def __aenter__(self) -> 'MCPServerManager':
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.cleanup()