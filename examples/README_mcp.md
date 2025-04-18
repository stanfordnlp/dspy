# DSPy Model Context Protocol (MCP) Guide

## Overview

The Model Context Protocol (MCP) is a flexible interface that allows DSPy to connect with external tools and services through a standardized protocol. This README provides guidance on how to use MCP with DSPy, particularly focusing on multi-server configurations.

## Multi-Server Example

The `mcp_multi_server_example.py` script demonstrates how to use DSPy with multiple MCP servers simultaneously. This allows you to leverage tools from different servers in a single DSPy application.

### Prerequisites

- Python 3.8+
- DSPy installed
- Access to MCP-compatible services or local servers
- Proper API keys (if connecting to external LLM providers)

### Configuration

The multi-server example uses a JSON configuration file (`servers_config.json`) to define MCP server connections. Example structure:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "F:/AI/DSPy_MCP/test"
      ]
    },
    "airbnb": {
      "command": "npx",
      "args": [
        "-y",
        "@openbnb/mcp-server-airbnb",
        "--ignore-robots-txt"
      ]
    }
  }
}
```

or 

```python
config = {
        "mcpServers": {
            "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "F:/AI/DSPy_MCP/test"
            ]
            },
            "airbnb": {
            "command": "npx",
            "args": [
                "-y",
                "@openbnb/mcp-server-airbnb",
                "--ignore-robots-txt"
            ]
            }
        }
}
```

### Key Components

The example demonstrates:

1. **Server Management**: Using `MCPServerManager` to initialize and manage multiple MCP servers
2. **Tool Discovery**: Retrieving tools from all connected servers
3. **ReAct Agent**: Using DSPy's ReAct framework with MCP tools
4. **Async Execution**: Handling asynchronous operations with MCP servers

### Example Usage

```python
import asyncio
import os
import dspy

# Define a simple DSPy Signature for the agent
class MultiServerSignature(dspy.Signature):
    """Perform operations using tools potentially available across multiple MCP servers."""
    request: str = dspy.InputField(desc="The user's request, potentially requiring external tools.")
    output: str = dspy.OutputField(desc="The final response to the user's request after potentially using tools.")

async def main():
    # Initialize language model
    lm = dspy.LM("gemini/gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))
    dspy.configure(lm=lm)
    
    # Configure MCP servers
    config_path = "path/to/servers_config.json"
    
    async with dspy.MCPServerManager() as server_manager:
        # Load and initialize servers from config
        config = server_manager.load_config(config_path)
        await server_manager.initialize_servers(config)
        
        # Get all tools from all connected servers
        all_mcp_tools = await server_manager.get_all_tools()
        
        # Create a ReAct agent with the tools
        react_agent = dspy.ReAct(
            MultiServerSignature,
            tools=all_mcp_tools,
            max_iters=7
        )
        
        # Execute a request
        result = await react_agent.async_forward(request="Your request here")
        print("Final Result:", result.output)

if __name__ == "__main__":
    asyncio.run(main())
```


## Additional Resources

- [DSPy Documentation](https://dspy.ai/)
- [MCP Specification](https://github.com/modelcontextprotocol)