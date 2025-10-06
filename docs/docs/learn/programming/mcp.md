---
sidebar_position: 3
---

# Model Context Protocol (MCP)

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open protocol that standardizes how applications provide context to language models. DSPy supports MCP, allowing you to use tools from any MCP server with DSPy agents.

## Installation

Install DSPy with MCP support:

```bash
pip install -U "dspy[mcp]"
```

## Overview

MCP enables you to:

- **Use standardized tools** - Connect to any MCP-compatible server
- **Share tools across stacks** - Use the same tools across different frameworks
- **Simplify integration** - Convert MCP tools to DSPy tools with one line

As long as you have an `mcp.ClientSession`, you can use MCP tools with DSPy's `ReAct` module or any other tool-based workflow.

## Using MCP with DSPy

### 1. Stdio Server (Local Process)

The most common way to use MCP is with a local server process communicating via stdio:

```python
import asyncio
import dspy
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Configure the stdio server
    server_params = StdioServerParameters(
        command="python",                    # Command to run
        args=["path/to/your/mcp_server.py"], # Server script path
        env=None,                            # Optional environment variables
    )

    # Connect to the server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()

            # List available tools
            response = await session.list_tools()

            # Convert MCP tools to DSPy tools
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in response.tools
            ]

            # Create a ReAct agent with the tools
            class QuestionAnswer(dspy.Signature):
                """Answer questions using available tools."""
                question: str = dspy.InputField()
                answer: str = dspy.OutputField()

            react_agent = dspy.ReAct(
                signature=QuestionAnswer,
                tools=dspy_tools,
                max_iters=5
            )

            # Use the agent
            result = await react_agent.acall(
                question="What is 25 + 17?"
            )
            print(result.answer)

# Run the async function
asyncio.run(main())
```

### 2. HTTP Server (Remote)

For remote MCP servers over HTTP, use the streamable HTTP transport:

```python
import asyncio
import dspy
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    # Connect to HTTP MCP server
    async with streamablehttp_client("http://localhost:8000/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()

            # List and convert tools
            response = await session.list_tools()
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in response.tools
            ]

            # Create and use ReAct agent
            class TaskSignature(dspy.Signature):
                task: str = dspy.InputField()
                result: str = dspy.OutputField()

            react_agent = dspy.ReAct(
                signature=TaskSignature,
                tools=dspy_tools,
                max_iters=5
            )

            result = await react_agent.acall(task="Check the weather in Tokyo")
            print(result.result)

asyncio.run(main())
```

## Creating an MCP Server

Here's a simple example of an MCP server using FastMCP:

```python
from mcp.server.fastmcp import FastMCP
from datetime import datetime

# Create MCP server
mcp = FastMCP("My Tools Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@mcp.tool()
def get_current_time() -> str:
    """Get the current time."""
    return datetime.now().isoformat()

@mcp.tool()
def search_database(query: str, limit: int = 10) -> list[str]:
    """Search a database with a query string."""
    # Your search logic here
    return [f"Result {i} for '{query}'" for i in range(limit)]
```

Run the server with:

```bash
python your_server.py
```

## Tool Conversion

DSPy automatically handles the conversion from MCP tools to DSPy tools:

```python
# MCP tool from session
mcp_tool = response.tools[0]

# Convert to DSPy tool
dspy_tool = dspy.Tool.from_mcp_tool(session, mcp_tool)

# The DSPy tool preserves:
# - Tool name and description
# - Parameter schemas and types
# - Argument descriptions
# - Async execution support

# Use it like any DSPy tool
result = await dspy_tool.acall(param1="value", param2=123)
```

## Learn More

- [MCP Official Documentation](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [DSPy MCP Tutorial](https://dspy.ai/tutorials/mcp/)
- [DSPy Tools Documentation](./tools.md)

MCP integration in DSPy makes it easy to use standardized tools from any MCP server, enabling powerful agent capabilities with minimal setup.
