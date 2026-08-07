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

- **Use standardized tools** - Connect to any MCP-compatible server.
- **Share tools across stacks** - Use the same tools across different frameworks.
- **Simplify integration** - Convert MCP tools to DSPy tools with one line.

DSPy does not handle MCP server connections directly. You can use client interfaces of the `mcp` library to establish the connection and pass the client or session to `dspy.Tool.from_mcp_tool` in order to convert mcp tools into DSPy tools.

DSPy works with both major versions of the `mcp` Python SDK. The examples below note where the two versions differ.

## Using MCP with DSPy

### 1. HTTP Server (Remote)

For remote MCP servers over HTTP, use the streamable HTTP transport. With `mcp` SDK v2, pass the server URL to the high-level `Client`:

```python
import asyncio
import dspy
from mcp.client import Client

async def main():
    # Connect to HTTP MCP server
    async with Client("http://localhost:8000/mcp") as client:
        # List and convert tools
        response = await client.list_tools()
        dspy_tools = [
            dspy.Tool.from_mcp_tool(client, tool)
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

With `mcp` SDK v1, the same connection uses `streamablehttp_client` and a `ClientSession`:

```python
import asyncio
import dspy
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    async with streamablehttp_client("http://localhost:8000/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            response = await session.list_tools()
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in response.tools
            ]
            # Build and run your agent as in the v2 example above.

asyncio.run(main())
```

### 2. Stdio Server (Local Process)

The most common way to use MCP is with a local server process communicating via stdio. This example works with both SDK versions. Under SDK v2 the protocol no longer has a handshake, and `session.initialize()` is kept as a compatibility method, so calling it is still safe:

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

### Structured results

When an MCP server returns structured content alongside text, the DSPy tool returns the structured value instead of the text rendering. Servers built with the official Python SDK produce structured content for any tool with a typed return value. A tool that returns a Pydantic model produces a plain dict, and a tool that returns a primitive value such as `int` produces that value directly. Tools without a declared return type keep returning text.

If a server asks for additional input during a tool call, which the 2026-07-28 specification calls a multi round-trip request, the DSPy tool raises a `RuntimeError` that explains what the server requested. DSPy cannot answer these requests itself.

## Learn More

- [MCP Official Documentation](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [DSPy MCP Tutorial](https://dspy.ai/tutorials/mcp/)
- [DSPy Tools Documentation](./tools.md)

MCP integration in DSPy makes it easy to use standardized tools from any MCP server, enabling powerful agent capabilities with minimal setup.
