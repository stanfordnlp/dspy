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

DSPy does not handle MCP server connections directly. Use a client interface from the `mcp` library, then pass the client or session to `dspy.Tool.from_mcp_tool`. DSPy supports both major versions of the `mcp` Python SDK.

## Using MCP with DSPy

### 1. HTTP Server (Remote)

With `mcp` SDK v2, use the high-level `Client` for a remote server:

```python
import asyncio
import dspy
from mcp.client import Client

async def main():
    async with Client("http://localhost:8000/mcp") as client:
        response = await client.list_tools()
        dspy_tools = [dspy.Tool.from_mcp_tool(client, tool) for tool in response.tools]

asyncio.run(main())
```

With SDK v1, use the streamable HTTP transport and a `ClientSession`:

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

### 2. Stdio Server (Local Process)

The most common way to use MCP is with a local server process communicating via stdio. This example works with both SDK versions:

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

### Choosing the result representation

By default, DSPy uses the MCP result's model-facing `content` field. This preserves the existing behavior: one text block becomes a string, multiple text blocks become a list, and results without text fall back to their non-text content.

For programmatic workflows that need the MCP result's machine-readable JSON value, opt into structured content when converting the tool:

```python
structured_tool = dspy.Tool.from_mcp_tool(
    session,
    mcp_tool,
    result_mode="structured",
)

result = await structured_tool.acall(param1="value", param2=123)
```

Structured mode returns `structuredContent` exactly as the server sent it. Objects, arrays, strings, numbers, booleans, JSON `null`, and empty values are preserved without parsing or unwrapping. For example, the official MCP Python SDK may represent a tool annotated `-> int` as `{"result": 3}`; DSPy returns that entire object rather than guessing that the `result` field is an SDK-generated envelope. If a result has no structured content, DSPy falls back to the default content conversion.

Use the default mode for tools primarily observed by a ReAct agent or another language model, since MCP `content` is the server's model-facing representation. Use structured mode when application code needs native JSON values for validation, indexing, or explicit tool chaining. In both modes, treat tool results as untrusted server data and validate them before passing them to sensitive operations.

### Running against maintained MCP servers

The following complete example uses two maintained [MCP reference servers](https://github.com/modelcontextprotocol/servers): Everything, which returns a domain-shaped weather object, and Filesystem, which returns file content in an object. Neither requires API credentials. The package versions are pinned to the versions used to verify these outputs; Node.js and `npx` must be installed.

```python
import asyncio
import tempfile
from pathlib import Path

import dspy
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def call_in_both_modes(server, tool_name, arguments):
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            mcp_tool = next(tool for tool in tools if tool.name == tool_name)

            text_tool = dspy.Tool.from_mcp_tool(session, mcp_tool)
            structured_tool = dspy.Tool.from_mcp_tool(
                session,
                mcp_tool,
                result_mode="structured",
            )
            return (
                await text_tool.acall(**arguments),
                await structured_tool.acall(**arguments),
            )


async def main():
    everything = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-everything@2026.7.4"],
    )
    text, structured = await call_in_both_modes(
        everything,
        "get-structured-content",
        {"location": "New York"},
    )
    print(repr(text))
    # '{"temperature":33,"conditions":"Cloudy","humidity":82}'
    print(structured)
    # {'temperature': 33, 'conditions': 'Cloudy', 'humidity': 82}

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "example.txt")
        path.write_text("alpha\nbeta\ngamma\n")
        filesystem = StdioServerParameters(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-filesystem@2026.7.10",
                directory,
            ],
        )
        text, structured = await call_in_both_modes(
            filesystem,
            "read_text_file",
            {"path": str(path), "head": 2},
        )
        print(repr(text))
        # 'alpha\nbeta'
        print(structured)
        # {'content': 'alpha\nbeta'}


asyncio.run(main())
```

These results illustrate why DSPy keeps the representations separate: model-facing `content` is already useful text, while `structuredContent` preserves the native object for indexing and validation. The Filesystem server limits access to the directory passed on its command line. Review and pin any server package before using it with sensitive data.

## Learn More

- [MCP Official Documentation](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [DSPy MCP Tutorial](https://dspy.ai/tutorials/mcp/)
- [DSPy Tools Documentation](./tools.md)

MCP integration in DSPy makes it easy to use standardized tools from any MCP server, enabling powerful agent capabilities with minimal setup.
