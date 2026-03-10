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

DSPy provides `stdio_mcp_tools` and `http_mcp_tools` async context managers that handle the full MCP session lifecycle — connecting, initializing, listing tools, and converting them to `dspy.Tool` objects — so you can focus on building your agent.

## Using MCP with DSPy

### 1. HTTP Server (Remote)

For remote MCP servers over HTTP, use the streamable HTTP transport:

```python
import asyncio
import dspy
from dspy.utils.mcp import http_mcp_tools

async def main():
    # Connect to HTTP MCP server
    async with http_mcp_tools("http://localhost:8000/mcp") as tools:
        # Create and use ReAct agent
        class TaskSignature(dspy.Signature):
            task: str = dspy.InputField()
            result: str = dspy.OutputField()

        react_agent = dspy.ReAct(
            signature=TaskSignature,
            tools=tools,
            max_iters=5,
        )

        result = await react_agent.acall(task="Check the weather in Tokyo")
        print(result.result)

asyncio.run(main())
```

### 2. Stdio Server (Local Process)

The most common way to use MCP is with a local server process communicating via stdio:

```python
import asyncio
import dspy
from mcp import StdioServerParameters
from dspy.utils.mcp import stdio_mcp_tools

async def main():
    # Configure the stdio server
    server_params = StdioServerParameters(
        command="python",                    # Command to run
        args=["path/to/your/mcp_server.py"], # Server script path
    )

    # Connect to the server
    async with stdio_mcp_tools(server_params) as tools:
        # Create a ReAct agent with the tools
        class QuestionAnswer(dspy.Signature):
            """Answer questions using available tools."""
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        react_agent = dspy.ReAct(
            signature=QuestionAnswer,
            tools=tools,
            max_iters=5,
        )

        # Use the agent
        result = await react_agent.acall(question="What is 25 + 17?")
        print(result.answer)

# Run the async function
asyncio.run(main())
```

## Filtering Tools

Both adapters support `include_tools` and `exclude_tools` to control which tools are loaded from the server. Unknown tool names raise a `ValueError`.

```python
# Only load specific tools
async with stdio_mcp_tools(params, include_tools=["search", "lookup"]) as tools:
    ...

# Load all tools except specific ones
async with http_mcp_tools(url, exclude_tools=["dangerous_tool"]) as tools:
    ...
```

## Timeout

Pass `timeout` (in seconds) to set a read timeout on the MCP session:

```python
async with stdio_mcp_tools(params, timeout=30) as tools:
    ...
```

## Learn More

- [MCP Official Documentation](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [DSPy MCP Tutorial](https://dspy.ai/tutorials/mcp/)
- [DSPy Tools Documentation](./tools.md)
