# DSPy with Model Context Protocol (MCP) - Integration Guide

This documentation provides a comprehensive guide to integrating and using Model Context Protocol (MCP) tools with DSPy agents and frameworks.

## Table of Contents

1. [Introduction to Model Context Protocol (MCP)](#introduction-to-model-context-protocol-mcp)
2. [DSPy and MCP Integration](#dspy-and-mcp-integration)
3. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Basic Setup](#basic-setup)
4. [Working with MCP Tools](#working-with-mcp-tools)
   - [Creating MCP Tools](#creating-mcp-tools)
   - [Using MCP Tools with DSPy Agents](#using-mcp-tools-with-dspy-agents)
5. [Implementation Patterns](#implementation-patterns)
   - [Simple MCP Integration](#simple-mcp-integration)
   - [MCP with Non-MCP Tools](#mcp-with-non-mcp-tools)
   - [FastAPI Integration](#fastapi-integration)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [API Reference](#api-reference)
9. [Examples](#examples)
   - [Simple MCP Integration](#example-simple-mcp-integration)
   - [Advanced MCP Agent](#example-advanced-mcp-agent)
   - [Direct MCP Tool Calls](#example-direct-mcp-tool-calls)
   - [Non-MCP Tools](#example-non-mcp-tools)
   - [Combined MCP and Non-MCP Tools](#example-combined-mcp-and-non-mcp-tools)
   - [FastAPI Server Integration](#example-fastapi-server-integration)

## Introduction to Model Context Protocol (MCP)

The Model Context Protocol (MCP) is a standard for defining tools and capabilities for AI models. It defines a JSON-RPC based interface that allows models to interact with external tools and services. MCP helps create a consistent interface for tools that can be used across different models and platforms.

Key features of MCP:
- Standardized tool definitions with JSON schema
- Consistent method for tool execution
- Support for asynchronous operations
- Extensible design for various tool types

## DSPy and MCP Integration

DSPy, a framework for programming foundation models with programs and feedback, integrates with MCP to allow models to use tools defined in the MCP specification. This integration provides several benefits:

- Use external tools with DSPy agents like ReAct
- Leverage existing MCP tool libraries
- Combine MCP tools with native DSPy tools
- Build complex multi-tool applications

The integration is provided through:
- `MCPTools` class: A container for MCP tools that makes them compatible with DSPy
- `MCPTool` class: A wrapper for individual MCP tools that converts them to DSPy-compatible tools

## Getting Started

### Prerequisites

To use MCP with DSPy, you'll need:

- Python 3.9 or higher
- DSPy installed (`pip install dspy`)
- The MCP client package (`pip install mcp`)
- Node.js (for running JavaScript-based MCP servers)

### Basic Setup

The basic workflow for using MCP tools with DSPy involves:

1. Initializing an MCP server
2. Establishing an MCP client session
3. Retrieving available tools
4. Creating DSPy wrappers for these tools
5. Using the tools with DSPy reAct

Here's a minimal example:

```python
import asyncio
import os
import dspy

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Configure DSPy with an LLM
    LLM = dspy.LM("gemini/gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))
    dspy.configure(lm=LLM)
    
    # Initialize MCP server and tools
    server_params = StdioServerParameters(            
        command="npx",
        args=["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            
            # Create MCPTools instance
            mcp_tools = dspy.MCPTools(session=session, tools_list=tools.tools)
            
            # Create ReAct agent with MCP tools
            react_agent = dspy.ReAct("input->output", mcp_tools.get_tools())
            
            # Run the agent
            result = await react_agent(
                input="Find a place to stay in New York for 2 adults from May 1-5, 2025."
            )
            
            print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Working with MCP Tools

### Creating MCP Tools

DSPy provides a wrapper around MCP tools through the `MCPTools` and `MCPTool` classes. These convert MCP tool definitions into DSPy-compatible tools.

The `MCPTools` class handles:
- Converting MCP tool definitions to DSPy tools
- Managing the MCP session
- Providing access to tools by name
- Returning all tools as a list for use with agents

### Using MCP Tools with DSPy Agents

Once wrapped, MCP tools can be used with any DSPy agent, including ReAct. The tools are used like any other DSPy tool:

```python
# Create ReAct agent with MCP tools
react_agent = dspy.ReAct("input->output", mcp_tools.get_tools())

# Run the agent with a query
result = await react_agent(
    input="Find a place to stay in New York for 2 adults from May 1-5, 2025."
)
```

The agent will be able to:
1. Understand the available tools
2. Choose the appropriate tool for a task
3. Execute the tool with proper arguments
4. Process the results and continue reasoning

## Implementation Patterns

### Simple MCP Integration

For a simple integration with clean resource management, use Python's context managers:

```python
async def main():
    # Initialize MCP server and tools
    server_params = StdioServerParameters(command="npx", args=[...])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            
            # Create MCPTools and agent
            mcp_tools = dspy.MCPTools(session=session, tools_list=tools.tools)
            react_agent = dspy.ReAct("input->output", mcp_tools.get_tools())
            
            # Run the agent
            result = await react_agent(input="Your query here")
```

### MCP with Non-MCP Tools

You can combine MCP tools with regular DSPy tools:

```python
def calculate_sum(a: int, b: int) -> int:
   """Calculate the sum of two integers."""
   return a + b

# Create ReAct agent with both MCP and regular tools
react_agent = dspy.ReAct("input->output", mcp_tools.get_tools() + [calculate_sum])
```

### FastAPI Integration

For production applications, you can integrate MCP tools with a FastAPI server:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources
    session, tool_list = await initialize_stdio_client()
    mcp_tools = dspy.MCPTools(session=session, tools_list=tool_list)
    global react_agent
    react_agent = dspy.ReAct("input->output", mcp_tools.get_tools())
    yield
    # Cleanup resources
    await cleanup()

app = FastAPI(lifespan=lifespan)

@app.post("/api/query")
async def process_query(request: QueryRequest):
    react_result = await react_agent(input=request.query)
    return QueryResponse(result=react_result.output, status="success")
```

## Best Practices

1. **Resource Management**
   - Always clean up MCP resources properly
   - Use context managers or a dedicated cleanup function
   - Handle connection errors gracefully

2. **Error Handling**
   - Wrap MCP tool calls in try/except blocks
   - Provide meaningful error messages to agents
   - Handle connection issues and timeouts

3. **Asynchronous Operations**
   - Use `async`/`await` consistently
   - Don't mix sync and async code without proper planning
   - Understand event loop behavior when using `asyncio.run()`

4. **Tool Composition**
   - Combine MCP tools with regular DSPy tools when needed
   - Consider tool dependencies and interactions
   - Design tools with specific, focused purposes

## Troubleshooting

### Common Issues

1. **MCP Server Connection Failures**
   - Ensure the MCP server package is installed (`npm install -g @openbnb/mcp-server-airbnb`)
   - Check for network issues or firewall restrictions
   - Verify Node.js is properly installed

2. **Tool Execution Errors**
   - Validate input parameters against the tool's schema
   - Check for rate limiting or API key issues with external services
   - Examine logs for detailed error messages

3. **Event Loop Issues**
   - Avoid mixing sync and async code improperly
   - Use `asyncio.run()` at the top level only
   - Ensure cleanup happens in the same task context as initialization

## API Reference

### MCPTools Class

```python
class MCPTools:
    def __init__(self, session: Any, tools_list: List[Any]):
        """Initialize the MCPTools collection."""
        
    def __getitem__(self, tool_name: str) -> MCPTool:
        """Get a tool by name."""
        
    def get_tools(self) -> List[Tool]:
        """Get all tools as a list."""
        
    def get_tool_names(self) -> List[str]:
        """Get names of all available tools."""
```

### MCPTool Class

```python
class MCPTool(Tool):
    def __init__(self, tool_info: Any, session: Any):
        """Create a DSPy Tool from an MCP tool description."""
        
    async def call_tool_async(self, **kwargs: Any) -> Any:
        """Execute the MCP tool asynchronously."""
```

### Helper Functions

```python
def map_json_schema_to_tool_args(schema: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Type], Dict[str, str]]:
    """Maps a JSON schema to tool arguments compatible with DSPy Tool."""
    
async def initialize_stdio_client(command: str, command_args: list[str] = [], env: dict[str, str] | None = None):
    """Initialize an MCP client session with the specified command."""
    
async def cleanup() -> None:
    """Clean up MCP server resources."""
```

## Examples

The following examples demonstrate different patterns for using MCP with DSPy. Each example is designed to showcase a specific integration pattern or use case.

### Example: Simple MCP Integration

This example shows the simplest way to integrate MCP tools with DSPy using context managers for clean resource handling:

```python
# simple_mcp.py
import os
import sys
import json
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import dspy



async def main():
   """Main entry point for the async application."""
   try:
      print("Starting MCP client initialization...")
      # Configure DSPy with LLM
      LLM = dspy.LM("gemini/gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))
      dspy.configure(lm=LLM)
      
      # Initialize MCP server and tools
      server_params = StdioServerParameters(            
         command="npx",
         args=[
               "-y",
               "@openbnb/mcp-server-airbnb",
               "--ignore-robots-txt"
         ])
      async with stdio_client(server_params) as (read, write):
         async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            tools = tools.tools
            print("\nCreating MCPTools instance...")
            mcp_tools = dspy.MCPTools(session=session, tools_list=tools)
            
            # Create ReAct agent in the same async context
            react_agent = dspy.ReAct("input->output", mcp_tools.get_tools())
            
            # Run the agent (will use the existing event loop)
            print("\nRunning ReAct agent...")
            react_result = await react_agent(
                  input="Find a place to stay in New York for 2 adults from 2025-05-01 to 2025-05-05."
            )
            
            print("\nReAct Result:")
            print(react_result)
            
   except Exception as e:
      print(f"Error in main: {str(e)}")
      raise

if __name__ == "__main__":
    asyncio.run(main())
```

This approach uses Python's context managers (`async with`) to handle MCP resources properly. The context managers ensure that resources are cleaned up even if an exception occurs.

### Example: Advanced MCP Agent

This example shows a more robust implementation with proper resource management using a dedicated resource management pattern:

```python
# agent.py
import os
import sys
import json
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Any
from contextlib import AsyncExitStack

import dspy


stdio_context: Any | None = None
session: ClientSession | None = None
_cleanup_lock: asyncio.Lock = asyncio.Lock()
exit_stack: AsyncExitStack = AsyncExitStack()

async def initialize_stdio_client(
        command: str,
        command_args: list[str] = [],
        env: dict[str, str] | None = None     
):
    global stdio_context, session, exit_stack
    if stdio_context is not None:
        return stdio_context

    print(f"Initializing MCP server with command: {command} {' '.join(command_args)}")
    server_params = StdioServerParameters(
         command=command,
         args=command_args,
         env=env if env else None
    )
    try:
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        session = await exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        tools = await session.list_tools()

        return session, tools.tools
    except Exception as e:
        print(f"Error initializing MCP server: {str(e)}")
        await cleanup()
        raise

async def cleanup() -> None:
    """Clean up server resources."""
    global stdio_context, session, exit_stack
    print("Cleaning up MCP server resources...")
    async with _cleanup_lock:
            await exit_stack.aclose()
            session = None
            stdio_context = None
    print("Cleanup complete.")

async def main():
    """Main entry point for the async application."""
    try:
        print("Starting MCP client initialization...")
        # Configure DSPy with LLM
        LLM = dspy.LM("gemini/gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))
        dspy.configure(lm=LLM)
        
        # Initialize MCP server and tools
        session, tool_list = await initialize_stdio_client(
            command="npx",
            command_args=[
                "-y",
                "@openbnb/mcp-server-airbnb",
                "--ignore-robots-txt"
            ],
            env={}
        )
        
        print("\nCreating MCPTools instance...")
        mcp_tools = dspy.MCPTools(session=session, tools_list=tool_list)
        
        # Create ReAct agent in the same async context
        react_agent = dspy.ReAct("input->output", mcp_tools.get_tools())
        
        # Run the agent (will use the existing event loop)
        print("\nRunning ReAct agent...")
        react_result = await react_agent(
            input="Find a place to stay in New York for 2 adults from 2025-05-01 to 2025-05-05."
        )
        
        print("\nReAct Result:")
        print(react_result)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise
    finally:
        await cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

This implementation uses a more structured approach with dedicated functions for initialization and cleanup. It also handles exceptions more carefully with a `finally` block to ensure resources are always cleaned up.

### Example: Direct MCP Tool Calls

This example demonstrates how to use MCP tools directly without a ReAct agent:

```python
# direct.py
import os
import sys
import json


from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Any
import asyncio
import dspy
from contextlib import AsyncExitStack

stdio_context: Any | None = None
session: ClientSession | None = None
_cleanup_lock: asyncio.Lock = asyncio.Lock()
exit_stack: AsyncExitStack = AsyncExitStack()

async def initialize_stdio_client(
        command: str,
        command_args: list[str] = [],
        env: dict[str, str] | None = None     
):
    global stdio_context, session, exit_stack
    if stdio_context is not None:
        return stdio_context

    print(f"Initializing MCP server with command: {command} {' '.join(command_args)}")
    server_params = StdioServerParameters(
         command=command,
         args=command_args,
         env=env if env else None
    )
    try:
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        session = await exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        tools = await session.list_tools()

        return session, tools.tools
    except Exception as e:
        print(f"Error initializing MCP server: {str(e)}")
        await cleanup()
        raise

async def cleanup() -> None:
    """Clean up server resources."""
    global stdio_context, session, exit_stack
    print("Cleaning up MCP server resources...")
    async with _cleanup_lock:
            await exit_stack.aclose()
            session = None
            stdio_context = None
    print("Cleanup complete.")


# Create a safer async main function to properly handle errors
async def async_main():
    global session
    try:
        
        # Try initializing the MCP server - with proper error handling
        print("Starting MCP client initialization...")
        session, tool_list = await initialize_stdio_client(
            command="npx",
            command_args=[
                "-y",
                "@openbnb/mcp-server-airbnb",
                "--ignore-robots-txt"
            ],
            env={}
        )
        
        print("\nCreating MCPTools instance...")
        mcp_tools = dspy.utils.MCPTools(session=session, tools_list=tool_list)
        print("MCPTools instance created successfully.")
        tool = mcp_tools.get_tools()[0]
        

        search_result = await tool.aexecute(
            location="New York",
            checkin="2025-05-01",
            checkout="2025-05-05",
            adults=2
        )
        print("\nSearch Result:", search_result[:500] + "..." if len(str(search_result)) > 500 else search_result)

            
    finally:
        # Make sure to clean up resources
        if session:
            await cleanup()

# Run the main async function
if __name__ == "__main__":
    asyncio.run(async_main())
```

This example shows how to use MCPTools without a ReAct agent, directly calling a tool with specific parameters. This approach is useful when you want more control over the tool execution or when you want to use MCP tools in a non-agent context.

### Example: Non-MCP Tools

DSPy can also work with regular Python functions as tools, independent of MCP:

```python
# non_mpc.py
import dspy
import os
def calculate_sum(a: int, b: int) -> int:
   """Calculate the sum of two integers."""
   return a + b

def main():
   """Main entry point for the async application."""
   print("Starting MCP client initialization...")
   # Configure DSPy with LLM
   LLM = dspy.LM("gemini/gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))
   dspy.configure(lm=LLM)
   

   print("\nCreating MCPTools instance...")
   tools = [calculate_sum]
   
   # Create ReAct agent in the same async context
   react_agent = dspy.ReAct("input->output", tools)
   
   # Run the agent (will use the existing event loop)
   print("\nRunning ReAct agent...")
   react_result = react_agent(
      input="what is the sum of 5 and 10?",
   )
   
   
   print("\nReAct Result:")
   print(react_result)

if __name__ == "__main__":
   main()
```

This example demonstrates how to use regular Python functions as DSPy tools without any MCP integration. This is useful for simple tools or when you want to build applications that don't rely on external MCP servers.

### Example: Combined MCP and Non-MCP Tools

You can combine MCP tools with regular Python functions:

```python
# agent_mcp_non_mcp.py
import os
import sys
import json
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Any
from contextlib import AsyncExitStack

import dspy


stdio_context: Any | None = None
session: ClientSession | None = None
_cleanup_lock: asyncio.Lock = asyncio.Lock()
exit_stack: AsyncExitStack = AsyncExitStack()

def calculate_sum(a: int, b: int) -> int:
   """Calculate the sum of two integers."""
   return a + b

async def initialize_stdio_client(
        command: str,
        command_args: list[str] = [],
        env: dict[str, str] | None = None     
):
    global stdio_context, session, exit_stack
    if stdio_context is not None:
        return stdio_context

    print(f"Initializing MCP server with command: {command} {' '.join(command_args)}")
    server_params = StdioServerParameters(
         command=command,
         args=command_args,
         env=env if env else None
    )
    try:
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        session = await exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        tools = await session.list_tools()

        return session, tools.tools
    except Exception as e:
        print(f"Error initializing MCP server: {str(e)}")
        await cleanup()
        raise

async def cleanup() -> None:
    """Clean up server resources."""
    global stdio_context, session, exit_stack
    print("Cleaning up MCP server resources...")
    async with _cleanup_lock:
            await exit_stack.aclose()
            session = None
            stdio_context = None
    print("Cleanup complete.")

async def main():
    """Main entry point for the async application."""
    try:
        print("Starting MCP client initialization...")
        # Configure DSPy with LLM
        LLM = dspy.LM("gemini/gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))
        dspy.configure(lm=LLM)
        
        # Initialize MCP server and tools
        session, tool_list = await initialize_stdio_client(
            command="npx",
            command_args=[
                "-y",
                "@openbnb/mcp-server-airbnb",
                "--ignore-robots-txt"
            ],
            env={}
        )
        
        print("\nCreating MCPTools instance...")
        mcp_tools = dspy.MCPTools(session=session, tools_list=tool_list)
        
        # Create ReAct agent in the same async context
        react_agent = dspy.ReAct("input->output", mcp_tools.get_tools()+[calculate_sum])
        
        # Run the agent (will use the existing event loop)
        print("\nRunning ReAct agent...")
        react_result = await react_agent(
            input="Find a place to stay in New York for 2 adults from 2025-05-01 to 2025-05-05. and calculate the sum of 5 and 10.",
        )

        print("\nReAct Result:")
        print(react_result.output)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise
    finally:
        await cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

This example showcases how to combine MCP tools with simple Python functions in a single ReAct agent. This hybrid approach allows you to leverage both external MCP tools and custom local functionality.

### Example: FastAPI Server Integration

For production applications, you can integrate MCP tools with a FastAPI server:

```python
# fastapi_dspy_mcp.py
import os
import sys
import json
import asyncio
from typing import Any, Dict, List, Optional
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import dspy


# Global variables for MCP client session
stdio_context: Any | None = None
session: ClientSession | None = None
_cleanup_lock: asyncio.Lock = asyncio.Lock()
exit_stack: AsyncExitStack = AsyncExitStack()

# Default MCP configuration
DEFAULT_MODEL = "gemini/gemini-2.0-flash"
DEFAULT_MCP_COMMAND = "npx"
DEFAULT_MCP_ARGS = ["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"]
DEFAULT_ENV_VARS = {}

react_agent: Any | None = None

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
   global react_agent, session, exit_stack
   # Configure DSPy with LLM
   api_key = os.getenv("GOOGLE_API_KEY")
   if not api_key:
      raise HTTPException(status_code=400, detail="GOOGLE_API_KEY environment variable not set")
   
   LLM = dspy.LM(DEFAULT_MODEL, api_key=api_key)
   dspy.configure(lm=LLM)
   
   # Initialize MCP server and tools
   session, tool_list = await initialize_stdio_client()
   
   # Create MCPTools instance
   mcp_tools = dspy.MCPTools(session=session, tools_list=tool_list)
   
   # Create ReAct agent in the same async context
   react_agent = dspy.ReAct("input->output", mcp_tools.get_tools())
   
   yield
   # Shutdown - clean up resources 
   # This will be executed in the same task context where resources were initialized
   await cleanup()

# Create FastAPI app with lifespan
app = FastAPI(
    title="DSPy MCP API",
    description="FastAPI server for DSPy Model Context Protocol interactions",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    result: Any
    status: str

# MCP initialization function
async def initialize_stdio_client(
        command: str = DEFAULT_MCP_COMMAND,
        command_args: list[str] = DEFAULT_MCP_ARGS,
        env: dict[str, str] | None = DEFAULT_ENV_VARS     
):
    global stdio_context, session, exit_stack, react_agent
    if stdio_context is not None:
        return session, stdio_context

    print(f"Initializing MCP server with command: {command} {' '.join(command_args)}")
    server_params = StdioServerParameters(
         command=command,
         args=command_args,
         env=env if env else None
    )
    try:
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        session = await exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        tools = await session.list_tools()
        stdio_context = tools.tools

        return session, tools.tools
    except Exception as e:
        print(f"Error initializing MCP server: {str(e)}")
        await cleanup()
        raise

async def cleanup() -> None:
    """Clean up server resources."""
    global stdio_context, session, exit_stack
    print("Cleaning up MCP server resources...")
    async with _cleanup_lock:
        if session is not None:
            await exit_stack.aclose()
            session = None
            stdio_context = None
    print("Cleanup complete.")

# FastAPI endpoints
@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # Run the agent
        react_result = await react_agent(input=request.query)
        
        return QueryResponse(
            result=react_result.output,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_dspy_mcp:app", host="0.0.0.0", port=8000, reload=True)
```

This implementation shows how to create a production-ready FastAPI server that integrates DSPy's ReAct agent with MCP tools. It uses FastAPI's lifespan feature to manage the MCP resources and provides a clean API endpoint for queries.

## Additional Resources

- [MCP](https://github.com/modelcontextprotocol/python-sdk/)
