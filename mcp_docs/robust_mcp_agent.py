import os
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
        
        # # If the result is a coroutine (which it will be in an async context), await it
        # if asyncio.iscoroutine(react_result):
        #     react_result = await react_result
        
        print("\nReAct Result:")
        print(react_result)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise
    finally:
        await cleanup()

if __name__ == "__main__":
    asyncio.run(main())


