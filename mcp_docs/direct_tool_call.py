import os
import sys
import json


from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Any
import asyncio
import dspy  # Using absolute import now
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
        mcp_tools = dspy.MCPTools(session=session, tools_list=tool_list)
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
