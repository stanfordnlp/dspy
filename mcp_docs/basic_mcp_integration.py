import os
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
            
            # # If the result is a coroutine (which it will be in an async context), await it
            # if asyncio.iscoroutine(react_result):
            #     react_result = await react_result
            
            print("\nReAct Result:")
            print(react_result)
            
   except Exception as e:
      print(f"Error in main: {str(e)}")
      raise

if __name__ == "__main__":
    asyncio.run(main())


