import asyncio
import os
import dspy

# Define a simple DSPy Signature
class MultiServerSignature(dspy.Signature):
    """Perform operations using tools potentially available across multiple MCP servers."""
    request: str = dspy.InputField(desc="The user's request, potentially requiring external tools.")
    output: str = dspy.OutputField(desc="The final response to the user's request after potentially using tools.")

def print_message(message: str) -> None:
    """Print a message to the console."""
    print(message)

# --- Main Execution ---
async def main() -> None:
    """Initialize MCP servers, get tools, and run a DSPy agent."""
    lm = dspy.LM("gemini/gemini-2.0-flash",api_key=os.getenv("GOOGLE_API_KEY") )


    dspy.configure(lm=lm)
  
    # --- MCP Server Setup ---
    config_path = r"F:\AI\DSPy_MCP\dspy_dev\examples\servers_config.json" # Assumes config is in the same directory


    async with dspy.MCPServerManager() as server_manager:

        config = server_manager.load_config(config_path)
        #  or
        # config = {
        #             "mcpServers": {
        #                 "filesystem": {
        #                 "command": "npx",
        #                 "args": [
        #                     "-y",
        #                     "@modelcontextprotocol/server-filesystem",
        #                     "F:/AI/DSPy_MCP/test"
        #                 ]
        #                 },
        #                 "airbnb": {
        #                 "command": "npx",
        #                 "args": [
        #                     "-y",
        #                     "@openbnb/mcp-server-airbnb",
        #                     "--ignore-robots-txt"
        #                 ]
        #                 }
        #             }
        #         }
        await server_manager.initialize_servers(config)

        all_mcp_tools = await server_manager.get_all_tools()
        react_agent = dspy.ReAct(
            MultiServerSignature,
            tools=all_mcp_tools, # Pass the list of MCPTool instances
            max_iters=7 # Limit the number of steps
        )

        request = """Search hotel in airbnb in bangkok at 15/04/2025 - 20/04/2025 for 2 people 
        Write it's down to markdown format file. with name and price link"""

        result = await react_agent.async_forward(request=request)
        print("Final Result:", result.output)

if __name__ == "__main__":
    asyncio.run(main())
