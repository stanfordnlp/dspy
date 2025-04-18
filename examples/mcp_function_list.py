"""
Clean example of using DSPy MCP with proper resource management
"""

import asyncio
import os
import sys
import dspy


# Add DSPy module to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "../dspy_dev"))

# Import from DSPy - use the refactored MCP module

# A simple testing function to use the MCP tools
async def test_tools(tools) -> None:
    """Test the available MCP tools by printing their names and descriptions."""
    if not tools:
        print("No tools available")
        return
        
    print(f"\nFound {len(tools)} tools:")
    for idx, tool in enumerate(tools, 1):
        print(f"{idx}. {tool.name}: {tool.desc}")

async def main():
    # Use the default silent mode (logging is already disabled in the module)
    
    # If you want logging, uncomment these lines:
    # setup_logging(log_level=logging.INFO, log_to_file=True, log_dir="./logs")
    
    # Define a configuration file path
    config_path = os.path.join(os.path.dirname(__file__), r"F:\AI\DSPy_MCP\dspy_dev\examples\servers_config.json")
    
    try:
        # Create and use the MCP server manager with proper async context
        async with dspy.MCPServerManager() as manager:
            # Load configuration and initialize servers
            config = manager.load_config(config_path)
            await manager.initialize_servers(config)
            
            # Get all available tools
            tools = await manager.get_all_tools()
            
            # Test the tools
            await test_tools(tools)
            
            # Resources will be automatically cleaned up when exiting the context
            
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        print("Please create a servers_config.json file with proper MCP server configuration.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())