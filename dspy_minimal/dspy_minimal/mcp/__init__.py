"""MCP (Model Context Protocol) integration for DSPy minimal implementation."""

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    # Fallback implementations if MCP is not available
    class ClientSession:
        """Fallback ClientSession if MCP is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("MCP package is required for ClientSession. Install with: pip install mcp")
    
    class StdioServerParameters:
        """Fallback StdioServerParameters if MCP is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("MCP package is required for StdioServerParameters. Install with: pip install mcp")
    
    def stdio_client(*args, **kwargs):
        """Fallback stdio_client if MCP is not available."""
        raise ImportError("MCP package is required for stdio_client. Install with: pip install mcp")

__all__ = ["ClientSession", "StdioServerParameters", "stdio_client"] 