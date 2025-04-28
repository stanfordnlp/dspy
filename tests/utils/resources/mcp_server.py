from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def hello(names: list[str]) -> str:
    """Greet people"""
    return [f"Hello, {name}!" for name in names]

@mcp.tool()
def wrong_tool():
    """This tool raises an error"""
    raise ValueError("error!")

if __name__ == "__main__":
    mcp.run()