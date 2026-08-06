from pydantic import BaseModel

try:
    # mcp SDK v1
    from mcp.server.fastmcp import FastMCP as MCPServer
except ImportError:
    # mcp SDK v2
    from mcp.server import MCPServer

mcp = MCPServer("test")


class Profile(BaseModel):
    name: str
    age: int


class Account(BaseModel):
    profile: Profile
    account_id: str


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def hello(names: list[str]) -> list[str]:
    """Greet people"""
    return [f"Hello, {name}!" for name in names]


@mcp.tool()
def wrong_tool():
    """This tool raises an error"""
    raise ValueError("error!")


@mcp.tool()
def get_account_name(account: Account):
    """This extracts the name from account"""
    return account.profile.name


@mcp.tool()
def current_datetime() -> str:
    """Get the current datetime"""
    return "2025-07-23T09:10:10.0+00:00"


@mcp.tool()
def get_profile() -> Profile:
    """Get a user profile as structured output"""
    return Profile(name="Ann", age=30)


class SingleFieldResult(BaseModel):
    result: int


@mcp.tool()
def genuine_single_field() -> SingleFieldResult:
    """Returns an object whose only field is named result"""
    return SingleFieldResult(result=42)


if __name__ == "__main__":
    mcp.run()
