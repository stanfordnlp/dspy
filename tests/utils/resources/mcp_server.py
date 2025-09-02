from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP("test")


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
def hello(names: list[str]) -> str:
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


if __name__ == "__main__":
    mcp.run()
