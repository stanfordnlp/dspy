import asyncio
import importlib

import pytest

from dspy.utils.mcp import convert_mcp_tool, http_mcp_tools, stdio_mcp_tools

if importlib.util.find_spec("mcp") is None:
    pytest.skip(reason="mcp is not installed", allow_module_level=True)


@pytest.mark.asyncio
@pytest.mark.extra
async def test_convert_mcp_tool():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command="python",
        args=["tests/utils/resources/mcp_server.py"],
        env=None,
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=5)
            response = await session.list_tools()

            # Check add
            add_tool = convert_mcp_tool(session, response.tools[0])
            assert add_tool.name == "add"
            assert add_tool.desc == "Add two numbers"
            assert add_tool.args == {"a": {"title": "A", "type": "integer"}, "b": {"title": "B", "type": "integer"}}
            assert add_tool.arg_types == {"a": int, "b": int}
            assert add_tool.arg_desc == {
                "a": "No description provided. (Required)",
                "b": "No description provided. (Required)",
            }
            assert await add_tool.acall(a=1, b=2) == "3"

            # Check hello
            hello_tool = convert_mcp_tool(session, response.tools[1])
            assert hello_tool.name == "hello"
            assert hello_tool.desc == "Greet people"
            assert hello_tool.args == {"names": {"title": "Names", "type": "array", "items": {"type": "string"}}}
            assert hello_tool.arg_types == {"names": list}
            assert hello_tool.arg_desc == {"names": "No description provided. (Required)"}
            assert await hello_tool.acall(names=["Bob", "Tom"]) == ["Hello, Bob!", "Hello, Tom!"]

            # Check error handling
            error_tool = convert_mcp_tool(session, response.tools[2])
            assert error_tool.name == "wrong_tool"
            assert error_tool.desc == "This tool raises an error"
            with pytest.raises(
                RuntimeError, match="Failed to call a MCP tool: Error executing tool wrong_tool: error!"
            ):
                await error_tool.acall()

            # Check nested Pydantic arg
            nested_pydantic_tool = convert_mcp_tool(session, response.tools[3])

            assert nested_pydantic_tool.name == "get_account_name"
            assert nested_pydantic_tool.desc == "This extracts the name from account"
            assert nested_pydantic_tool.args == {
                "account": {
                    "title": "Account",
                    "type": "object",
                    "required": ["profile", "account_id"],
                    "properties": {
                        "profile": {
                            "title": "Profile",
                            "type": "object",
                            "properties": {
                                "name": {"title": "Name", "type": "string"},
                                "age": {"title": "Age", "type": "integer"},
                            },
                            "required": ["name", "age"],
                        },
                        "account_id": {"title": "Account Id", "type": "string"},
                    },
                }
            }
            account_in_json = {
                "profile": {
                    "name": "Bob",
                    "age": 20,
                },
                "account_id": "123",
            }
            result = await nested_pydantic_tool.acall(account=account_in_json)
            assert result == "Bob"

            # Check no input parameter current_datetime tool
            current_datetime_tool = convert_mcp_tool(session, response.tools[4])
            assert current_datetime_tool.name == "current_datetime"
            assert current_datetime_tool.desc == "Get the current datetime"
            assert current_datetime_tool.args == {}
            assert current_datetime_tool.arg_types == {}
            assert current_datetime_tool.arg_desc == {}
            assert await current_datetime_tool.acall() == "2025-07-23T09:10:10.0+00:00"


def _find_nested_exception(eg, exc_type):
    """Walk a (possibly nested) ExceptionGroup to find an exception of the given type."""
    if isinstance(eg, exc_type):
        return eg
    if isinstance(eg, BaseException) and hasattr(eg, "exceptions"):
        for exc in eg.exceptions:
            found = _find_nested_exception(exc, exc_type)
            if found:
                return found
    return None


@pytest.fixture()
def server_params():
    from mcp import StdioServerParameters

    return StdioServerParameters(
        command="python",
        args=["tests/utils/resources/mcp_server.py"],
        env=None,
    )


@pytest.mark.asyncio
@pytest.mark.extra
async def test_stdio_mcp_tools(server_params):
    # Basic usage: all tools available and callable
    async with stdio_mcp_tools(server_params) as tools:
        names = [t.name for t in tools]
        assert "add" in names
        assert "hello" in names
        add_tool = next(t for t in tools if t.name == "add")
        assert await add_tool.acall(a=1, b=2) == "3"

    # include_tools: only the specified tool is returned
    async with stdio_mcp_tools(server_params, include_tools=["add"]) as tools:
        assert len(tools) == 1
        assert tools[0].name == "add"

    # exclude_tools: the specified tool is filtered out
    async with stdio_mcp_tools(server_params, exclude_tools=["wrong_tool"]) as tools:
        names = [t.name for t in tools]
        assert "wrong_tool" not in names
        assert "add" in names

    # include_tools and exclude_tools together raises ValueError
    with pytest.raises(ValueError, match="Cannot specify both include_tools and exclude_tools"):
        async with stdio_mcp_tools(server_params, include_tools=["add"], exclude_tools=["hello"]) as _:
            pass

    # timeout: session works with a timeout set
    async with stdio_mcp_tools(server_params, timeout=10) as tools:
        assert len(tools) > 0

    # include_tools with unknown name raises ValueError
    try:
        async with stdio_mcp_tools(server_params, include_tools=["nonexistent"]) as _:
            pass
        pytest.fail("Expected ValueError")
    except BaseException as eg:
        err = _find_nested_exception(eg, ValueError)
        assert err and "Tool names not found on MCP server" in str(err)

    # exclude_tools with unknown name raises ValueError
    # NOTE: Currently unknown exclude_tools names raise ValueError.
    # This may be too strict — consider using warnings.warn instead so callers
    # can maintain a static exclusion list without needing to keep it in sync
    # with the server's exact tool set.
    try:
        async with stdio_mcp_tools(server_params, exclude_tools=["nonexistent"]) as _:
            pass
        pytest.fail("Expected ValueError")
    except BaseException as eg:
        err = _find_nested_exception(eg, ValueError)
        assert err and "Tool names not found on MCP server" in str(err)


@pytest.fixture()
def mcp_proxy_url():
    """Start the MCP server with streamable-http transport and yield the URL."""
    import socket
    import subprocess
    import sys
    import time

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    url = f"http://127.0.0.1:{port}/mcp"
    proc = subprocess.Popen(
        [sys.executable, "tests/utils/resources/mcp_server.py", "--transport", "streamable-http", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        for _ in range(50):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise TimeoutError(f"MCP server did not start on port {port}")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=5)


@pytest.mark.asyncio
@pytest.mark.extra
async def test_http_mcp_tools(mcp_proxy_url):
    # Basic usage: all tools available and callable
    async with http_mcp_tools(mcp_proxy_url) as tools:
        names = [t.name for t in tools]
        assert "add" in names
        assert "hello" in names
        add_tool = next(t for t in tools if t.name == "add")
        assert await add_tool.acall(a=1, b=2) == "3"

    # include_tools: only the specified tool is returned
    async with http_mcp_tools(mcp_proxy_url, include_tools=["add"]) as tools:
        assert len(tools) == 1
        assert tools[0].name == "add"

    # exclude_tools: the specified tool is filtered out
    async with http_mcp_tools(mcp_proxy_url, exclude_tools=["wrong_tool"]) as tools:
        names = [t.name for t in tools]
        assert "wrong_tool" not in names
        assert "add" in names

    # include_tools with unknown name raises ValueError
    try:
        async with http_mcp_tools(mcp_proxy_url, include_tools=["nonexistent"]) as _:
            pass
        pytest.fail("Expected ValueError")
    except BaseException as eg:
        err = _find_nested_exception(eg, ValueError)
        assert err and "Tool names not found on MCP server" in str(err)

    # exclude_tools with unknown name raises ValueError
    # NOTE: Currently unknown exclude_tools names raise ValueError.
    # This may be too strict — consider using warnings.warn instead so callers
    # can maintain a static exclusion list without needing to keep it in sync
    # with the server's exact tool set.
    try:
        async with http_mcp_tools(mcp_proxy_url, exclude_tools=["nonexistent"]) as _:
            pass
        pytest.fail("Expected ValueError")
    except BaseException as eg:
        err = _find_nested_exception(eg, ValueError)
        assert err and "Tool names not found on MCP server" in str(err)
