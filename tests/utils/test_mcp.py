import asyncio
import importlib
import sys
from types import SimpleNamespace

import pytest

from dspy.utils.mcp import _MCP_MAJOR_VERSION, _convert_mcp_tool_result, convert_mcp_tool

if importlib.util.find_spec("mcp") is None:
    pytest.skip(reason="mcp is not installed", allow_module_level=True)


def make_call_tool_result(texts=(), structured=None, is_error=False):
    from mcp.types import CallToolResult, TextContent

    kwargs = {"content": [TextContent(type="text", text=text) for text in texts]}
    if _MCP_MAJOR_VERSION >= 2:
        kwargs.update(structured_content=structured, is_error=is_error)
    else:
        kwargs.update(structuredContent=structured, isError=is_error)
    return CallToolResult(**kwargs)


WRAPPED_SCHEMA = {"type": "object", "properties": {"result": {"type": "integer"}}, "required": ["result"]}


@pytest.mark.extra
def test_convert_text_result():
    assert _convert_mcp_tool_result(make_call_tool_result(texts=["hi"])) == "hi"
    assert _convert_mcp_tool_result(make_call_tool_result(texts=["a", "b"])) == ["a", "b"]


@pytest.mark.extra
def test_structured_content_preferred_over_text():
    result = make_call_tool_result(texts=['{"name": "Ann"}'], structured={"name": "Ann"})
    assert _convert_mcp_tool_result(result) == {"name": "Ann"}


@pytest.mark.extra
def test_wrapped_structured_content_is_unwrapped():
    result = make_call_tool_result(texts=["3"], structured={"result": 3})
    assert _convert_mcp_tool_result(result, output_schema=WRAPPED_SCHEMA) == 3
    # Without text content there is no evidence against the envelope reading.
    result = make_call_tool_result(structured={"result": 3})
    assert _convert_mcp_tool_result(result, output_schema=WRAPPED_SCHEMA) == 3


@pytest.mark.extra
def test_genuine_single_result_field_object_is_not_unwrapped():
    # A tool that genuinely returns an object with one required field named "result"
    # has the same output schema as the SDK's wrapper envelope, but its text content
    # renders the full object rather than the inner value.
    result = make_call_tool_result(texts=['{"result": 42}'], structured={"result": 42})
    assert _convert_mcp_tool_result(result, output_schema=WRAPPED_SCHEMA) == {"result": 42}


@pytest.mark.extra
def test_structured_content_not_unwrapped_without_wrapper_schema():
    result = make_call_tool_result(texts=["3"], structured={"result": 3})
    assert _convert_mcp_tool_result(result) == {"result": 3}
    non_wrapper_schema = {"type": "object", "properties": {"result": {}, "other": {}}}
    assert _convert_mcp_tool_result(result, output_schema=non_wrapper_schema) == {"result": 3}


@pytest.mark.extra
def test_error_result_raises():
    with pytest.raises(RuntimeError, match="Failed to call a MCP tool: boom"):
        _convert_mcp_tool_result(make_call_tool_result(texts=["boom"], is_error=True))


@pytest.mark.extra
def test_input_required_result_raises():
    result = SimpleNamespace(result_type="input_required", input_requests=[])
    with pytest.raises(RuntimeError, match="needs additional input"):
        _convert_mcp_tool_result(result)


@pytest.mark.asyncio
@pytest.mark.extra
async def test_convert_mcp_tool():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["tests/utils/resources/mcp_server.py"],
        env=None,
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=5)
            response = await session.list_tools()
            tools = {tool.name: convert_mcp_tool(session, tool) for tool in response.tools}

            # Check add
            add_tool = tools["add"]
            assert add_tool.name == "add"
            assert add_tool.desc == "Add two numbers"
            assert add_tool.args["a"]["type"] == "integer"
            assert add_tool.args["b"]["type"] == "integer"
            assert add_tool.arg_types == {"a": int, "b": int}
            assert add_tool.arg_desc == {
                "a": "No description provided. (Required)",
                "b": "No description provided. (Required)",
            }
            # Typed return values surface as structured output, unwrapped to the plain value.
            assert await add_tool.acall(a=1, b=2) == 3

            # Check hello
            hello_tool = tools["hello"]
            assert hello_tool.name == "hello"
            assert hello_tool.desc == "Greet people"
            assert hello_tool.args["names"]["type"] == "array"
            assert hello_tool.arg_types == {"names": list}
            assert hello_tool.arg_desc == {"names": "No description provided. (Required)"}
            assert await hello_tool.acall(names=["Bob", "Tom"]) == ["Hello, Bob!", "Hello, Tom!"]

            # Check error handling
            error_tool = tools["wrong_tool"]
            assert error_tool.name == "wrong_tool"
            assert error_tool.desc == "This tool raises an error"
            with pytest.raises(RuntimeError, match="error!"):
                await error_tool.acall()

            # Check nested Pydantic arg
            nested_pydantic_tool = tools["get_account_name"]
            assert nested_pydantic_tool.name == "get_account_name"
            assert nested_pydantic_tool.desc == "This extracts the name from account"
            account_schema = nested_pydantic_tool.args["account"]
            assert account_schema["type"] == "object"
            assert set(account_schema["properties"]) == {"profile", "account_id"}
            assert set(account_schema["properties"]["profile"]["properties"]) == {"name", "age"}
            account_in_json = {
                "profile": {
                    "name": "Bob",
                    "age": 20,
                },
                "account_id": "123",
            }
            assert await nested_pydantic_tool.acall(account=account_in_json) == "Bob"

            # Check no input parameter current_datetime tool
            current_datetime_tool = tools["current_datetime"]
            assert current_datetime_tool.name == "current_datetime"
            assert current_datetime_tool.desc == "Get the current datetime"
            assert current_datetime_tool.args == {}
            assert current_datetime_tool.arg_types == {}
            assert current_datetime_tool.arg_desc == {}
            assert await current_datetime_tool.acall() == "2025-07-23T09:10:10.0+00:00"

            # Check structured output from a Pydantic return type
            profile_tool = tools["get_profile"]
            assert profile_tool.name == "get_profile"
            assert await profile_tool.acall() == {"name": "Ann", "age": 30}

            # An object whose only field is named "result" keeps its declared shape.
            single_field_tool = tools["genuine_single_field"]
            assert await single_field_tool.acall() == {"result": 42}
