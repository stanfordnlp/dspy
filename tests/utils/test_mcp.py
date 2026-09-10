import asyncio
import importlib
import json
import sys
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pytest

import dspy
from dspy import Tool
from dspy.dsp.utils.utils import dotdict
from dspy.utils.mcp import _convert_mcp_tool_result, convert_mcp_tool

if importlib.util.find_spec("mcp") is None:
    pytest.skip(reason="mcp is not installed", allow_module_level=True)

_UNSET = object()


def make_call_tool_result(field_style, texts=(), structured=_UNSET, is_error=False):
    from mcp.types import TextContent

    fields = {
        "content": [TextContent(type="text", text=text) for text in texts],
        "is_error" if field_style == "snake" else "isError": is_error,
    }
    if structured is not _UNSET:
        fields["structured_content" if field_style == "snake" else "structuredContent"] = structured
    return SimpleNamespace(**fields)


@pytest.mark.extra
def test_convert_mcp_tool_result_supports_both_field_styles_without_changing_results():
    camel_result = make_call_tool_result("camel", texts=["hi"], structured={"result": "ignored"})
    snake_result = make_call_tool_result("snake", texts=["a", "b"], structured={"result": "ignored"})

    assert _convert_mcp_tool_result(camel_result) == "hi"
    assert _convert_mcp_tool_result(snake_result) == ["a", "b"]


@pytest.mark.extra
@pytest.mark.parametrize("field_style", ["camel", "snake"])
@pytest.mark.parametrize(
    "structured",
    [
        pytest.param({"answer": 42}, id="object"),
        pytest.param([1, 2], id="array"),
        pytest.param("answer", id="string"),
        pytest.param(3.5, id="number"),
        pytest.param(False, id="boolean"),
        pytest.param(None, id="null"),
        pytest.param({}, id="empty-object"),
        pytest.param([], id="empty-array"),
        pytest.param("", id="empty-string"),
        pytest.param(0, id="zero"),
    ],
)
def test_structured_result_mode_returns_every_json_value_exactly(field_style, structured):
    result = make_call_tool_result(field_style, texts=["fallback"], structured=structured)

    converted = _convert_mcp_tool_result(result, result_mode="structured")
    assert converted == structured
    assert type(converted) is type(structured)


@pytest.mark.extra
@pytest.mark.parametrize("field_style", ["camel", "snake"])
def test_structured_result_mode_falls_back_when_field_is_absent(field_style):
    result = make_call_tool_result(field_style, texts=["fallback"])

    assert _convert_mcp_tool_result(result, result_mode="structured") == "fallback"


@pytest.mark.extra
def test_structured_result_mode_distinguishes_explicit_null_from_omission():
    from mcp.types import CallToolResult

    model_fields = getattr(CallToolResult, "model_fields", None)
    if model_fields is None:
        model_fields = getattr(CallToolResult, "__fields__", {})
    field_name = next(
        (name for name in ("structured_content", "structuredContent") if name in model_fields),
        None,
    )
    if field_name is None:
        pytest.skip("This MCP version predates structured content")

    omitted = CallToolResult(content=[])
    explicit_null = CallToolResult(content=[], **{field_name: None})

    assert _convert_mcp_tool_result(omitted, result_mode="structured") == []
    assert _convert_mcp_tool_result(explicit_null, result_mode="structured") is None


@pytest.mark.extra
@pytest.mark.parametrize("field_style", ["camel", "snake"])
@pytest.mark.parametrize("result_mode", ["text", "structured"])
def test_error_result_raises_before_conversion(field_style, result_mode):
    result = make_call_tool_result(field_style, texts=["boom"], structured={"result": "ignored"}, is_error=True)
    with pytest.raises(RuntimeError, match="Failed to call a MCP tool: boom"):
        _convert_mcp_tool_result(result, result_mode=result_mode)


@pytest.mark.extra
def test_convert_mcp_tool_rejects_unknown_result_mode():
    tool = SimpleNamespace(name="test", description="test", input_schema={})

    with pytest.raises(ValueError, match="Unsupported MCP result mode: 'invalid'"):
        convert_mcp_tool(SimpleNamespace(), tool, result_mode="invalid")


@pytest.mark.asyncio
@pytest.mark.extra
async def test_convert_mcp_tool_with_v2_client():
    if int(version("mcp").split(".")[0]) < 2:
        pytest.skip("The high-level MCP client is available in SDK v2")

    from mcp.client import Client
    from mcp.server import MCPServer

    server = MCPServer("test")

    @server.tool()
    def increment(value: int) -> int:
        return value + 1

    async with Client(server) as client:
        response = await client.list_tools()
        increment_tool = Tool.from_mcp_tool(client, response.tools[0])
        structured_increment_tool = Tool.from_mcp_tool(client, response.tools[0], result_mode="structured")

        assert await increment_tool.acall(value=1) == "2"
        assert await structured_increment_tool.acall(value=1) == {"result": 2}


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

            # Check add
            add_tool = convert_mcp_tool(session, response.tools[0])
            assert add_tool.name == "add"
            assert add_tool.desc == "Add two numbers"
            assert add_tool.args == {
                "a": {"title": "A", "type": "integer"},
                "b": {"title": "B", "type": "integer"},
            }
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
            with pytest.raises(RuntimeError, match="error!"):
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


@pytest.mark.asyncio
@pytest.mark.extra
async def test_react_v2_native_mcp_end_to_end():
    """Exercise native messages and a real stdio MCP server without an API key."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    class NativeMCPLM(dspy.BaseLM):
        def __init__(self):
            super().__init__("native-mcp-test")
            self.requests = []

        @property
        def supports_function_calling(self):
            return True

        def forward(self, *args, **kwargs):
            pytest.fail("ReActV2 must use the async LM path")

        async def aforward(self, prompt=None, messages=None, **kwargs):
            self.requests.append({"messages": messages, "kwargs": kwargs})
            if len(self.requests) == 1:
                calls = [
                    ("mcp_error", "wrong_tool", {}),
                    ("mcp_add", "add", {"a": 25, "b": 17}),
                ]
            else:
                observations = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
                assert "error!" in observations["mcp_error"]
                assert observations["mcp_add"] == "42"
                calls = [("final", "submit", {"answer": int(observations["mcp_add"])})]
            return dotdict(
                choices=[
                    dotdict(
                        message=dotdict(
                            content=None,
                            tool_calls=[
                                dotdict(
                                    id=call_id, type="function", function=dotdict(name=name, arguments=json.dumps(args))
                                )
                                for call_id, name, args in calls
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                model=self.model,
            )

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).parent / "resources" / "mcp_server.py")],
    )
    lm = NativeMCPLM()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=5)
            response = await session.list_tools()
            tools = [Tool.from_mcp_tool(session, tool) for tool in response.tools]
            with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_native_function_calling=True)):
                pred = await asyncio.wait_for(
                    dspy.ReActV2("question -> answer: int", tools=tools).acall(question="What is 25 + 17?"),
                    timeout=15,
                )

    assert pred.answer == 42
    assert pred.termination_reason == "submit"
    assert len(lm.requests) == 2
    assert {t["function"]["name"] for t in lm.requests[0]["kwargs"]["tools"]} == {tool.name for tool in tools} | {
        "submit"
    }
    results = pred.history.messages[0]["tool_calls"].tool_call_results.tool_call_results
    assert [(r.call_id, r.is_error) for r in results] == [("mcp_error", True), ("mcp_add", False)]
    assert results[1].value == "42"


@pytest.mark.asyncio
@pytest.mark.extra
async def test_react_v2_mcp_cancellation_preserves_session(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    started, release = tmp_path / "started", tmp_path / "release"
    executed = []

    def later() -> str:
        executed.append("later")
        return "unexpected"

    async def wait_until_started():
        while not started.exists():
            await asyncio.sleep(0.01)

    lm = dspy.utils.DummyLM(
        [
            {
                "next_thought": "Wait.",
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [
                        {"name": "wait_for_release", "args": {"started": str(started), "release": str(release)}},
                        {"name": "later", "args": {}},
                        {"name": "submit", "args": {"answer": 999}},
                    ]
                ),
            },
            {
                "next_thought": "Add.",
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [
                        {"name": "add", "args": {"a": 23, "b": -6}},
                    ]
                ),
            },
            {
                "next_thought": "Done.",
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [
                        {"name": "submit", "args": {"answer": 17}},
                    ]
                ),
            },
        ]
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).parent / "resources" / "mcp_server.py")],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=5)
            tools = [Tool.from_mcp_tool(session, tool) for tool in (await session.list_tools()).tools]
            agent = dspy.ReActV2("question -> answer: int", tools=[*tools, later])
            with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_native_function_calling=False)):
                task = asyncio.create_task(agent.acall(question="Wait for release"))
                try:
                    # Cancel only after the remote tool has actually started.
                    await asyncio.wait_for(wait_until_started(), timeout=5)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(task, timeout=5)
                    assert executed == []
                    assert len(lm.history) == 1  # No continuation or forced-submit request.

                    # The agent borrows the session: cancellation must not close it.
                    pred = await asyncio.wait_for(agent.acall(question="What is 23 minus 6?"), timeout=5)
                    assert (pred.answer, pred.termination_reason) == (17, "submit")
                    result = pred.history.messages[0]["tool_calls"].tool_call_results.tool_call_results[0]
                    assert result.value == "17"
                    assert result.is_error is False
                finally:
                    # Local cancellation need not stop remote work. Release it explicitly.
                    release.touch()
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
