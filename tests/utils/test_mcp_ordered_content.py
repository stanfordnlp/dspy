"""Proof-of-concept regression checks for ordered MCP tool content.

The raw content mode is an opt-in programmatic contract, not a claim that
DSPy adapters natively send these objects to multimodal model endpoints.
"""

import pytest

from dspy.utils.mcp import _convert_mcp_tool_result, convert_mcp_tool


@pytest.mark.extra
def test_content_mode_preserves_interleaved_text_and_images() -> None:
    from mcp.types import CallToolResult, ImageContent, TextContent

    before = TextContent(type="text", text="before")
    image = ImageContent(type="image", data="aW1n", mimeType="image/png")
    after = TextContent(type="text", text="after")
    mixed = CallToolResult(content=[before, image, after])

    assert _convert_mcp_tool_result(mixed, result_mode="content") == [before, image, after]
    assert _convert_mcp_tool_result(mixed) == ["before", "after"]
    assert _convert_mcp_tool_result(CallToolResult(content=[before, after]), result_mode="content") == [before, after]


@pytest.mark.extra
def test_content_mode_preserves_non_text_only_and_empty_results() -> None:
    from mcp.types import CallToolResult, ImageContent

    image = ImageContent(type="image", data="aW1n", mimeType="image/png")
    assert _convert_mcp_tool_result(CallToolResult(content=[image]), result_mode="content") == [image]
    assert _convert_mcp_tool_result(CallToolResult(content=[]), result_mode="content") == []


@pytest.mark.extra
def test_content_mode_does_not_override_structured_or_errors() -> None:
    from mcp.types import CallToolResult, TextContent

    block = TextContent(type="text", text="evidence")
    result = CallToolResult(content=[block], structuredContent={"summary": 1})
    assert _convert_mcp_tool_result(result, result_mode="content") == [block]
    assert _convert_mcp_tool_result(result, result_mode="structured") == {"summary": 1}
    failed = CallToolResult(content=[block], isError=True)
    with pytest.raises(RuntimeError, match="Failed to call a MCP tool"):
        _convert_mcp_tool_result(failed, result_mode="content")


@pytest.mark.extra
def test_convert_mcp_tool_accepts_opt_in_content_mode() -> None:
    from types import SimpleNamespace

    tool = SimpleNamespace(name="example", description="example", input_schema={})
    converted = convert_mcp_tool(SimpleNamespace(), tool, result_mode="content")
    assert converted.name == "example"

@pytest.mark.asyncio
@pytest.mark.extra
async def test_content_mode_survives_dspy_tool_acall() -> None:
    from types import SimpleNamespace

    from mcp.types import CallToolResult, ImageContent, TextContent

    before = TextContent(type="text", text="before")
    image = ImageContent(type="image", data="aW1n", mimeType="image/png")
    after = TextContent(type="text", text="after")

    class MockMCPClient:
        async def call_tool(self, name, arguments):
            assert name == "ordered"
            assert arguments == {}
            return CallToolResult(content=[before, image, after])

    mcp_tool = SimpleNamespace(name="ordered", description="Test tool", input_schema={})
    tool = convert_mcp_tool(MockMCPClient(), mcp_tool, result_mode="content")

    assert await tool.acall() == [before, image, after]
