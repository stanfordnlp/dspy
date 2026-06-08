"""
TWZRD Agent Intel MCP Server — trust-scoring for x402 agents in DSPy.

This example shows how to use the TWZRD Agent Intel MCP server with DSPy.
The server exposes two free tools:
  - score_agent(wallet)      — returns 0-100 trust score + risk flags
  - preflight_check(wallet)  — PASS/FAIL gate for x402 payment flows

And one paid tool (HTTP 402):
  - get_trust_receipt(wallet) — cryptographically signed trust receipt

MCP endpoint: https://intel.twzrd.xyz/mcp  (streamable-http, no auth required)

Install:
    pip install -U "dspy[mcp]"

Usage:
    python examples/twzrd_agent_intel_mcp.py
"""

import asyncio
import dspy
from dspy.utils.mcp import MCPClient


async def main():
    # Connect to TWZRD Agent Intel MCP server (streamable-http, no auth required)
    mcp_client = MCPClient(
        "twzrd-agent-intel",
        transport="streamable-http",
        url="https://intel.twzrd.xyz/mcp",
    )
    await mcp_client.connect()

    # List available tools
    tools = await mcp_client.list_tools()
    print("Available tools:", [t.name for t in tools])

    # Configure DSPy with your preferred LM
    # lm = dspy.LM("openai/gpt-4o-mini")  # or any DSPy-supported model
    # dspy.configure(lm=lm)

    # Example: Check trust score before sending an x402 payment
    wallet = "4LkEFjHsF2ubC8K4oF2r3rCFqPZQVGBjL9mV6xkNPZdf"  # example wallet

    # Call score_agent directly via MCP
    score_result = await mcp_client.call_tool(
        "score_agent",
        {"wallet": wallet},
    )
    print(f"\nscore_agent({wallet[:8]}...):")
    print(score_result)

    # Call preflight_check — PASS/FAIL gate
    preflight_result = await mcp_client.call_tool(
        "preflight_check",
        {"wallet": wallet},
    )
    print(f"\npreflight_check({wallet[:8]}...):")
    print(preflight_result)

    await mcp_client.disconnect()


# --- DSPy ReAct agent integration ---
class TrustAwareAgent(dspy.Signature):
    """Check agent trust scores before authorizing x402 payments.
    Use score_agent to get a 0-100 trust score, and preflight_check
    for a binary PASS/FAIL gate. Only proceed with payments for PASS agents."""

    wallet: str = dspy.InputField(desc="Solana wallet address of the agent to check")
    decision: str = dspy.OutputField(desc="APPROVE or REJECT with reasoning")


async def run_trust_aware_agent(wallet: str):
    """Run a DSPy ReAct agent that uses TWZRD tools to make trust decisions."""
    mcp_client = MCPClient(
        "twzrd-agent-intel",
        transport="streamable-http",
        url="https://intel.twzrd.xyz/mcp",
    )
    await mcp_client.connect()
    tools = await mcp_client.list_tools()

    # Build a ReAct agent with TWZRD trust tools
    agent = dspy.ReAct(TrustAwareAgent, tools=tools)

    result = await agent.acall(wallet=wallet)
    print(f"\nTrust decision for {wallet[:8]}...: {result.decision}")

    await mcp_client.disconnect()
    return result


if __name__ == "__main__":
    asyncio.run(main())
