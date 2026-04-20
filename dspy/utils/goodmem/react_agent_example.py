"""GoodMem + DSPy ReAct Agent Example.

Demonstrates how the GoodMem integration plugs into DSPy's ReAct
across four scenarios that highlight different agent capabilities:

    Scenario 1 -- Conversational memory agent
        A ReAct agent handles a sequence of turns. It stores facts
        the user shares via goodmem_create_memory, then retrieves
        them via goodmem_retrieve_memories when asked follow-up
        questions. Each turn is an independent ReAct call -- memory
        lives in GoodMem, not in the agent.

    Scenario 2 -- Cross-agent memory persistence
        A brand-new ReAct agent (no prior calls, no conversation
        history) answers questions about the user by querying
        GoodMem. Demonstrates the core value of an external memory
        store: memory outlives the agent instance.

    Scenario 3 -- Metadata-tagged memories
        Writes memories tagged with a 'category' field directly via
        GoodMemClient (deterministic), then has an agent list them
        and filter by category. Shows that structured metadata
        round-trips through GoodMem and is available to agents for
        downstream reasoning.

    Scenario 4 -- Trajectory inspection
        Prints the thought/action/observation trajectory ReAct
        produced. Useful for debugging agent behaviour and for
        proving a tool-using agent actually reached for the
        integration rather than answering from its weights.

Prerequisites:
    - A running GoodMem server (see https://docs.goodmem.ai)
    - An OpenAI API key (or any LiteLLM-supported provider)
    - At least one embedder registered on your GoodMem server

Usage::

    # 1. Set environment variables
    export OPENAI_API_KEY="sk-..."
    export GOODMEM_API_KEY="gm_..."
    export GOODMEM_BASE_URL="https://localhost:8080"

    # 2. Run the example
    python -m dspy.utils.goodmem.react_agent_example

Note:
    This example uses ``openai/gpt-5-mini``, but any LiteLLM-supported
    provider works (Anthropic, Gemini, Ollama, Databricks, etc.).  Swap
    the model string in the ``dspy.LM()`` call inside ``main()`` and
    set the provider's API key instead.

    Optional environment variable:

        GOODMEM_VERIFY_SSL
            Whether to verify the GoodMem server's TLS certificate.
            Defaults to "true". Set it to "false" only when
            connecting to a server that uses a self-signed
            certificate (e.g. a local dev instance on
            https://localhost). Keep it "true" in production.
"""

from __future__ import annotations

import json
import os
import sys
import time

import requests
import urllib3

import dspy
from dspy.utils.goodmem import GoodMemClient, make_goodmem_tools

# Load environment variables from a .env file at the repo root if one
# exists.  This is a convenience for running the example locally --
# export the vars in your shell and the file is simply ignored.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional; the example still works if env vars
    # are already set in the shell.
    pass

REQUIRED_ENV_VARS = [
    (
        "GOODMEM_API_KEY",
        "GoodMem API key (sent as X-API-Key).",
    ),
    (
        "GOODMEM_BASE_URL",
        "Base URL of your GoodMem server, e.g. https://localhost:8080.",
    ),
    (
        "OPENAI_API_KEY",
        "OpenAI API key used by the default dspy.LM. Swap the dspy.LM() "
        "call to use a different LiteLLM-supported provider.",
    ),
]


def check_env_vars() -> None:
    """Exit with a helpful message if required env vars are missing."""
    missing = [(name, desc) for name, desc in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if not missing:
        return

    lines = ["Error: missing required environment variables:", ""]
    for name, desc in missing:
        lines.append(f"  - {name}: {desc}")
    lines.extend(
        [
            "",
            "Set them before running, e.g.:",
            "",
            "  bash:",
            "    export GOODMEM_API_KEY='gm_...'",
            "    export GOODMEM_BASE_URL='https://localhost:8080'",
            "    export OPENAI_API_KEY='sk-...'",
            "",
            "  PowerShell:",
            "    $env:GOODMEM_API_KEY='gm_...'",
            "    $env:GOODMEM_BASE_URL='https://localhost:8080'",
            "    $env:OPENAI_API_KEY='sk-...'",
        ]
    )
    sys.exit("\n".join(lines))


# =========================================================================
# Configuration
# =========================================================================

SPACE_NAME = "dspy-goodmem-react-example"

SCENARIO_1_TURNS = [
    "I live in Austin, Texas.",
    "My favorite database is AcmeDB.",
    "What's my favorite database?",
    "And where do I live?",
]

SCENARIO_2_QUESTION = "Tell me everything you know about the user."

# Scenario 3 fixtures -- (content, category) pairs written directly
# via the client so the metadata is deterministic.  An agent then
# queries these and filters by the 'category' metadata field.
TAGGED_FACTS = [
    ("I work as a senior engineer at Acme Corp.", "work"),
    ("My manager is named Sarah.", "work"),
    ("I play guitar every Saturday morning.", "hobby"),
    ("I run 5 miles every Sunday.", "hobby"),
    ("I have a black cat named Luna.", "personal"),
]

SCENARIO_3_QUESTION = "Show me only the facts whose category is 'hobby'."


# =========================================================================
# Signatures
# =========================================================================


class MemoryAssistant(dspy.Signature):
    """You are a personal assistant with access to a semantic memory store via GoodMem tools.

    When the user shares a fact about themselves, call goodmem_create_memory to store it
    as a memory in the GoodMem space given by space_id.

    When the user asks a question about themselves, call goodmem_retrieve_memories to
    search that same space_id before answering.  Always call a GoodMem tool rather than
    relying on your own memory.
    """

    space_id: str = dspy.InputField(desc="GoodMem space ID to read from and write to.")
    user_message: str = dspy.InputField(desc="What the user said or asked.")
    assistant_response: str = dspy.OutputField(desc="Your grounded reply to the user.")


class MemoryAnalyst(dspy.Signature):
    """You are a knowledge analyst with access to GoodMem tools.

    The user's memories live in the GoodMem space given by space_id.  Each memory has a
    'category' field in its metadata (one of: 'work', 'hobby', 'personal').

    When the user asks about a specific category, call goodmem_list_memories to fetch
    every memory in the space and filter them by the metadata.category field yourself.
    Do not rely on semantic search alone.
    """

    space_id: str = dspy.InputField(desc="GoodMem space ID that holds the tagged memories.")
    user_message: str = dspy.InputField(desc="The analyst task from the user.")
    assistant_response: str = dspy.OutputField(desc="The filtered answer.")


# =========================================================================
# Helpers
# =========================================================================


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def subsection(title: str) -> None:
    print(f"\n{'- ' * 30}")
    print(f"  {title}")
    print(f"{'- ' * 30}")


def setup_space(client: GoodMemClient, space_name: str = SPACE_NAME) -> str:
    """Discover an embedder and create (or reuse) the demo space."""
    try:
        embedders = client.list_embedders()
    except requests.HTTPError as http_error:
        status = http_error.response.status_code if http_error.response is not None else None
        if status == 401:
            sys.exit(
                "Error: GoodMem rejected the request with 401 Unauthorized.\n"
                "Your GOODMEM_API_KEY is set but invalid or expired."
            )
        if status == 403:
            sys.exit(
                "Error: GoodMem rejected the request with 403 Forbidden.\n"
                "Your GOODMEM_API_KEY is valid but lacks permission for this operation."
            )
        raise
    except requests.exceptions.SSLError:
        sys.exit(
            "Error: TLS certificate verification failed for "
            f"{client.base_url}.\n"
            "If the server uses a self-signed cert, set GOODMEM_VERIFY_SSL=false."
        )
    except requests.ConnectionError:
        sys.exit(
            "Error: could not connect to the GoodMem server at "
            f"{client.base_url}.\n"
            "Verify GOODMEM_BASE_URL is correct and the server is running."
        )

    if not embedders:
        sys.exit(
            "Error: No embedders found on the GoodMem server.\n"
            "Register one first -- see https://docs.goodmem.ai"
        )
    embedder_id = embedders[0]["embedderId"]
    print(f"  Using embedder: {embedders[0].get('displayName', embedder_id)}")

    space_result = client.create_space(space_name, embedder_id)
    space_id = space_result["spaceId"]
    reused = space_result.get("reused", False)
    print(f"  Space '{space_name}' ({'reused' if reused else 'created'}): {space_id}")
    return space_id


def cleanup(client: GoodMemClient, space_ids: list[str]) -> None:
    """Best-effort cleanup: delete every memory in each space, then each space itself."""
    for space_id in space_ids:
        if not space_id:
            continue
        try:
            memories = client.list_memories(space_id)
        except Exception:
            memories = []

        print(f"  Space {space_id}: deleting {len(memories)} memories...")
        for memory in memories:
            memory_id = memory.get("memoryId") or memory.get("id")
            if not memory_id:
                continue
            try:
                client.delete_memory(memory_id)
            except Exception:
                pass  # Best-effort cleanup

        print(f"  Deleting space {space_id}...")
        try:
            client.delete_space(space_id)
        except Exception:
            pass  # Best-effort cleanup

    print("  Cleanup complete.")


# =========================================================================
# Scenarios
# =========================================================================


def scenario_1_conversational_agent(tools: list, space_id: str) -> None:
    """Run a sequence of turns through a ReAct memory assistant.

    Each turn is an independent ReAct call -- there's no built-in
    conversation history.  The point of this scenario is that the
    agent uses GoodMem tools to persist facts across calls, so by
    turn 3/4 it can still answer questions about turns 1/2.
    """
    section("Scenario 1: Conversational memory agent (multi-turn)")

    agent = dspy.ReAct(MemoryAssistant, tools=tools, max_iters=6)

    for turn_index, user_message in enumerate(SCENARIO_1_TURNS, start=1):
        print(f"\n  Turn {turn_index}")
        print(f"  User:  {user_message}")
        result = agent(space_id=space_id, user_message=user_message)
        print(f"  Agent: {result.assistant_response}")

        # Wait briefly after write turns so indexing catches up before
        # the next read turn.
        if turn_index in (1, 2):
            time.sleep(3)


def scenario_2_cross_agent_memory(tools: list, space_id: str) -> None:
    """Prove GoodMem outlives the agent instance.

    Spins up a brand-new ReAct agent and asks it what it knows about
    the user.  The only way it can answer is by calling
    goodmem_retrieve_memories against the shared space.
    """
    section("Scenario 2: Cross-agent memory persistence")
    print("  (Building a fresh ReAct agent with no prior calls.)")

    # A short wait lets any late-arriving indexing from Scenario 1
    # finish before the reader agent queries.
    time.sleep(3)

    reader_agent = dspy.ReAct(MemoryAssistant, tools=tools, max_iters=6)

    print(f"\n  User:  {SCENARIO_2_QUESTION}")
    result = reader_agent(space_id=space_id, user_message=SCENARIO_2_QUESTION)
    print(f"  Agent: {result.assistant_response}")


def scenario_3_metadata_filtering(
    client: GoodMemClient, tools: list
) -> tuple[str, dspy.Prediction]:
    """Demonstrate metadata-tagged memories.

    Writes five memories with a 'category' metadata field directly via
    the client (deterministic), then has a ReAct agent list them and
    filter by category.  Returns the tagged space ID (for cleanup) and
    the agent's prediction (so Scenario 4 can inspect it).
    """
    section("Scenario 3: Metadata-tagged memories and filtering")

    # Use a dedicated space so the tagged memories don't mix with
    # Scenario 1's untagged ones.
    tagged_space_name = f"{SPACE_NAME}-tagged"
    tagged_space_id = setup_space(client, tagged_space_name)

    # Write tagged memories directly.  Going through the client
    # (rather than the agent) keeps the metadata payload deterministic.
    print(f"\n  Ingesting {len(TAGGED_FACTS)} tagged memories...")
    for content, category in TAGGED_FACTS:
        client.create_memory(
            tagged_space_id,
            text_content=content,
            metadata={"category": category},
        )
        print(f"    [{category:>8}] {content}")

    # Let indexing catch up before the agent reads.
    print("  Waiting for indexing to complete...")
    time.sleep(5)

    analyst_agent = dspy.ReAct(MemoryAnalyst, tools=tools, max_iters=6)

    print(f"\n  User:  {SCENARIO_3_QUESTION}")
    result = analyst_agent(space_id=tagged_space_id, user_message=SCENARIO_3_QUESTION)
    print(f"  Agent: {result.assistant_response}")

    return tagged_space_id, result


def scenario_4_inspect_trajectory(result: dspy.Prediction) -> None:
    """Print the thought/action/observation trajectory ReAct produced.

    dspy.ReAct returns a Prediction with a ``trajectory`` dict whose
    keys are indexed by step (e.g. ``thought_0``, ``tool_name_0``,
    ``tool_args_0``, ``observation_0``).  Iterating over it is the
    most useful slice for proving the agent actually used GoodMem.
    """
    section("Scenario 4: Trajectory inspection")

    trajectory = getattr(result, "trajectory", None) or {}
    if not trajectory:
        print("\n  (No trajectory recorded on this prediction.)")
        return

    # Trajectory keys are suffixed with a step index (e.g. "thought_0").
    # Group them by step number for readable output.
    step_count = sum(1 for k in trajectory if k.startswith("thought_"))
    print(f"\n  ReAct steps: {step_count}")

    for i in range(step_count):
        thought = trajectory.get(f"thought_{i}", "")
        tool_name = trajectory.get(f"tool_name_{i}", "")
        tool_args = trajectory.get(f"tool_args_{i}", {})
        observation = trajectory.get(f"observation_{i}", "")

        observation_preview = str(observation)
        if len(observation_preview) > 120:
            observation_preview = observation_preview[:120] + "..."

        print(f"\n    Step {i + 1}")
        print(f"      Thought: {thought}")
        print(f"      Tool:    {tool_name}({json.dumps(tool_args)})")
        print(f"      Result:  {observation_preview}")


# =========================================================================
# Main
# =========================================================================


def main() -> None:
    """Run all four scenarios end-to-end."""
    check_env_vars()

    print("=" * 60)
    print("  GoodMem + DSPy ReAct Agent Example")
    print("=" * 60)

    # verify_ssl comes from GOODMEM_VERIFY_SSL; defaults to True so
    # the safe behaviour is the default in production.
    verify_ssl = os.environ.get("GOODMEM_VERIFY_SSL", "true").lower() != "false"
    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Configure the LM.  Any LiteLLM-supported provider works:
    #   dspy.LM("anthropic/claude-sonnet-4-20250514")
    #   dspy.LM("ollama_chat/llama3.2")
    #   dspy.LM("databricks/databricks-meta-llama-3-1-70b-instruct")
    lm = dspy.LM("openai/gpt-5-mini")
    dspy.configure(lm=lm)
    print("\n  LM: openai/gpt-5-mini")

    client = GoodMemClient(
        api_key=os.environ["GOODMEM_API_KEY"],
        base_url=os.environ["GOODMEM_BASE_URL"],
        verify_ssl=verify_ssl,
    )
    print(f"  GoodMem: {client.base_url}")

    # Build the tool list once -- the same 11 callables work for every
    # ReAct agent we instantiate in the scenarios.
    tools = [dspy.Tool(fn) for fn in make_goodmem_tools(client)]
    print(f"  Tools: {len(tools)} GoodMem callables wrapped as dspy.Tool")

    subsection("Setup: Discovering embedder and creating space")
    space_id = setup_space(client)
    tagged_space_id: str | None = None

    try:
        scenario_1_conversational_agent(tools, space_id)
        scenario_2_cross_agent_memory(tools, space_id)
        tagged_space_id, analyst_result = scenario_3_metadata_filtering(client, tools)
        scenario_4_inspect_trajectory(analyst_result)
    finally:
        subsection("Cleanup")
        spaces_to_clean = [space_id]
        if tagged_space_id:
            spaces_to_clean.append(tagged_space_id)
        cleanup(client, spaces_to_clean)

    print(f"\n{'=' * 60}")
    print("  Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
