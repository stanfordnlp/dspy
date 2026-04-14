"""GoodMem + DSPy RAG Pipeline Example.

Demonstrates how to build a retrieval-augmented generation (RAG) system
using GoodMem as the vector memory backend and DSPy for the LM pipeline.

The example walks through every step:

    1. Connect to GoodMem and discover embedders
    2. Create a space and ingest documents as memories
    3. Build a DSPy RAG module (GoodMemRM + ChainOfThought)
    4. Ask questions and inspect the grounded answers
    5. Evaluate answer quality with SemanticF1
    6. Clean up (delete memories and space)

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
    python -m dspy.utils.goodmem.example

    # Or, to skip SSL verification (e.g. localhost with self-signed cert):
    export GOODMEM_VERIFY_SSL=false
    python -m dspy.utils.goodmem.example

Note:
    This example uses ``openai/gpt-5-mini``, but any LiteLLM-supported provider
    works (Anthropic, Gemini, Ollama, Databricks, etc.).  Swap the model string in
    the ``dspy.LM()`` call inside ``main()`` and set the provider's API key instead.
"""

from __future__ import annotations

import os
import sys
import time

import dspy
from dspy.evaluate import SemanticF1
from dspy.retrievers.goodmem_rm import GoodMemRM
from dspy.utils.goodmem.client import GoodMemClient

# =========================================================================
# Configuration
# =========================================================================

def get_config() -> dict:
    """Read configuration from environment variables.

    Required:
        GOODMEM_API_KEY  -- Your GoodMem API key (e.g. "gm_...")
        GOODMEM_BASE_URL -- GoodMem server URL (e.g. "https://localhost:8080")
        OPENAI_API_KEY   -- OpenAI API key for the language model

    Optional:
        GOODMEM_VERIFY_SSL -- Set to "false" to skip TLS verification
                              (useful for localhost with self-signed certs)
    """
    api_key = os.environ.get("GOODMEM_API_KEY")
    base_url = os.environ.get("GOODMEM_BASE_URL")

    if not api_key:
        sys.exit("Error: GOODMEM_API_KEY environment variable is not set.")
    if not base_url:
        sys.exit("Error: GOODMEM_BASE_URL environment variable is not set.")

    verify_ssl = os.environ.get("GOODMEM_VERIFY_SSL", "true").lower() != "false"

    return {
        "api_key": api_key,
        "base_url": base_url,
        "verify_ssl": verify_ssl,
    }


# =========================================================================
# Sample Documents
# =========================================================================

# A small fictional knowledge base about "Acme Corp".  Each string becomes
# one GoodMem memory.  GoodMem will chunk, embed, and index these so they
# can be retrieved later via semantic search.

DOCUMENTS = [
    (
        "Acme Corp was founded in 2019 by Jane Smith and Carlos Rivera in "
        "Austin, Texas. The company started as a two-person garage project "
        "focused on making databases easier to operate at scale."
    ),
    (
        "Acme Corp's flagship product is AcmeDB, a cloud-native distributed "
        "database designed for real-time analytics. AcmeDB supports ACID "
        "transactions, horizontal sharding, and automatic failover across "
        "multiple availability zones."
    ),
    (
        "In March 2023, Acme Corp raised $50 million in Series B funding led "
        "by Benchmark Capital, with participation from Sequoia and Y Combinator "
        "Continuity. The round valued the company at $400 million."
    ),
    (
        "AcmeDB's free tier supports up to 10 GB of storage, 1 million reads "
        "per month, and 100,000 writes per month. Paid plans start at $29/month "
        "for the Starter tier and $199/month for the Professional tier."
    ),
    (
        "Acme Corp is headquartered in Austin, Texas, with remote engineering "
        "offices in Berlin, Germany and Bangalore, India. The company employs "
        "approximately 120 people across all locations as of early 2024."
    ),
    (
        "AcmeDB uses a custom query language called AQL (Acme Query Language) "
        "that is compatible with a subset of SQL. AQL adds native support for "
        "time-series data, JSON document queries, and graph traversals within "
        "a single unified syntax."
    ),
]


# =========================================================================
# Step 1: Set Up GoodMem (Space & Memories)
# =========================================================================

def setup_goodmem(client: GoodMemClient) -> tuple[str, list[str]]:
    """Create a GoodMem space and ingest the sample documents.

    This function:
      - Discovers available embedders on the GoodMem server
      - Creates a new space (or reuses one with the same name)
      - Stores each document as a separate memory

    Args:
        client: An initialised GoodMemClient.

    Returns:
        A (space_id, memory_ids) tuple for later cleanup.
    """
    # --- Discover embedders ---
    # An embedder is a model that converts text into vector embeddings.
    # GoodMem needs at least one registered embedder to create a space.
    print("  Discovering embedders...")
    embedders = client.list_embedders()
    if not embedders:
        sys.exit(
            "Error: No embedders registered on the GoodMem server.\n"
            "Register one first — see https://docs.goodmem.ai"
        )

    embedder = embedders[0]
    embedder_id = embedder["embedderId"]
    print(f"  Using embedder: {embedder.get('displayName', embedder_id)}")

    # --- Create a space ---
    # A space is a logical container for related memories, configured with
    # a specific embedder.  create_space() is idempotent: if a space with
    # the same name already exists, it returns the existing space.
    space_name = "dspy-goodmem-example"
    print(f"  Creating space '{space_name}'...")
    result = client.create_space(space_name, embedder_id)
    space_id = result["spaceId"]
    reused = result.get("reused", False)
    print(f"  Space ID: {space_id} ({'reused' if reused else 'created'})")

    # --- Ingest documents ---
    # Each document is stored as a memory.  GoodMem automatically chunks
    # the text, generates vector embeddings, and indexes them for search.
    print(f"  Ingesting {len(DOCUMENTS)} documents...")
    memory_ids: list[str] = []
    for i, doc in enumerate(DOCUMENTS):
        mem = client.create_memory(
            space_id,
            text_content=doc,
            source="acme-knowledge-base",
            tags=f"example,doc-{i}",
        )
        memory_ids.append(mem["memoryId"])
        print(f"    [{i + 1}/{len(DOCUMENTS)}] memoryId={mem['memoryId']}")

    # --- Wait for indexing ---
    # Memories are processed asynchronously.  We wait a few seconds so
    # the embeddings are ready before we try to retrieve.
    print("  Waiting for indexing to complete...")
    time.sleep(5)

    return space_id, memory_ids


# =========================================================================
# Step 2: Build the RAG Module
# =========================================================================

class RAG(dspy.Module):
    """A retrieval-augmented generation module powered by GoodMem.

    This is a standard DSPy module that:
      1. Retrieves relevant passages from GoodMem using semantic search
      2. Feeds them as context to a language model with chain-of-thought
         reasoning to produce a grounded answer

    DSPy concepts used:
      - **dspy.Module**: Base class for composable AI programs
      - **dspy.ChainOfThought**: A predict module that asks the LM to
        show its reasoning step-by-step before giving a final answer
      - **Signature** ``"context, question -> answer"``: Declares what the
        LM should do without prescribing *how* to prompt it.  DSPy
        automatically expands this into a full prompt at runtime.
    """

    def __init__(self, retriever: GoodMemRM):
        super().__init__()
        self.retriever = retriever
        # ChainOfThought wraps a signature and adds a "reasoning" field.
        # The LM will produce reasoning first, then the answer.
        self.respond = dspy.ChainOfThought("context, question -> response")

    def forward(self, question: str) -> dspy.Prediction:
        # Step 1: Retrieve relevant passages from GoodMem.
        # GoodMemRM.forward() returns a list of dotdict({"long_text": ...})
        # objects — the same format used by all DSPy retrievers.
        passages = self.retriever(question)

        # Step 2: Combine passages into a single context string.
        context = "\n\n".join(p["long_text"] for p in passages)

        # Step 3: Feed context + question to the LM with chain-of-thought.
        # The LM will reason through the passages and produce an answer.
        return self.respond(context=context, question=question)


# =========================================================================
# Step 3: Run Inference
# =========================================================================

def run_inference(rag: RAG) -> None:
    """Ask a few questions and display the RAG pipeline's answers.

    Each answer shows:
      - The question asked
      - The chain-of-thought reasoning (how the LM arrived at the answer)
      - The final answer
    """
    questions = [
        "When was Acme Corp founded and by whom?",
        "How much funding did Acme Corp raise in Series B?",
        "What are the storage limits on AcmeDB's free tier?",
        "Where is Acme Corp headquartered and how many employees does it have?",
    ]

    for i, question in enumerate(questions):
        print(f"\n  Q{i + 1}: {question}")
        result = rag(question=question)
        # dspy.ChainOfThought adds a 'reasoning' field to the output.
        print(f"  Reasoning: {result.reasoning}")
        print(f"  Answer: {result.response}")
        print(f"  {'- ' * 30}")


# =========================================================================
# Step 4: Evaluate Quality
# =========================================================================

def evaluate_quality(rag: RAG) -> float:
    """Measure answer quality using DSPy's SemanticF1 metric.

    DSPy concepts used:
      - **dspy.Example**: A labeled data point with input fields and gold
        outputs, used for evaluation (and optionally for optimization).
      - **SemanticF1**: A metric that uses an LM to judge how well the
        predicted answer captures the meaning of the gold answer —
        more robust than exact-match for free-form text.
      - **dspy.Evaluate**: Runs a module over a dataset, computes the
        metric for each example, and reports aggregate scores.
    """
    # Define a small evaluation set with gold-standard answers.
    # In a real project, you'd have dozens or hundreds of these.
    devset = [
        dspy.Example(
            question="When was Acme Corp founded?",
            response="Acme Corp was founded in 2019.",
        ).with_inputs("question"),
        dspy.Example(
            question="Who founded Acme Corp?",
            response="Jane Smith and Carlos Rivera founded Acme Corp.",
        ).with_inputs("question"),
        dspy.Example(
            question="How much did Acme Corp raise in Series B?",
            response="Acme Corp raised $50 million in Series B funding.",
        ).with_inputs("question"),
        dspy.Example(
            question="What is AcmeDB's free tier storage limit?",
            response="AcmeDB's free tier supports up to 10 GB of storage.",
        ).with_inputs("question"),
        dspy.Example(
            question="What query language does AcmeDB use?",
            response="AcmeDB uses AQL (Acme Query Language), which is compatible with a subset of SQL.",
        ).with_inputs("question"),
    ]

    # SemanticF1 breaks the predicted and gold answers into atomic facts
    # and measures overlap — like F1, but at the semantic level.
    metric = SemanticF1(decompositional=True)

    # Evaluate the RAG module on the dev set.
    evaluator = dspy.Evaluate(
        devset=devset,
        metric=metric,
        num_threads=1,
        display_progress=True,
        display_table=True,
    )
    result = evaluator(rag)
    return result.score


# =========================================================================
# Step 5: Cleanup
# =========================================================================

def cleanup(client: GoodMemClient, space_id: str, memory_ids: list[str]) -> None:
    """Delete all test memories and the space created by this example."""
    print(f"  Deleting {len(memory_ids)} memories...")
    for mid in memory_ids:
        try:
            client.delete_memory(mid)
        except Exception:
            pass  # Best-effort cleanup

    print(f"  Deleting space {space_id}...")
    try:
        client.delete_space(space_id)
    except Exception:
        pass

    print("  Cleanup complete.")


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    """Run the full GoodMem + DSPy RAG example end-to-end."""

    print("=" * 60)
    print("  GoodMem + DSPy RAG Pipeline Example")
    print("=" * 60)

    # --- Configuration ---
    config = get_config()

    # Suppress SSL warnings when using self-signed certs.
    if not config["verify_ssl"]:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Configure the language model.
    # DSPy uses LiteLLM under the hood, so any provider works:
    #   dspy.LM("anthropic/claude-sonnet-4-20250514")
    #   dspy.LM("ollama_chat/llama3.2")
    #   dspy.LM("databricks/databricks-meta-llama-3-1-70b-instruct")
    lm = dspy.LM("openai/gpt-5-mini")
    dspy.configure(lm=lm)
    print("\n  LM: openai/gpt-5-mini")

    # Create a GoodMem client.
    client = GoodMemClient(
        api_key=config["api_key"],
        base_url=config["base_url"],
        verify_ssl=config["verify_ssl"],
    )
    print(f"  GoodMem: {config['base_url']}")

    # --- Step 1: Set up GoodMem ---
    print(f"\n{'- ' * 30}")
    print("  Step 1: Setting up GoodMem (space + memories)")
    print(f"{'- ' * 30}")
    space_id, memory_ids = setup_goodmem(client)

    try:
        # --- Step 2: Build the RAG module ---
        print(f"\n{'- ' * 30}")
        print("  Step 2: Building RAG module")
        print(f"{'- ' * 30}")

        # GoodMemRM is a DSPy retriever that searches GoodMem spaces.
        # It returns passages in the dotdict format that all DSPy
        # retrievers use, so it plugs into any DSPy pipeline.
        retriever = GoodMemRM(
            space_ids=[space_id],
            api_key=config["api_key"],
            base_url=config["base_url"],
            k=3,  # retrieve top-3 passages per query
            verify_ssl=config["verify_ssl"],
        )
        rag = RAG(retriever=retriever)
        print("  RAG module ready (GoodMemRM + ChainOfThought)")

        # --- Step 3: Run inference ---
        print(f"\n{'- ' * 30}")
        print("  Step 3: Running inference")
        print(f"{'- ' * 30}")
        run_inference(rag)

        # --- Step 4: Evaluate quality ---
        print(f"\n{'- ' * 30}")
        print("  Step 4: Evaluating answer quality")
        print(f"{'- ' * 30}")
        score = evaluate_quality(rag)
        print(f"\n  SemanticF1 score: {score:.1f}%")

    finally:
        # --- Step 5: Cleanup ---
        print(f"\n{'- ' * 30}")
        print("  Step 5: Cleaning up")
        print(f"{'- ' * 30}")
        cleanup(client, space_id, memory_ids)

    print(f"\n{'=' * 60}")
    print("  Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
