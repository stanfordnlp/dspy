"""Apple on-device language models with DSPy.

Demonstrates three usage patterns:

1. Basic generation with AppleFoundationLM (Apple Intelligence system model)
2. Basic generation with AppleLocalLM (mlx-lm, any HuggingFace model)
3. Mixed-LM pipeline: on-device preprocessing → cloud reasoning

Requirements
------------
For pattern 1 (AppleFoundationLM):
    pip install apple-fm-sdk
    macOS 26+ with Apple Intelligence enabled

For pattern 2 & 3 (AppleLocalLM):
    pip install mlx-lm
    macOS 26+ on Apple Silicon (M1/M2/M3/M4)

For pattern 3 (cloud step):
    export ANTHROPIC_API_KEY=your_key
"""

import dspy


# ---------------------------------------------------------------------------
# Pattern 1: Apple Intelligence system model
# ---------------------------------------------------------------------------
# AppleFoundationLM wraps Apple's on-device Foundation Models SDK.
# No API key required, no network call, data never leaves the device.


def demo_apple_foundation_lm():
    lm = dspy.AppleFoundationLM()
    dspy.configure(lm=lm)

    qa = dspy.Predict("question -> answer")
    result = qa(question="What are the main benefits of on-device AI?")
    print("AppleFoundationLM answer:", result.answer)


# ---------------------------------------------------------------------------
# Pattern 2: Local model via mlx-lm
# ---------------------------------------------------------------------------
# AppleLocalLM loads any mlx-lm-compatible model from HuggingFace or a local
# directory and runs inference entirely on Apple Silicon.
# Hundreds of pre-quantized models are at https://huggingface.co/mlx-community


def demo_apple_local_lm():
    lm = dspy.AppleLocalLM(
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        bits=4,
        temperature=0.0,
        max_tokens=256,
    )
    dspy.configure(lm=lm)

    qa = dspy.Predict("question -> answer")
    result = qa(question="Explain DSPy in one sentence.")
    print("AppleLocalLM answer:", result.answer)


# ---------------------------------------------------------------------------
# Pattern 3: Mixed-LM pipeline
# ---------------------------------------------------------------------------
# Use an on-device model for cheap, private preprocessing (entity extraction,
# normalization, schema conversion) and a cloud model for the expensive
# reasoning step.  DSPy's per-module lm= override makes the routing explicit.


class NewsAnalysisPipeline(dspy.Module):
    """Extract structured fields on-device, then reason in the cloud.

    The on-device stage is free, private, and handles the bulk of the text
    processing.  The cloud stage receives a compact, structured payload — so
    you're billing tokens for reasoning, not repetition of raw text.
    """

    def __init__(self, local_lm: dspy.BaseLM, cloud_lm: dspy.BaseLM):
        # Extraction runs on-device: zero cost, zero privacy exposure.
        self.extract = dspy.Predict(
            "article -> tickers, headline_date, sentiment_passages",
            lm=local_lm,
        )
        # Sentiment scoring runs in the cloud with the clean extracted payload.
        self.score = dspy.Predict(
            "tickers, headline_date, sentiment_passages -> sentiment, confidence",
            lm=cloud_lm,
        )

    def forward(self, article: str) -> dspy.Prediction:
        extracted = self.extract(article=article)
        return self.score(
            tickers=extracted.tickers,
            headline_date=extracted.headline_date,
            sentiment_passages=extracted.sentiment_passages,
        )


def demo_mixed_pipeline():
    local_lm = dspy.AppleLocalLM(
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        temperature=0.0,
        max_tokens=256,
    )
    cloud_lm = dspy.LM("anthropic/claude-sonnet-4-6")

    pipeline = NewsAnalysisPipeline(local_lm=local_lm, cloud_lm=cloud_lm)

    article = (
        "Apple Inc. reported record quarterly revenue of $124.3 billion on Tuesday, "
        "driven by strong iPhone 17 sales and Services growth. CEO Tim Cook said the "
        "company sees 'enormous opportunity' in AI-powered features. AAPL shares rose "
        "3.2% in after-hours trading."
    )

    result = pipeline(article=article)
    print("Tickers found (on-device):", result.tickers)
    print("Sentiment (cloud):", result.sentiment)
    print("Confidence (cloud):", result.confidence)


# ---------------------------------------------------------------------------
# Pattern 4: Streaming with AppleLocalLM
# ---------------------------------------------------------------------------
# dspy.streamify() wraps any DSPy module so it yields tokens incrementally.
# AppleLocalLM sends _LocalStreamChunk objects for each token; the final
# dspy.Prediction is yielded last with all output fields parsed.


def demo_streaming():
    import asyncio

    from dspy.clients.apple_local import _LocalStreamChunk

    lm = dspy.AppleLocalLM(
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        temperature=0.0,
        max_tokens=256,
        cache=False,
    )
    dspy.configure(lm=lm)

    prog = dspy.streamify(dspy.Predict("question -> answer"))

    async def run():
        print("Streaming: ", end="", flush=True)
        async for chunk in prog(question="Explain what DSPy is in two sentences."):
            if isinstance(chunk, _LocalStreamChunk):
                print(chunk.text, end="", flush=True)
            elif isinstance(chunk, dspy.Prediction):
                print(f"\n\nParsed answer: {chunk.answer}")

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Pattern 5: Chain of Thought with OpenAI gpt-oss-20b via mlx-lm
# ---------------------------------------------------------------------------
# Uses InferenceIllusionist/gpt-oss-20b-MLX-4bit, a community 4-bit MLX
# conversion of OpenAI's gpt-oss-20b (~10 GB unified memory required).
# dspy.ChainOfThought adds a "reasoning" field before the answer, letting
# the model work through multi-step problems before committing to an answer.


def demo_chain_of_thought():
    lm = dspy.AppleLocalLM(
        "InferenceIllusionist/gpt-oss-20b-MLX-4bit",
        temperature=0.6,
        max_tokens=1024,
    )
    dspy.configure(lm=lm)

    cot = dspy.ChainOfThought("question -> answer")
    result = cot(
        question=(
            "A bat and a ball together cost $1.10. "
            "The bat costs $1.00 more than the ball. "
            "How much does the ball cost?"
        )
    )
    print("Reasoning:", result.reasoning)
    print("Answer:", result.answer)


# ---------------------------------------------------------------------------
# Structured output with AppleFoundationLM
# ---------------------------------------------------------------------------
# When response_format is a Pydantic model, AppleFoundationLM uses Apple's
# native @generable constrained decoding instead of injecting a JSON schema
# into the prompt.  This improves reliability for small on-device models.


def demo_structured_output():
    from typing import Literal

    from pydantic import BaseModel, Field

    class StockSignal(BaseModel):
        ticker: str
        action: Literal["buy", "hold", "sell"]
        confidence: int = Field(ge=1, le=10)

    lm = dspy.AppleFoundationLM()
    dspy.configure(lm=lm)

    extract = dspy.Predict("headline -> signal", response_format=StockSignal)
    result = extract(
        headline="Apple beats Q4 estimates; analysts raise price targets across the board."
    )
    # result.signal is parsed back into a StockSignal-compatible dict
    print("Structured signal:", result.signal)


if __name__ == "__main__":
    import sys

    demos = {
        "foundation": demo_apple_foundation_lm,
        "local": demo_apple_local_lm,
        "mixed": demo_mixed_pipeline,
        "structured": demo_structured_output,
        "streaming": demo_streaming,
        "cot": demo_chain_of_thought,
    }

    choice = sys.argv[1] if len(sys.argv) > 1 else "local"
    if choice not in demos:
        print(f"Usage: python {sys.argv[0]} [{' | '.join(demos)}]")
        sys.exit(1)

    demos[choice]()
