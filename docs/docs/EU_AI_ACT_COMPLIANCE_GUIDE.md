# EU AI Act Compliance Guide for DSPy Deployers

DSPy is an Apache 2.0 framework. Frameworks are exempt under Article 25(4) of the EU AI Act: the obligations fall on deployers who build systems with them, not on the framework itself. If you are using DSPy to build something that touches Annex III categories (biometrics, critical infrastructure, employment, credit scoring, law enforcement, migration, education, insurance), this guide is for you.

The core problem DSPy creates for compliance is opacity by design. Teleprompters optimize your prompts automatically. The compiled pipeline behaves differently from what you wrote. Regulators will want to know what changed and why.

## Article 12: Record-Keeping

The Act requires high-risk AI systems to maintain logs sufficient to trace system behavior. DSPy pipelines have four layers that need logging:

1. **Compiled prompts.** After a Teleprompter optimizes your `dspy.Signature`, the actual prompt sent to the model is not the one you wrote. Log both the source signature and the compiled output. Use `dspy.settings.configure(trace=[])` to enable tracing, then persist those traces.

2. **Model calls.** Every `dspy.Predict` and `dspy.ChainOfThought` call hits an LM endpoint. Log the model name, version, input tokens, output tokens, and latency. DSPy's built-in history (`lm.history`) captures this per call.

3. **Retrieval queries.** If your pipeline uses `dspy.Retrieve`, log the query, the retriever configuration, and the returned documents. Retrieval-augmented generation shifts the compliance surface: you need to know not just what the model said, but what context it was given.

4. **Chain-of-thought traces.** `ChainOfThought` modules produce intermediate reasoning. These are Article 12 gold: they show the system's decision path. Store them alongside the final output.

Minimum retention: the Act does not specify a fixed period for all cases, but national authorities may require logs for the lifetime of the system plus a reasonable period after decommissioning. Design your storage accordingly.

## Article 13: Transparency

Document your optimization pipeline end to end:

- **Which Teleprompter** was used (`BootstrapFewShot`, `MIPRO`, `BayesianSignatureOptimizer`, etc.) and its hyperparameters.
- **Training examples** fed to the optimizer, their source, and any filtering applied.
- **Evaluation metrics** used to select the final compiled program. If you used `dspy.evaluate.Evaluate`, log the metric function, the dev set, and the scores.
- **Behavioral delta** between the uncompiled and compiled pipeline. Run identical inputs through both and document where outputs diverge. This is not optional for high-risk systems: the regulator needs to understand what optimization actually changed.

A compiled DSPy program is a black box unless you document the compilation. The source code alone is insufficient because the optimized prompts, few-shot examples, and instruction rewrites live in the compiled state, not in your `.py` files.

## Article 25: Value Chain Obligations

DSPy modules compose. A typical pipeline chains `ChainOfThought` into `Retrieve` into another `Predict`, potentially calling external APIs at each step. Each link in that chain carries its own compliance surface:

- **Your DSPy code**: you are the deployer. Full Article 25 obligations.
- **Model providers** (OpenAI, Anthropic, Google): they carry their own Article 25 obligations as providers of general-purpose AI models. Your responsibility is to document which provider and model version you use, and to keep records of their published compliance documentation.
- **Retrieval sources**: if `dspy.Retrieve` pulls from a third-party knowledge base, document the data source, update frequency, and any filtering.

Map the full chain. When something goes wrong, regulators will trace the pipeline link by link.

## Article 50: User Disclosure

If your DSPy pipeline interacts with end users (chatbots, decision-support tools, content generation), you must disclose that the output is AI-generated — unless this is obvious from the circumstances and context of use (Article 50(1)). A dedicated AI assistant satisfies this inherently. An AI feature embedded in a non-AI product requires explicit disclosure. This applies regardless of how the pipeline is optimized.

---

*This guide covers the EU AI Act as published (Regulation 2024/1689). National implementing measures may add requirements. If your system falls under Annex III, get legal counsel. This document is a technical mapping, not legal advice.*
