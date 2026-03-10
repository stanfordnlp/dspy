# Real-Time Web Research with DSPy and Exa

## The Problem: DSPy Programs Can't Access Live Information

Most DSPy retrieval setups rely on static corpora — ColBERTv2 over a 2017 Wikipedia dump, a one-time vector DB load, or a fixed document collection. This works for benchmarks, but real applications often need answers grounded in **current** information.

Consider a simple question: *"What major AI models were released in February 2026?"*

A ColBERTv2 pipeline or vector DB loaded last month will fail — the information simply doesn't exist in the corpus. The LM either hallucinates or says it doesn't know. There's no built-in way in DSPy to reach out to the live web.

This tutorial shows how to fix that by giving DSPy ReAct agents access to [Exa](https://exa.ai), a search engine designed specifically for programmatic AI access.

## Why Exa Over Other Search APIs

There are several search APIs available (Tavily, Serper, Brave, etc.). Here's what's different about Exa for DSPy users:

**Neural search handles the queries DSPy generates.** When a DSPy optimizer rewrites your program's prompts, the search queries it generates are natural-language sentences, not keyword strings. Most search APIs wrap Google or Bing under the hood, which are optimized for keyword queries. Exa uses a [custom neural search index](https://exa.ai/blog/building-web-scale-vector-db) trained on embeddings, so it matches on meaning. This matters because optimized DSPy programs produce varied, sometimes unusual queries — and Exa handles them without you needing to add a query-formatting step.

Exa achieves [state-of-the-art results on search quality benchmarks](https://exa.ai/evals) including MS Marco (10k queries), SimpleQA (factual RAG grounding), and challenging hand-crafted query sets — outperforming other search APIs on all three.

**Content extraction is built in.** Exa returns clean extracted text alongside search results. You don't need a separate scraping step, no headless browser, no HTML parsing. One API call returns both the search hits and readable page content. For ReAct agents that often need to search → read → reason, this cuts the tool-call count roughly in half.

**Structured filtering at the API level.** You can scope searches by domain, date range, and content category (`"news"`, `"research paper"`, `"company"`, `"tweet"`, etc.) in the API call itself. This lets you build precise research agents without post-filtering logic in your DSPy program.

**Find-similar search.** Given a URL, Exa returns semantically similar pages — a capability no other search API offers. This enables a research pattern where an agent finds one good source, then branches out to discover related work from different sites.

**Speed.** [Exa Instant](https://exa.ai/blog/exa-instant) returns results in under 200ms — up to 15x faster than competing APIs. For ReAct agents that make multiple search calls per trajectory, this adds up.

## Setup

```bash
pip install dspy exa-dspy
```

```python
import os
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["EXA_API_KEY"] = "your-exa-key"  # Get one free at exa.ai
```

## A Simple ReAct Agent with Web Search

[`exa-dspy`](https://github.com/exa-labs/exa-dspy) provides tool classes that plug directly into `dspy.ReAct`. Each tool has `name`, `desc`, and `args` properties, so ReAct discovers and invokes them automatically:

```python
import dspy
from exa_dspy import ExaSearchTool, ExaContentsTool

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Search returns titles, URLs, and text snippets in one call
search = ExaSearchTool(num_results=5, max_chars_per_result=500)

# Content extraction for reading full pages when needed
contents = ExaContentsTool(max_characters=3000)

agent = dspy.ReAct(
    "question -> answer",
    tools=[search, contents],
    max_iters=5,
)

result = agent(question="What major AI models were released in February 2026?")
print(result.answer)
```

**Sample output:**

```
The major AI models released in February 2026 include:

1. GPT-5.2 - An evolution of the GPT series, known for significant advancements in
   natural language processing and understanding.
2. Claude Opus 4.6 - Recognized for its exceptional capabilities in complex reasoning
   and problem-solving tasks, it ranked highly in user-preference leaderboards.
3. Gemini 3.1 Pro - Launched on February 19, this model is designed to offer versatility
   across various tasks and applications.
```

This is a question no static corpus can answer. The agent called `exa_search`, found recent articles about model launches, and synthesized a grounded answer from live web data.

## When To Use This (and When Not To)

**Use Exa when your DSPy program needs:**

- **Recency** — questions about events, releases, news from the last days/weeks/months
- **Breadth** — topics not covered by your existing corpus (long-tail queries, niche domains)
- **Diverse web sources** — when you need to pull from multiple sites rather than a single knowledge base
- **Structured web exploration** — searching within specific domains, date ranges, or content types

**Stick with vector DBs / ColBERTv2 when:**

- You have a fixed, curated corpus and don't need live data
- Your queries are domain-specific and your corpus has good coverage
- You need deterministic retrieval (same query always returns same results)

The two approaches are complementary. You can use a vector DB for your proprietary documents and Exa for live web context in the same DSPy program.

## Building a Focused Research Agent

The filtering capabilities are where Exa adds the most value for structured research. Here's an agent that only searches recent academic papers:

```python
from exa_dspy import ExaSearchTool, ExaContentsTool

arxiv_search = ExaSearchTool(
    num_results=5,
    include_domains=["arxiv.org"],
    start_published_date="2025-01-01",
    category="research paper",
    max_chars_per_result=800,
)
contents = ExaContentsTool(max_characters=5000)

agent = dspy.ReAct(
    "research_question -> literature_summary",
    tools=[arxiv_search, contents],
    max_iters=5,
)

result = agent(research_question="Recent advances in protein structure prediction beyond AlphaFold")
print(result.literature_summary)
```

With a keyword search API, you'd need to search, filter the results yourself, then make separate HTTP requests to scrape each page — and hope the sites don't block you. Here, domain restriction, date filtering, and content extraction all happen in one tool call.

You can also build a news-monitoring agent by setting `category="news"` and a rolling date window, or a competitive-intelligence agent with `include_domains` set to competitor sites.

## Multi-Source Exploration with Find Similar

`ExaFindSimilarTool` enables a research pattern that's unique to Exa: find one good source, then discover related pages from different sites.

```python
from exa_dspy import ExaSearchTool, ExaContentsTool, ExaFindSimilarTool

class ResearchAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.react = dspy.ReAct(
            "topic -> report",
            tools=[
                ExaSearchTool(num_results=5, max_chars_per_result=500),
                ExaContentsTool(max_characters=3000),
                ExaFindSimilarTool(num_results=3, exclude_source_domain=True),
            ],
            max_iters=8,
        )

    def forward(self, topic: str):
        return self.react(topic=topic)

agent = ResearchAgent()
result = agent(topic="How are large language models being used in drug discovery?")
print(result.report)
```

A typical trajectory the agent follows:

1. `exa_search("LLMs drug discovery")` → finds a Nature review article
2. `exa_get_contents("https://nature.com/...")` → reads the full text
3. `exa_find_similar("https://nature.com/...")` → discovers related papers from Science, bioRxiv, etc.
4. Reads one more source, then synthesizes a report grounded in multiple perspectives

The `exclude_source_domain=True` flag ensures similar results come from *different* sites. Without this, the agent tends to find more articles from the same publication.

## Optimizing with DSPy Compilers

Since `ResearchAgent` is a standard `dspy.Module`, you can use DSPy's optimizers to improve how it formulates search queries. This is where the Exa + DSPy combination becomes especially powerful.

DSPy's optimizers (BootstrapFewShot, MIPRO, SIMBA, etc.) tune the instructions and few-shot examples in your program to maximize a metric. When your program includes search tools, the optimizer effectively learns **what makes a good search query** for your specific task:

```python
from dspy.teleprompt import BootstrapFewShot

def has_citations(example, prediction, trace=None):
    """Check if the report references specific sources."""
    answer = prediction.report.lower()
    return (
        "http" in answer or "according to" in answer or "found that" in answer
    )

trainset = [
    dspy.Example(
        topic="mRNA vaccine delivery mechanisms",
        report="..."  # your gold-standard report
    ).with_inputs("topic"),
    dspy.Example(
        topic="quantum error correction progress in 2025",
        report="..."
    ).with_inputs("topic"),
    # 5-10 examples is usually enough for BootstrapFewShot
]

optimizer = BootstrapFewShot(metric=has_citations, max_bootstrapped_demos=3)
optimized_agent = optimizer.compile(ResearchAgent(), trainset=trainset)

# The optimized agent generates better search queries
result = optimized_agent(topic="CRISPR applications in agriculture")
print(result.report)
```

Because Exa's neural search handles natural-language queries well, the varied query styles that optimizers produce translate directly into good search results. With keyword-based search APIs, you'd often need an extra prompt or module to reformat the optimizer's output into effective keywords — with Exa, you don't.

## Tool Reference

### ExaSearchTool

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_results` | int | 5 | Number of results to return |
| `search_type` | str | `"auto"` | `"auto"`, `"neural"`, or `"keyword"` |
| `include_domains` | list | None | Only search these domains |
| `exclude_domains` | list | None | Exclude these domains |
| `start_published_date` | str | None | ISO date filter, e.g. `"2025-01-01"` |
| `end_published_date` | str | None | ISO date filter |
| `category` | str | None | `"news"`, `"research paper"`, `"company"`, `"tweet"`, etc. |
| `max_chars_per_result` | int | 1000 | Max text characters per result |

### ExaContentsTool

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_characters` | int | 5000 | Max characters to extract from a page |
| `summary` | bool | False | Include an AI-generated summary |
| `highlights` | bool | False | Include key text highlights |

### ExaFindSimilarTool

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_results` | int | 5 | Number of similar pages to return |
| `include_domains` | list | None | Only return results from these domains |
| `exclude_domains` | list | None | Exclude these domains |
| `exclude_source_domain` | bool | False | Exclude the input URL's domain from results |
| `max_chars_per_result` | int | 1000 | Max text characters per result |

## Further Reading

- [Exa search quality benchmarks](https://exa.ai/evals) — eval methodology and results vs other search APIs
- [Exa API docs](https://docs.exa.ai) — full API reference
- [`exa-dspy` on GitHub](https://github.com/exa-labs/exa-dspy) — source code and additional examples
- [DSPy ReAct documentation](https://dspy.ai/api/modules/ReAct/) — how ReAct discovers and invokes tools
