# Web Research Agent with DSPy and Exa

Most DSPy programs retrieve from static corpora — a vector DB loaded once, or ColBERTv2 over a Wikipedia snapshot. This works for closed-domain QA, but breaks down when your agent needs **live, up-to-date web information**: recent news, current research, real-time market data, or pages that didn't exist when your corpus was built.

[Exa](https://exa.ai) is a search engine built for AI agents. Unlike scraping Google or calling a keyword API, Exa uses a neural search index that understands *meaning* — so the natural-language queries that DSPy modules generate work well out of the box. It also returns clean extracted text, so you don't need a browser or HTML parser in your pipeline.

This tutorial shows how to give a DSPy ReAct agent the ability to search the web, read pages, and discover related sources using Exa.

## Why Exa + DSPy

**Neural search handles programmatic queries well.** DSPy optimizers rewrite prompts and generate queries automatically. These queries are often natural-language sentences rather than keyword strings. Exa's neural search mode is designed for exactly this — it matches on meaning, not keyword overlap. This means optimized DSPy programs produce better search results without you having to engineer query formats.

**Clean text extraction without scraping.** Exa returns extracted page text directly from its index. No headless browser, no BeautifulSoup, no rate limiting from site owners. Your agent gets clean content in one API call.

**Precision filtering.** You can scope searches by domain, date range, and content category (news, research paper, company, tweet, etc.) at the API level. This lets you build focused research agents — e.g., "only search arxiv.org papers from the last 6 months" — without post-filtering in your DSPy program.

**Find similar.** Exa can take a URL and return semantically similar pages. No other search API offers this. It's useful for agents that need to explore a topic from multiple angles — find one good source, then branch out.

## Setup

```bash
pip install dspy exa-dspy
```

```python
import os
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["EXA_API_KEY"] = "your-exa-key"  # Get one at exa.ai
```

## Quick Start: ReAct Agent with Web Search

`exa-dspy` provides tool classes that plug directly into `dspy.ReAct`:

```python
import dspy
from exa_dspy import ExaSearchTool, ExaContentsTool, ExaFindSimilarTool

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

search = ExaSearchTool(num_results=5, max_chars_per_result=500)
contents = ExaContentsTool(max_characters=3000)

agent = dspy.ReAct(
    "question -> answer",
    tools=[search, contents],
    max_iters=5,
)

result = agent(question="What were the major AI research breakthroughs in the last month?")
print(result.answer)
```

The agent will call `exa_search` to find relevant pages, optionally call `exa_get_contents` to read a full page, then reason over the results. Because Exa returns text content with search results, the agent often gets enough context from the search step alone.

## Building a Focused Research Agent

The real power shows up when you constrain searches. Here's an agent that only searches academic sources from the last year:

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

This is hard to replicate with generic search APIs — you'd need to search, filter results yourself, then scrape each page for content. Here it's one tool call.

## Multi-Source Exploration with Find Similar

`ExaFindSimilarTool` lets the agent branch out from a known good source. Combine all three tools for deeper research:

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

A typical agent trajectory:

1. `exa_search("LLMs in drug discovery")` → finds a Nature review article
2. `exa_get_contents("https://nature.com/...")` → reads the full article
3. `exa_find_similar("https://nature.com/...")` → discovers related papers from other journals
4. Synthesizes a report grounded in multiple sources

The `exclude_source_domain=True` flag ensures the similar results come from *different* sites, giving the agent diverse perspectives.

## Optimizing Search Quality with DSPy

Since `ResearchAgent` is a standard `dspy.Module`, you can optimize it with any DSPy optimizer. The optimizer will tune the instructions and few-shot examples that guide how the agent formulates search queries:

```python
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

# Define your metric
def research_quality(example, prediction, trace=None):
    """Score whether the report is grounded and comprehensive."""
    # Your evaluation logic here
    ...

# Prepare training examples
trainset = [
    dspy.Example(topic="mRNA vaccine delivery mechanisms", report="...").with_inputs("topic"),
    dspy.Example(topic="quantum error correction progress", report="...").with_inputs("topic"),
    # ...
]

# Optimize
optimizer = BootstrapFewShot(metric=research_quality, max_bootstrapped_demos=3)
optimized_agent = optimizer.compile(ResearchAgent(), trainset=trainset)
```

The optimizer learns what makes a good search query for your domain. Since Exa's neural search handles natural language well, the optimized queries translate directly into better results — you don't need to separately optimize query formatting.

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
