# Web Search Agent with DSPy ReAct and Exa

This tutorial shows how to build a research agent that can search the web, fetch page content, and find similar sources using DSPy ReAct with [Exa](https://exa.ai)'s search API.

## What You'll Build

A research agent that can:

- **Search the web** for real-time information using neural, keyword, or auto search
- **Fetch and extract content** from any URL
- **Find similar pages** given a reference URL
- **Reason over multiple sources** to produce grounded answers

## Setup

```bash
pip install dspy exa-dspy
```

Set your API keys:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["EXA_API_KEY"] = "your-exa-key"
```

## Step 1: Configure Tools

`exa-dspy` provides three tool classes that work directly with `dspy.ReAct`:

```python
import dspy
from exa_dspy import ExaSearchTool, ExaContentsTool, ExaFindSimilarTool

# Configure DSPy
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Web search — returns titles, URLs, and text snippets
search = ExaSearchTool(num_results=5, max_chars_per_result=500)

# Content extraction — fetches clean text from a URL
contents = ExaContentsTool(max_characters=3000)

# Similar page discovery — finds related pages given a URL
similar = ExaFindSimilarTool(num_results=3)
```

Each tool exposes `name`, `desc`, and `args` properties so ReAct can discover and invoke them automatically.

## Step 2: Build the ReAct Agent

```python
agent = dspy.ReAct(
    "question -> answer",
    tools=[search, contents, similar],
    max_iters=5,
)

result = agent(question="What are the latest breakthroughs in quantum computing?")
print(result.answer)
```

The agent will:

1. Call `exa_search` with a query
2. Read the returned titles, URLs, and snippets
3. Optionally call `exa_get_contents` to read a full page
4. Reason over the information and produce an answer

## Step 3: Domain-Specific Research Agent

You can constrain searches to specific domains or date ranges:

```python
# Only search academic sources published in the last year
arxiv_search = ExaSearchTool(
    num_results=5,
    include_domains=["arxiv.org", "scholar.google.com"],
    start_published_date="2025-01-01",
    max_chars_per_result=800,
)

agent = dspy.ReAct(
    "research_question -> summary",
    tools=[arxiv_search, contents],
    max_iters=5,
)

result = agent(research_question="Recent advances in protein structure prediction")
print(result.summary)
```

## Step 4: Multi-Source Research with Find Similar

Combine all three tools for deeper research. The agent can search, read pages, then discover related sources:

```python
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

## Step 5: Run It

```python
def main():
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    search = ExaSearchTool(num_results=5, max_chars_per_result=500)
    contents = ExaContentsTool(max_characters=3000)
    similar = ExaFindSimilarTool(num_results=3)

    agent = dspy.ReAct(
        "question -> answer",
        tools=[search, contents, similar],
        max_iters=5,
    )

    queries = [
        "What is the current state of nuclear fusion research?",
        "What companies are leading in autonomous vehicle technology?",
        "How is CRISPR being used in agriculture?",
    ]

    for q in queries:
        print(f"Q: {q}")
        result = agent(question=q)
        print(f"A: {result.answer}\n")

if __name__ == "__main__":
    main()
```

## Tool Configuration Reference

### ExaSearchTool

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_results` | int | 5 | Number of results to return |
| `search_type` | str | "auto" | Search mode: "auto", "neural", or "keyword" |
| `include_domains` | list | None | Restrict to these domains |
| `exclude_domains` | list | None | Exclude these domains |
| `start_published_date` | str | None | Filter by publish date (ISO format) |
| `end_published_date` | str | None | Filter by publish date (ISO format) |
| `category` | str | None | Filter by content category |
| `max_chars_per_result` | int | 1000 | Max text per result |

### ExaContentsTool

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_characters` | int | 5000 | Max characters to extract |
| `summary` | bool | False | Include AI-generated summary |
| `highlights` | bool | False | Include key highlights |

### ExaFindSimilarTool

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_results` | int | 5 | Number of similar pages |
| `include_domains` | list | None | Restrict to these domains |
| `exclude_domains` | list | None | Exclude these domains |
| `exclude_source_domain` | bool | False | Exclude the input URL's domain |
| `max_chars_per_result` | int | 1000 | Max text per result |

## Next Steps

- **Optimize with DSPy** &mdash; use `dspy.BootstrapFewShot` or `dspy.MIPRO` to auto-tune the agent's prompts for better search queries and reasoning
- **Add more tools** &mdash; combine Exa search with database lookups, calculators, or other APIs
- **Use category filters** &mdash; Exa supports categories like `"news"`, `"research paper"`, `"company"`, `"tweet"` for more targeted results
- **Stream responses** &mdash; use DSPy's streaming support for real-time output in production
