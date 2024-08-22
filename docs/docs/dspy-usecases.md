# DSPy Use Cases

We often get questions like "How are people using DSPy in practice?", like in production or for research. This list was created to collect a few pointers and to encourage others in the community to add their own work below.

This list is ever expanding and highly incomplete (WIP)! We'll be adding a bunch more. If you would like to add your product or research to this list, please make a PR.

## Table of Contents

1. [Company Use Cases](#a-few-company-use-cases)
2. [Providers and Integrations](#a-few-providers-and-integrations)
3. [Papers Using DSPy](#a-few-papers-using-dspy)
4. [Repositories about DSPy](#a-few-repositories-about-dspy)
5. [Blogs about DSPy](#a-few-blogs-about-dspy)

## A Few Company Use Cases

| **Name** | **Use Cases** |
|---|---|
| **[Replit](https://replit.com/)** | Synthesize diffs using large pre-trained code LLMs with a few-shot prompt pipeline implemented with DSPy. [Blog](https://blog.replit.com/code-repair) |
| **[Haize Labs](https://www.haizelabs.com/)** | Automated red-teaming for LLMs. [Blog](https://blog.haizelabs.com/posts/dspy/) |
| **[Jetblue](https://www.jetblue.com/) + [Databricks](https://www.databricks.com/)** | Multiple Chatbot Use Cases. [Blog](https://www.databricks.com/blog/optimizing-databricks-llm-pipelines-dspy) |
| **[Normal Computing](https://www.normalcomputing.com/)** | Specs from chip companies from English to intermediate formal languages |
| **[PingCAP](https://pingcap.com/)** | Building a knowledge graph. [Article](https://www.pingcap.com/article/building-a-graphrag-from-wikipedia-page-using-dspy-openai-and-tidb-vector-database/) |
| **[Plastic Labs](https://www.plasticlabs.ai/)** | Different pipelines within Honcho. [Blog](https://blog.plasticlabs.ai/blog/User-State-is-State-of-the-Art) |
| **[Procure.FYI](https://www.procure.fyi/)** | Process messy, publicly available technology spending and pricing data via DSPy. |
| **[RadiantLogic](https://www.radiantlogic.com/)** | AI Data Assistant. DSPy is used for the agent that routes the query, the context extraction module, the text-to-sql conversion engine, and the table summarization module. |
| **[Starops](https://staropshq.com/) & [Saya](https://heysaya.ai/)** | Building research documents given a user's corpus. Generate prompts to create more articles from example articles. |
| **[Hyperlint](https://hyperlint.com)** | Uses DSPy to generate technical documentation. DSPy helps to fetch relevant information and synthesize that into tutorials. |
| **[Tessel AI](https://tesselai.com/)** | Enhancing human-machine interaction with data use cases. |
| **[Dicer.ai](https://dicer.ai/)** | Uses DSPy for marketing AI to get the most from their paid ads |

Though we are aware of plenty of usage at Fortune 500 companies, this list only includes companies that have OKed being included for specific products so far.

## A Few Providers and Integrations

| **Name** | **Link** |
|---|---|
| **Databricks** | [https://www.databricks.com/blog/dspy-databricks](https://www.databricks.com/blog/dspy-databricks) |
| **Zenbase** | [https://zenbase.ai/](https://zenbase.ai/) |
| **LangWatch** | [https://langwatch.ai/blog/introducing-dspy-visualizer](https://langwatch.ai/blog/introducing-dspy-visualizer) |
| **Gradient** | [https://gradient.ai/blog/achieving-gpt-4-level-performance-at-lower-cost-using-dspy](https://gradient.ai/blog/achieving-gpt-4-level-performance-at-lower-cost-using-dspy) |
| **Snowflake** | [https://medium.com/snowflake/dspy-snowflake-140d6d947d73](https://medium.com/snowflake/dspy-snowflake-140d6d947d73) |
| **Langchain** | [https://python.langchain.com/v0.2/docs/integrations/providers/dspy/](https://python.langchain.com/v0.2/docs/integrations/providers/dspy/) |
| **Weaviate** | [https://weaviate.io/blog/dspy-optimizers](https://weaviate.io/blog/dspy-optimizers) |

## A Few Papers Using DSPy

| **Name** | **Description** |
|---|---|
| **[STORM](https://arxiv.org/abs/2402.14207v1)** | Wikipage generation |
| **[IReRa](https://arxiv.org/abs/2401.12178)** | Extreme reranking |
| **[The Unreasonable Effectiveness of Eccentric Automatic Prompts](https://arxiv.org/abs/2402.10949v2)** | General Prompt Optimization |
| **[Palimpzest](https://arxiv.org/abs/2405.14696)** | A Declarative System for Optimizing AI Workloads |
| **[AI Agents that Matter](https://arxiv.org/abs/2407.01502v1)** | Agent efficiency optimization |
| **[Prompts as Auto-Optimized Training Hyperparameters (PATH)](https://arxiv.org/abs/2406.11706)** | Training Best-in-Class IR Models from Scratch with 10 Gold Labels |
| **[Empathetic Dialogues for English learning (EDEN)](https://arxiv.org/abs/2406.17982v1)** | Uses adaptive empathetic feedback to improve student grit |
| **[ECG-Chat](https://arxiv.org/pdf/2408.08849)** | Uses DSPy with GraphRAG for medical report generation |

TODO: This list is missing a few key things like UofT's winning system at MEDIQA and UMD's Suicide Detection system that outperforms 20-hour human prompt engineering by 40%, etc.

## A Few Repositories about DSPy

| **Name** | **Description/Link** |
|---|---|
| **Homework from Stanford CS 224U** | [Github](https://github.com/cgpotts/cs224u/blob/main/hw_openqa.ipynb) |
| **Optimizing LM for Text2SQL using DSPy** | Text2SQL [Github](https://github.com/jjovalle99/DSPy-Text2SQL) |
| **DSPy PII Masking Demo by Eric Ness** | PII Masking [Colab](https://colab.research.google.com/drive/1KZR1sGTp_RLWUJPAiK1FKPKI-Qn9neUm?usp=sharing) |

TODO: This list in particular is highly incomplete. There are dozens of others, e.g. see "Used By" in GitHub for some examples, e.g. STORM (10k stars), Theory of Mind, BIG BENCH, Indian Languages NLI, etc.

## A Few Blogs about DSPy

| **Name** |
|---|
| **[Why I bet on DSPy](https://blog.isaacbmiller.com/posts/dspy)** |
| **[Why I'm excited about DSPy](https://substack.stephen.so/p/why-im-excited-about-dspy)** |

TODO: This list in particular is highly incomplete. There are several good other ones.

### Weaviate has a directory of 10 amazing notebooks and 6 podcasts!

Huge shoutout to them for the massive support <3. See the [Weaviate DSPy directory](https://weaviate.io/developers/weaviate/more-resources/dspy).
