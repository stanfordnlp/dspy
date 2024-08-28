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
| **[JetBlue](https://www.jetblue.com/) + [Databricks](https://www.databricks.com/)** | Multiple Chatbot Use Cases. [Blog](https://www.databricks.com/blog/optimizing-databricks-llm-pipelines-dspy) |
| **[PingCAP](https://pingcap.com/)** | Building a knowledge graph. [Article](https://www.pingcap.com/article/building-a-graphrag-from-wikipedia-page-using-dspy-openai-and-tidb-vector-database/) |
| **[Plastic Labs](https://www.plasticlabs.ai/)** | Different pipelines within Honcho. [Blog](https://blog.plasticlabs.ai/blog/User-State-is-State-of-the-Art) |
| **[Normal Computing](https://www.normalcomputing.com/)** | Translate specs from chip companies from English to intermediate formal languages |
| **[Procure.FYI](https://www.procure.fyi/)** | Process messy, publicly available technology spending and pricing data via DSPy. |
| **[RadiantLogic](https://www.radiantlogic.com/)** | AI Data Assistant. DSPy is used for the agent that routes the query, the context extraction module, the text-to-sql conversion engine, and the table summarization module. |
| **[Starops](https://staropshq.com/) & [Saya](https://heysaya.ai/)** | Building research documents given a user's corpus. Generate prompts to create more articles from example articles. |
| **[Hyperlint](https://hyperlint.com)** | Uses DSPy to generate technical documentation. DSPy helps to fetch relevant information and synthesize that into tutorials. |
| **[Tessel AI](https://tesselai.com/)** | Enhancing human-machine interaction with data use cases. |
| **[Dicer.ai](https://dicer.ai/)** | Uses DSPy for marketing AI to get the most from their paid ads |
| **[Howie](https://howie.ai)** | Using DSPy to automate meeting scheduling through email. |
| **[Isoform.ai](https://isoform.ai)** | Building custom integrations using DSPy. |

Though we are aware of plenty of usage at Fortune 500 companies, this list only includes companies that have public posts or have OKed being included for specific products so far.

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
| **[Stanford DSPy - Qdrant](https://qdrant.tech/documentation/frameworks/dspy/)** |
| **Weights & Biases Weave** | [https://weave-docs.wandb.ai/guides/integrations/dspy/](https://weave-docs.wandb.ai/guides/integrations/dspy/) |
| **Milvus** | [https://milvus.io/docs/integrate_with_dspy.md](https://milvus.io/docs/integrate_with_dspy.md) |
| **Neo4j** | [https://neo4j.com/labs/genai-ecosystem/dspy/](https://neo4j.com/labs/genai-ecosystem/dspy/) |
| **Lightning AI** | [https://lightning.ai/lightning-ai/studios/dspy-programming-with-foundation-models](https://lightning.ai/lightning-ai/studios/dspy-programming-with-foundation-models) |
| **Haystack** | [https://towardsdatascience.com/automating-prompt-engineering-with-dspy-and-haystack-926a637a3f43](https://towardsdatascience.com/automating-prompt-engineering-with-dspy-and-haystack-926a637a3f43) |


## A Few Papers Using DSPy

| **Name** | **Description** |
|---|---|
| **[STORM](https://arxiv.org/abs/2402.14207)** | Writing Wikipedia-like Articles From Scratch. |
| **[PATH](https://arxiv.org/abs/2406.11706)** | Prompts as Auto-Optimized Training Hyperparameters: Training Best-in-Class IR Models from Scratch with 10 Gold Labels |
| **[WangLab @ MEDIQA](https://arxiv.org/abs/2404.14544)** | UofT's winning system at MEDIQA, outperforms the next best system by 20 points |
| **[UMD's Suicide Detection System](https://arxiv.org/abs/2406.06608)** | Outperforms 20-hour expert human prompt engineering by 40% |
| **[IReRa](https://arxiv.org/abs/2401.12178)** | Infer-Retrieve-Rank: Extreme Classification with > 10,000 Labels |
| **[The Unreasonable Effectiveness of Eccentric Prompts](https://arxiv.org/abs/2402.10949v2)** | General Prompt Optimization |
| **[Palimpzest](https://arxiv.org/abs/2405.14696)** | A Declarative System for Optimizing AI Workloads |
| **[AI Agents that Matter](https://arxiv.org/abs/2407.01502v1)** | Agent Efficiency Optimization |
| **[EDEN](https://arxiv.org/abs/2406.17982v1)** | Empathetic Dialogues for English Learning: Uses adaptive empathetic feedback to improve student grit |
| **[ECG-Chat](https://arxiv.org/pdf/2408.08849)** | Uses DSPy with GraphRAG for medical report generation |
| **[DSPy Assertions](https://arxiv.org/abs/2312.13382)** | Various applications of imposing hard and soft constraints on LM outputs |
| **[DSPy Guardrails](https://boxiyu.github.io/assets/pdf/DSPy_Guardrails.pdf)** | Reduce the attack success rate of CodeAttack, decreasing from 75% to 5% |


## A Few Repositories (or other OSS examples) using DSPy

| **Name** | **Description/Link** |
|---|---|
| **Stanford CS 224U Homework** | [Github](https://github.com/cgpotts/cs224u/blob/main/hw_openqa.ipynb) |
| **STORM Report Generation (10,000 GitHub stars)** | [Github](https://github.com/stanford-oval/storm) |
| **DSPy Redteaming** | [Github](https://github.com/haizelabs/dspy-redteam) |
| **DSPy Theory of Mind** |  [Github](https://github.com/plastic-labs/dspy-opentom) |
| **Indic cross-lingual Natural Language Inference** |  [Github](https://github.com/saifulhaq95/DSPy-Indic/blob/main/indicxlni.ipynb) |
| **Optimizing LM for Text2SQL using DSPy** | Text2SQL [Github](https://github.com/jjovalle99/DSPy-Text2SQL) |
| **DSPy PII Masking Demo by Eric Ness** | PII Masking [Colab](https://colab.research.google.com/drive/1KZR1sGTp_RLWUJPAiK1FKPKI-Qn9neUm?usp=sharing) |
| **DSPy on BIG-Bench Hard Example** |  [Github](https://drchrislevy.github.io/posts/dspy/dspy.html) |
| **Building a chess playing agent using DSPy** |  [Github](https://medium.com/thoughts-on-machine-learning/building-a-chess-playing-agent-using-dspy-9b87c868f71e) |

TODO: This list in particular is highly incomplete. There is a couple dozen other good ones.

## A Few Blogs & Videos on using DSPy

| **Name** | **Link** |
|---|---|
| **Why I bet on DSPy** | [Blog](https://blog.isaacbmiller.com/posts/dspy) |
| **Not Your Average Prompt Engineering** | [Blog](https://jina.ai/news/dspy-not-your-average-prompt-engineering/) |
| **Why I'm excited about DSPy** | [Blog](https://substack.stephen.so/p/why-im-excited-about-dspy) |
| **DSPy Intro from Sephora (25K views)** | [Link](https://www.youtube.com/watch?v=D2HurSldDkE) |
| **DSPy Explained! (60K views)** | [Link](https://www.youtube.com/watch?v=41EfOY0Ldkc) |
| **Structured Outputs with DSPy** | [Link](https://www.youtube.com/watch?v=tVw3CwrN5-8) |
| **Achieving GPT-4 Performance at Lower Cost** | [Link](https://gradient.ai/blog/achieving-gpt-4-level-performance-at-lower-cost-using-dspy) |
| **Optimization with DSPy and LangChain** | [Link](https://www.youtube.com/watch?v=4EXOmWeqXRc) |
| **Automated Prompt Engineering + Visualization** | [Link](https://www.youtube.com/watch?v=eAZ2LtJ6D5k) |
| **Transforming LM Calls into Pipelines** | [Link](https://www.youtube.com/watch?v=NoaDWKHdkHg) |
| **NeurIPS Hacker Cup: DSPy for Code Gen** | [Link](https://www.youtube.com/watch?v=gpe-rtJN8z8) |
| **MIPRO and DSPy - Weaviate Podcast** | [Link](https://www.youtube.com/watch?v=skMH3DOV_UQ) |
| **Getting Started with RAG in DSPy** | [Link](https://www.youtube.com/watch?v=CEuUG4Umfxs) |
| **Adding Depth to DSPy Programs** | [Link](https://www.youtube.com/watch?v=0c7Ksd6BG88) |
| **Programming Foundation Models with DSPy** | [Link](https://www.youtube.com/watch?v=Y94tw4eDHW0) |
| **DSPy End-to-End: SF Meetup** | [Link](https://www.youtube.com/watch?v=Y81DoFmt-2U) |
| **SBTB23 DSPy** | [Link](https://www.youtube.com/watch?v=Dt3H2ninoeY) |
| **DSPy and ColBERT - Weaviate Podcast** | [Link](https://www.youtube.com/watch?v=CDung1LnLbY) |



TODO: This list in particular is highly incomplete. There are dozens of other good ones.

### Weaviate has a directory of 10 amazing notebooks and 6 podcasts!

Huge shoutout to them for the massive support <3. See the [Weaviate DSPy directory](https://weaviate.io/developers/weaviate/more-resources/dspy).
