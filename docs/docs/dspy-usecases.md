# DSPy Use Cases

We often get questions like "How are people using DSPy in practice?", both in production and for research. This list was created to collect a few pointers and to encourage others in the community to add their own work below.

This list is ever expanding and highly incomplete (WIP)! We'll be adding a bunch more. If you would like to add your product or research to this list, please make a PR.

## Table of Contents

1. [Company Use Cases](#a-few-company-use-cases)
2. [Research Papers using DSPy](#a-few-papers-using-dspy)
3. [Open-Source Software using DSPy](#a-few-repositories-or-other-oss-examples-using-dspy)
4. [Providers with DSPy support](#a-few-providers-integrations-and-related-blog-releases)
5. [Blogs & Videos on using DSPy](#a-few-blogs--videos-on-using-dspy)


## A Few Company Use Cases

| **Name** | **Use Cases** |
|---|---|
| **[JetBlue](https://www.jetblue.com/)** | Multiple chatbot use cases. [Blog](https://www.databricks.com/blog/optimizing-databricks-llm-pipelines-dspy) |
| **[Replit](https://replit.com/)** | Synthesize diffs using code LLMs using a DSPy pipeline. [Blog](https://blog.replit.com/code-repair) |
| **[Databricks](https://www.databricks.com/)** | Research, products, and customer solutions around LM Judges, RAG, classification, and other applications. [Blog](https://www.databricks.com/blog/dspy-databricks) [Blog II](https://www.databricks.com/customers/ddi) |
| **[Sephora](https://www.sephora.com/)** | Undisclosed agent usecases; perspectives shared in [DAIS Session](https://www.youtube.com/watch?v=D2HurSldDkE). |
| **[Zoro UK](https://www.zoro.co.uk/)** | E-commerce applications around structured shopping. [Portkey Session](https://www.youtube.com/watch?v=_vGKSc1tekE) |
| **[VMware](https://www.vmware.com/)** | RAG and other prompt optimization applications. [Interview in The Register.](https://www.theregister.com/2024/02/22/prompt_engineering_ai_models/) [Business Insider.](https://www.businessinsider.com/chaptgpt-large-language-model-ai-prompt-engineering-automated-optimizer-2024-3) |
| **[Haize Labs](https://www.haizelabs.com/)** | Automated red-teaming for LLMs. [Blog](https://blog.haizelabs.com/posts/dspy/) |
| **[Plastic Labs](https://www.plasticlabs.ai/)** | Different pipelines within Honcho. [Blog](https://blog.plasticlabs.ai/blog/User-State-is-State-of-the-Art) |
| **[PingCAP](https://pingcap.com/)** | Building a knowledge graph. [Article](https://www.pingcap.com/article/building-a-graphrag-from-wikipedia-page-using-dspy-openai-and-tidb-vector-database/) |
| **[Salomatic](https://langtrace.ai/blog/case-study-how-salomatic-used-langtrace-to-build-a-reliable-medical-report-generation-system)** | Enriching medical reports using DSPy. [Blog](https://langtrace.ai/blog/case-study-how-salomatic-used-langtrace-to-build-a-reliable-medical-report-generation-system) |
| **[Truelaw](https://www.youtube.com/watch?v=O0F3RAWZNfM)** | How Truelaw builds bespoke LLM pipelines for law firms using DSPy. [Podcast](https://www.youtube.com/watch?v=O0F3RAWZNfM) |
| **[Moody's](https://www.moodys.com/)** | Leveraging DSPy to optimize RAG systems, LLM-as-a-Judge, and agentic systems for financial workflows. |
| **[Normal Computing](https://www.normalcomputing.com/)** | Translate specs from chip companies from English to intermediate formal languages |
| **[Procure.FYI](https://www.procure.fyi/)** | Process messy, publicly available technology spending and pricing data via DSPy. |
| **[RadiantLogic](https://www.radiantlogic.com/)** | AI Data Assistant. DSPy is used for the agent that routes the query, the context extraction module, the text-to-sql conversion engine, and the table summarization module. |
| **[Raia](https://raiahealth.com/)** | Using DSPy for AI-powered Personal Healthcare Agents. |
| **[Hyperlint](https://hyperlint.com)** | Uses DSPy to generate technical documentation. DSPy helps to fetch relevant information and synthesize that into tutorials. |
| **[Starops](https://staropshq.com/) & [Saya](https://heysaya.ai/)** | Building research documents given a user's corpus. Generate prompts to create more articles from example articles. |
| **[Tessel AI](https://tesselai.com/)** | Enhancing human-machine interaction with data use cases. |
| **[Dicer.ai](https://dicer.ai/)** | Uses DSPy for marketing AI to get the most from their paid ads. |
| **[Howie](https://howie.ai)** | Using DSPy to automate meeting scheduling through email. |
| **[Isoform.ai](https://isoform.ai)** | Building custom integrations using DSPy. |
| **[Trampoline AI](https://trampoline.ai)** | Uses DSPy to power their data-augmentation and LM pipelines. |
| **[Pretrain](https://pretrain.com)** | Uses DSPy to automatically optimize AI performance towards user-defined tasks based on uploaded examples. |

WIP. This list mainly includes companies that have public posts or have OKed being included for specific products so far.


## A Few Papers Using DSPy

| **Name** | **Description** |
|---|---|
| **[STORM](https://arxiv.org/abs/2402.14207)** | Writing Wikipedia-like Articles From Scratch. |
| **[PATH](https://arxiv.org/abs/2406.11706)** | Prompts as Auto-Optimized Training Hyperparameters: Training Best-in-Class IR Models from Scratch with 10 Gold Labels |
| **[WangLab @ MEDIQA](https://arxiv.org/abs/2404.14544)** | UofT's winning system at MEDIQA, outperforms the next best system by 20 points |
| **[UMD's Suicide Detection System](https://arxiv.org/abs/2406.06608)** | Outperforms 20-hour expert human prompt engineering by 40% |
| **[IReRa](https://arxiv.org/abs/2401.12178)** | Infer-Retrieve-Rank: Extreme Classification with > 10,000 Labels |
| **[Unreasonably Effective Eccentric Prompts](https://arxiv.org/abs/2402.10949v2)** | General Prompt Optimization |
| **[Palimpzest](https://arxiv.org/abs/2405.14696)** | A Declarative System for Optimizing AI Workloads |
| **[AI Agents that Matter](https://arxiv.org/abs/2407.01502v1)** | Agent Efficiency Optimization |
| **[EDEN](https://arxiv.org/abs/2406.17982v1)** | Empathetic Dialogues for English Learning: Uses adaptive empathetic feedback to improve student grit |
| **[ECG-Chat](https://arxiv.org/pdf/2408.08849)** | Uses DSPy with GraphRAG for medical report generation |
| **[DSPy Assertions](https://arxiv.org/abs/2312.13382)** | Various applications of imposing hard and soft constraints on LM outputs |
| **[DSPy Guardrails](https://boxiyu.github.io/assets/pdf/DSPy_Guardrails.pdf)** | Reduce the attack success rate of CodeAttack, decreasing from 75% to 5% |
| **[Co-STORM](https://arxiv.org/pdf/2408.15232)** | Collaborative STORM: Generate Wikipedia-like articles through collaborative discourse among users and multiple LM agents |

## A Few Repositories (or other OSS examples) using DSPy

| **Name** | **Description/Link** |
|---|---|
| **Stanford CS 224U Homework** | [Github](https://github.com/cgpotts/cs224u/blob/main/hw_openqa.ipynb) |
| **STORM Report Generation (10,000 GitHub stars)** | [Github](https://github.com/stanford-oval/storm) |
| **DSPy Redteaming** | [Github](https://github.com/haizelabs/dspy-redteam) |
| **DSPy Theory of Mind** |  [Github](https://github.com/plastic-labs/dspy-opentom) |
| **Indic cross-lingual Natural Language Inference** |  [Github](https://github.com/saifulhaq95/DSPy-Indic/blob/main/indicxlni.ipynb) |
| **Optimizing LM for Text2SQL using DSPy** | [Github](https://github.com/jjovalle99/DSPy-Text2SQL) |
| **DSPy PII Masking Demo by Eric Ness** | [Colab](https://colab.research.google.com/drive/1KZR1sGTp_RLWUJPAiK1FKPKI-Qn9neUm?usp=sharing) |
| **DSPy on BIG-Bench Hard Example** |  [Github](https://drchrislevy.github.io/posts/dspy/dspy.html) |
| **Building a chess playing agent using DSPy** |  [Github](https://medium.com/thoughts-on-machine-learning/building-a-chess-playing-agent-using-dspy-9b87c868f71e) |
| **Ittia Research Fact Checking** | [Github](https://github.com/ittia-research/check) |
| **Strategic Debate via Tree-of-Thought** | [Github](https://github.com/zbambergerNLP/strategic-debate-tot) |
| **Sanskrit to English Translation App**| [Github](https://github.com/ganarajpr/sanskrit-translator-dspy) |
| **DSPy for extracting features from PDFs on arXiv**| [Github](https://github.com/S1M0N38/dspy-arxiv) |
| **DSPygen: DSPy in Ruby on Rails**| [Github](https://github.com/seanchatmangpt/dspygen) |
| **DSPy Inspector**| [Github](https://github.com/Neoxelox/dspy-inspector) |
| **DSPy with FastAPI**| [Github](https://github.com/diicellman/dspy-rag-fastapi) |
| **DSPy for Indian Languages**| [Github](https://github.com/saifulhaq95/DSPy-Indic) |
| **Hurricane: Blog Posts with Generative Feedback Loops!**| [Github](https://github.com/weaviate-tutorials/Hurricane) |
| **RAG example using DSPy, Gradio, FastAPI, and Ollama**| [Github](https://github.com/diicellman/dspy-gradio-rag) |
| **Synthetic Data Generation**| [Github](https://colab.research.google.com/drive/1CweVOu0qhTC0yOfW5QkLDRIKuAuWJKEr?usp=sharing) |
| **Self Discover**| [Github](https://colab.research.google.com/drive/1GkAQKmw1XQgg5UNzzy8OncRe79V6pADB?usp=sharing) |

TODO: This list in particular is highly incomplete. There are a couple dozen other good ones.

## A Few Providers, Integrations, and related Blog Releases

| **Name** | **Link** |
|---|---|
| **Databricks** | [Link](https://www.databricks.com/blog/dspy-databricks) |
| **Zenbase** | [Link](https://zenbase.ai/) |
| **LangWatch** | [Link](https://langwatch.ai/blog/introducing-dspy-visualizer) |
| **Gradient** | [Link](https://gradient.ai/blog/achieving-gpt-4-level-performance-at-lower-cost-using-dspy) |
| **Snowflake** | [Link](https://medium.com/snowflake/dspy-snowflake-140d6d947d73) |
| **Langchain** | [Link](https://python.langchain.com/v0.2/docs/integrations/providers/dspy/) |
| **Weaviate** | [Link](https://weaviate.io/blog/dspy-optimizers) |
| **Qdrant** | [Link](https://qdrant.tech/documentation/frameworks/dspy/) |
| **Weights & Biases Weave** | [Link](https://weave-docs.wandb.ai/guides/integrations/dspy/) |
| **Milvus** | [Link](https://milvus.io/docs/integrate_with_dspy.md) |
| **Neo4j** | [Link](https://neo4j.com/labs/genai-ecosystem/dspy/) |
| **Lightning AI** | [Link](https://lightning.ai/lightning-ai/studios/dspy-programming-with-foundation-models) |
| **Haystack** | [Link](https://towardsdatascience.com/automating-prompt-engineering-with-dspy-and-haystack-926a637a3f43) |
| **Arize** | [Link](https://docs.arize.com/phoenix/tracing/integrations-tracing/dspy) |
| **LlamaIndex** | [Link](https://github.com/stanfordnlp/dspy/blob/main/examples/llamaindex/dspy_llamaindex_rag.ipynb) |
| **Langtrace** | [Link](https://docs.langtrace.ai/supported-integrations/llm-frameworks/dspy) |
| **Langfuse** | [Link](https://langfuse.com/docs/integrations/dspy) |

## A Few Blogs & Videos on using DSPy

| **Name** | **Link** |
|---|---|
| **Blog Posts** | |
| **Why I bet on DSPy** | [Blog](https://blog.isaacbmiller.com/posts/dspy) |
| **Not Your Average Prompt Engineering** | [Blog](https://jina.ai/news/dspy-not-your-average-prompt-engineering/) |
| **Why I'm excited about DSPy** | [Blog](https://substack.stephen.so/p/why-im-excited-about-dspy) |
| **Achieving GPT-4 Performance at Lower Cost** | [Link](https://gradient.ai/blog/achieving-gpt-4-level-performance-at-lower-cost-using-dspy) |
| **Prompt engineering is a task best left to AI models** | [Link](https://www.theregister.com/2024/02/22/prompt_engineering_ai_models/) |
| **What makes DSPy a valuable framework for developing complex language model pipelines?** | [Link](https://medium.com/@sujathamudadla1213/what-makes-dspy-a-valuable-framework-for-developing-complex-language-model-pipelines-edfa5b4bcf9b) |
| **DSPy: A new framework to program your foundation models just by prompting** | [Link](https://www.linkedin.com/pulse/dspy-new-framework-program-your-foundation-models-just-prompting-lli4c/) |
| **Intro to DSPy: Goodbye Prompting, Hello Programming** | [Link](https://medium.com/towards-data-science/intro-to-dspy-goodbye-prompting-hello-programming-4ca1c6ce3eb9) |
| **DSPyGen: Revolutionizing AI** | [Link](https://www.linkedin.com/pulse/launch-alert-dspygen-20242252-revolutionizing-ai-sean-chatman--g9f1c/) |
| **Building an AI Assistant with DSPy** | [Link](https://www.linkedin.com/pulse/building-ai-assistant-dspy-valliappa-lakshmanan-vgnsc/) |
| **Videos** | |
| **DSPy Explained! (60K views)** | [Link](https://www.youtube.com/watch?v=41EfOY0Ldkc) |
| **DSPy Intro from Sephora (25K views)** | [Link](https://www.youtube.com/watch?v=D2HurSldDkE) |
| **Structured Outputs with DSPy** | [Link](https://www.youtube.com/watch?v=tVw3CwrN5-8) |
| **DSPy and ColBERT - Weaviate Podcast** | [Link](https://www.youtube.com/watch?v=CDung1LnLbY) |
| **SBTB23 DSPy** | [Link](https://www.youtube.com/watch?v=Dt3H2ninoeY) |
| **Optimization with DSPy and LangChain** | [Link](https://www.youtube.com/watch?v=4EXOmWeqXRc) |
| **Automated Prompt Engineering + Visualization** | [Link](https://www.youtube.com/watch?v=eAZ2LtJ6D5k) |
| **Transforming LM Calls into Pipelines** | [Link](https://www.youtube.com/watch?v=NoaDWKHdkHg) |
| **NeurIPS Hacker Cup: DSPy for Code Gen** | [Link](https://www.youtube.com/watch?v=gpe-rtJN8z8) |
| **MIPRO and DSPy - Weaviate Podcast** | [Link](https://www.youtube.com/watch?v=skMH3DOV_UQ) |
| **Getting Started with RAG in DSPy** | [Link](https://www.youtube.com/watch?v=CEuUG4Umfxs) |
| **Adding Depth to DSPy Programs** | [Link](https://www.youtube.com/watch?v=0c7Ksd6BG88) |
| **Programming Foundation Models with DSPy** | [Link](https://www.youtube.com/watch?v=Y94tw4eDHW0) |
| **DSPy End-to-End: SF Meetup** | [Link](https://www.youtube.com/watch?v=Y81DoFmt-2U) |
| **Monitoring & Tracing DSPy with Langtrace** | [Link](https://langtrace.ai/blog/announcing-dspy-support-in-langtrace) |
| **Teaching chat models to solve chess puzzles using DSPy + Finetuning** | [Link](https://raw.sh/posts/chess_puzzles) |

TODO: This list in particular is highly incomplete. There are dozens of other good ones. To allow space, divide into opintionated blogs / podcasts / interviews vs. tutorials & talks.

Credit: Some of these resources were originally compiled in the [Awesome DSPy](https://github.com/ganarajpr/awesome-dspy/tree/master) repo.

### Weaviate has a directory of 10 amazing notebooks and 6 podcasts!

Huge shoutout to them for the massive support ❤️. See the [Weaviate DSPy directory](https://weaviate.io/developers/weaviate/more-resources/dspy).
