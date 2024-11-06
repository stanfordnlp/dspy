---
sidebar_position: 99999
---

# Additional Resources

## Tutorials

| **Level** |  **Tutorial** |  **Run in Colab** |  **Description** |
| --- | -------------  |  -------------  |  -------------  | 
| Beginner |  [**Getting Started**](https://github.com/stanfordnlp/dspy/blob/main/intro.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/intro.ipynb)  |  Introduces the basic building blocks in DSPy. Tackles the task of complex question answering with HotPotQA. |
| Beginner | [**Minimal Working Example**](/docs/docs/quick-start/getting-started-01.md) | N/A | Builds a very simple chain-of-thought program in DSPy for question answering. Very short. |
| Beginner | [**Compiling for Tricky Tasks**](https://github.com/stanfordnlp/dspy/blob/main/examples/nli/scone/scone.ipynb) | N/A | Teaches LMs to reason about logical statements and negation. Uses GPT-4 to bootstrap few-shot CoT demonstrations for GPT-3.5. Establishes a state-of-the-art result on [ScoNe](https://arxiv.org/abs/2305.19426). Contributed by [Chris Potts](https://twitter.com/ChrisGPotts/status/1740033519446057077). |
| Beginner | [**Local Models & Custom Datasets**](https://github.com/stanfordnlp/dspy/blob/main/skycamp2023.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/skycamp2023.ipynb) | Illustrates two different things together: how to use local models (Llama-2-13B in particular) and how to use your own data examples for training and development.
| Intermediate | [**The DSPy Paper**](https://arxiv.org/abs/2310.03714) | N/A | Sections 3, 5, 6, and 7 of the DSPy paper can be consumed as a tutorial. They include explained code snippets, results, and discussions of the abstractions and API.
| Intermediate | [**DSPy Assertions**](https://arxiv.org/abs/2312.13382) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/examples/longformqa/longformqa_assertions.ipynb) | Introduces example of applying DSPy Assertions while generating long-form responses to questions with citations. Presents comparative evaluation in both zero-shot and compiled settings.
| Intermediate | [**Finetuning for Complex Programs**](https://twitter.com/lateinteraction/status/1712135660797317577) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/examples/qa/hotpot/multihop_finetune.ipynb) | Teaches a local T5 model (770M) to do exceptionally well on HotPotQA. Uses only 200 labeled answers. Uses no hand-written prompts, no calls to OpenAI, and no labels for retrieval or reasoning.
| Advanced | [**Information Extraction**](https://twitter.com/KarelDoostrlnck/status/1724991014207930696) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1CpsOiLiLYKeGrhmq579_FmtGsD5uZ3Qe) | Tackles extracting information from long articles (biomedical research papers). Combines in-context learning and retrieval to set SOTA on BioDEX. Contributed by [Karel D’Oosterlinck](https://twitter.com/KarelDoostrlnck/status/1724991014207930696).  |


## Resources

- [DSPy talk at ScaleByTheBay Nov 2023](https://www.youtube.com/watch?v=Dt3H2ninoeY).
- [DSPy webinar with MLOps Learners](https://www.youtube.com/watch?v=im7bCLW2aM4), a bit longer with Q&A.
- Hands-on Overviews of DSPy by the community: [DSPy Explained! by Connor Shorten](https://www.youtube.com/watch?v=41EfOY0Ldkc), [DSPy explained by code_your_own_ai](https://www.youtube.com/watch?v=ycfnKPxBMck), [DSPy Crash Course by AI Bites](https://youtu.be/5-zgASQKkKQ?si=3gnmVouT5_rpk_nu)
- Interviews: [Weaviate Podcast in-person](https://www.youtube.com/watch?v=CDung1LnLbY), and you can find 6-7 other remote podcasts on YouTube from a few different perspectives/audiences.
- **Tracing in DSPy** with Arize Phoenix: [Tutorial for tracing your prompts and the steps of your DSPy programs](https://colab.research.google.com/github/Arize-ai/phoenix/blob/main/tutorials/tracing/dspy_tracing_tutorial.ipynb)
- **Tracing & Optimization Tracking in DSPy** with Parea AI: [Tutorial on tracing & evaluating a DSPy RAG program](https://docs.parea.ai/tutorials/dspy-rag-trace-evaluate/tutorial)
- **Prompt Optimization with DSPy and G-Eval Metrics** by Alberto Romero: [Medium article](https://medium.com/@a-romero/prompt-optimization-with-dspy-and-g-eval-metrics-e7d0bdd21b8b), [Repo](https://github.com/a-romero/dspy-risk-assessment), [Video](https://youtu.be/kK30U-XiiNI)