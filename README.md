# 🎓𝗗𝗦𝗣: The Demonstrate–Search–Predict Framework

The **DSP** framework provides a programming abstraction for rapidly building reliable AI systems with language models (LMs) and retrieval models (RMs). You write a **DSP program** in a few lines of code, which describes at high level how the problem you'd like to tackle (e.g., answering complex questions about the world) should be _decomposed_ into smaller _transformations_. Your program implements a sophisticated pipeline composed of such transformations (e.g., `generate a search query to find missing information` or `answer this question using the supplied context`), which the LM or RM can handle much more reliably than the overarching problem. Our [research paper](https://arxiv.org/abs/2212.14024) show this can easily lead to up to 120% gains over GPT-3.

Unlike vanilla in-context learning, you do not hard-code any few-shot prompts in **DSP**. We discourage "prompt engineering" and view it much the same way as hyperparameter tuning in traditional ML: it's a final (and relatively minor) step that's best done _after_ building up an effective architecture and connecting its components together.

Instead, **DSP** offers a number of powerful primitives and systematic design patterns for building these architectures — using LMs and search. The **DSP** primitives get the transformations you build working together on your behalf. For instance, **DSP** *annotates* few-shot demonstrations for the LM calls within your arbitrary pipeline automatically, and uses them to improve the quality of your transformations. Once you're happy with things, **DSP** can *compile* your program into a much smaller LM that's much cheaper to deploy.


<p align="center">
  <img align="center" src="docs/images/DSP-tasks.png" width="460px" />
</p>
<p align="left">
  <b>Figure 1:</b> A comparison between three GPT3.5-based systems. The LM often makes false assertions, while the popular retrieve-then-read pipeline fails when simple search can’t find an answer. In contrast, a task-aware DSP program systematically decomposes the problem and produces a correct response. Texts edited for presentation.
</p>


## Installation

```pip install dsp-ml```

## 🏃 Getting Started

Our [intro notebook](intro.ipynb) provides examples of five "multi-hop" question answering programs of increasing complexity written in DSP.

You can [open the intro notebook in Google Colab](https://colab.research.google.com/github/stanfordnlp/dsp/blob/main/intro.ipynb). You don't even need an API key to get started with it.

Once you go through the notebook, you'll be ready to create your own DSP pipelines!
&nbsp;

<p align="center">
  <img align="center" src="docs/images/DSP-example.png" width="850px" />
</p>
<p align="left">
  <b>Figure 2:</b> A DSP program for multi-hop question answering, given an input question and a 2-shot training set. The Demonstrate stage programmatically annotates intermediate transformations on the training examples. Learning from the resulting demonstration, the Search stage decomposes the complex input question and retrieves supporting information over two hops. The Predict stage uses the retrieved passages to answer the question.
</p>


## ⚡️ DSP Compiler [NEW!]

Our [compiler notebook](compiler.ipynb) introduces the new experimental compiler, which can optimize DSP programs automatically for (much) cheaper execution.

You can [open the compiler notebook in Google Colab](https://colab.research.google.com/github/stanfordnlp/dsp/blob/main/compiler.ipynb). You don't even need an API key to get started with it.

## Picking in-context examples using KNN/ANN methods [NEW!]

Our [knn demo notebook](tests/knn_demonstrations_test.ipynb) provides examples of adding the KNN stage, as described in the paper. This improvement in the Demonstrate stage of DSP allows you not to sample Examples randomly but instead search for better and similar options. You can get an idea from [this paper](https://arxiv.org/abs/2101.06804).

## 📜 Reading More

You can get an overview via our Twitter threads:
* [**Introducing DSP**](https://twitter.com/lateinteraction/status/1617953413576425472)  (Jan 24, 2023)
* [**Releasing the DSP Compiler (v0.1)**](https://twitter.com/lateinteraction/status/1625231662849073160)  (Feb 13, 2023)

And read more in the academic paper:
* [**Demonstrate-Search-Predict: Composing retrieval and language models for knowledge-intensive NLP**](https://arxiv.org/abs/2212.14024.pdf)

## ✍️ Reference

If you use DSP in a research paper, please cite our work as follows:

```
@article{khattab2022demonstrate,
  title={Demonstrate-Search-Predict: Composing Retrieval and Language Models for Knowledge-Intensive {NLP}},
  author={Khattab, Omar and Santhanam, Keshav and Li, Xiang Lisa and Hall, David and Liang, Percy and Potts, Christopher and Zaharia, Matei},
  journal={arXiv preprint arXiv:2212.14024},
  year={2022}
}
```
