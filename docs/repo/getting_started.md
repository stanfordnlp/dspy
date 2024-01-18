## 💻 Installation

To install python packages, you need:

```python
pip install dspy-ai
```

Or open our intro notebook in Google Colab: [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/intro.ipynb)

!!! info
By default, DSPy depends on `openai==0.28`. However, if you install `openai>=1.0`, the library will use that just fine. Both are supported.

For the optional Pinecone, Qdrant, [chromadb](https://github.com/chroma-core/chroma), or [marqo](https://github.com/marqo-ai/marqo) retrieval integration(s), include the extra(s) below:

```python
pip install dspy-ai[pinecone]  # or [qdrant] or [chromadb] or [marqo] or [mongodb]
```

## ℹ️ Examples

The DSPy team believes complexity has to be justified. We take this seriously: we never release a complex tutorial (above) or example (below) _unless we can demonstrate empirically that this complexity has generally led to improved quality or cost._ This kind of rule is rarely enforced by other frameworks or docs, but you can count on it in DSPy examples.

There's a bunch of examples in the `examples/` directory and in the top-level directory. We welcome contributions!

You can find other examples tweeted by [@lateinteraction](https://twitter.com/lateinteraction) on Twitter/X.

## 🔍 Detailed Tutorials

If you're new to DSPy, it's probably best to go in sequential order. You will probably refer to these guides frequently after that, e.g. to copy/paste snippets that you can edit for your own DSPy programs.

1. **[DSPy Signatures](docs/guides/signatures.ipynb)**

2. **[Language Models](docs/guides/language_models.ipynb)** and **[Retrieval Models](docs/guides/retrieval_models.ipynb)**

3. **[DSPy Modules](docs/guides/modules.ipynb)**

4. **[DSPy Optimizers](docs/guides/optimizers.ipynb)**

5. **[DSPy Metrics](docs/guides/metrics.ipynb)**

6. **[DSPy Assertions](docs/guides/assertions.ipynb)**
