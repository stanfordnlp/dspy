---
sidebar_position: 1
---

# Installation

To install DSPy run:


```text
pip install dspy-ai
```

Or open our intro notebook in Google Colab: [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/intro.ipynb)

By default, DSPy depends on `openai==0.28`. However, if you install `openai>=1.0`, the library will use that just fine. Both are supported.

For the optional LanceDB, Pinecone, Qdrant, ChromaDB, Marqo, or Milvus retrieval integration(s), include the extra(s) below:

!!! info "Installation Command"

    === "No Extras"
        ```markdown
        pip install dspy-ai
        ```

    === "LanceDB"
        ```markdown
        pip install dspy-ai[lancedb]
        ```

    === "Pinecone"
        ```markdown
        pip install "dspy-ai[pinecone]"
        ```

    === "Qdrant"
        ```markdown
        pip install "dspy-ai[qdrant]"
        ```

    === "ChromaDB"
        ```markdown
        pip install "dspy-ai[chromadb]"
        ```

    === "Marqo"
        ```markdown
        pip install "dspy-ai[marqo]"
        ```

    === "MongoDB"
        ```markdown
        pip install "dspy-ai[mongodb]"
        ```

    === "Weaviate"
        ```markdown
        pip install "dspy-ai[weaviate]"
        ```

    === "Milvus"
        ```markdown
        pip install "dspy-ai[milvus]"
        ```
