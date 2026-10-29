# dspy.Embedder

!!! note "Requires numpy"
    `dspy.Embedder` requires numpy. Install it with `pip install dspy[numpy]`.

## Hosted endpoints and custom embedding functions

A string model uses the OpenAI SDK against OpenAI, Azure OpenAI, or an
OpenAI-compatible embeddings endpoint. For example:

```python
import dspy

embedder = dspy.Embedder("openai/text-embedding-3-small", dimensions=512)
vectors = embedder(["hello", "world"])  # uses OPENAI_API_KEY

compatible = dspy.Embedder(
    "openai/your-org/your-embedding-model",
    api_base="https://your-provider.example/v1",
    api_key="YOUR_PROVIDER_KEY",
)
```

Bare OpenAI model names are also accepted. For compatible endpoints, only the
leading `openai/` is stripped, preserving an organization/model id. `base_url`
is an alias for `api_base`; the default endpoint can also come from
`OPENAI_BASE_URL` or `OPENAI_API_BASE`, with credentials in `OPENAI_API_KEY`.

Azure embeddings use `azure/<deployment>` and the Azure SDK client:

```python
azure = dspy.Embedder(
    "azure/your-embedding-deployment",
    api_base="https://your-resource.openai.azure.com",  # resource root, not /embeddings
    api_version="YOUR_SUPPORTED_API_VERSION",
    api_key="YOUR_AZURE_KEY",
)
```

For Azure, `azure_endpoint` can replace `api_base`; use `base_url` only for an
explicit SDK-style base path. Environment defaults are
`AZURE_OPENAI_ENDPOINT` / `AZURE_API_BASE`, `OPENAI_API_VERSION` /
`AZURE_API_VERSION`, and `AZURE_OPENAI_API_KEY` / `AZURE_API_KEY`. This embeddings
configuration is separate from native `dspy.LM`'s Azure v1 route.

This is not a multi-provider routing layer. Other LiteLLM embedding prefixes
and their provider-specific kwargs are removed. Installing `dspy[litellm]` enables
LM compatibility, not embedding fallback. For a provider with another protocol or
authentication flow, pass a callable that accepts a list of strings and returns
one embedding vector per input, for example `dspy.Embedder(model.encode)` for a
local embedding model. LM provider registration does not register an embeddings
backend. See the [migration guide](../../community/normalized-lm-api-migration.md).

## Retaining a Cohere retrieval index

Cohere's `search_document` and `search_query` modes are not interchangeable.
Its OpenAI-compatible endpoint does not support `input_type`, so changing only
the model prefix does not preserve this workflow. Keep the provider call in an
application-owned callable instead. For example, install LiteLLM separately
with `pip install litellm` and set `COHERE_API_KEY`:

```python
import dspy
import litellm
from dspy.retrievers import Embeddings


def cohere_vectors(texts, input_type):
    response = litellm.embedding(
        model="cohere/embed-english-v3.0",
        input=texts,
        input_type=input_type,
    )
    return [item["embedding"] for item in response.data]


embedder = dspy.Embedder(cohere_vectors, input_type="search_document")
retrieve = Embeddings(
    ["Paris is the capital of France", "Berlin is the capital of Germany"],
    embedder,
    k=1,
)
# Construction above embeds the corpus. Subsequent calls embed queries.
embedder.default_kwargs["input_type"] = "search_query"
result = retrieve("What is the capital of France?")
```

Keep a separate document-mode embedder if you also index new documents after
this switch; do not toggle shared defaults while serving concurrent requests.
Preserve the original model, dimensions, normalization and input modes when
using an existing index. The callable can instead use Cohere's SDK; DSPy does
not install or automatically select either dependency. DSPy still handles
batching and embedding caching, including `input_type` in callable kwargs.

<!-- START_API_REF -->
::: dspy.Embedder
    handler: python
    options:
        members:
            - __call__
            - acall
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
<!-- END_API_REF -->
