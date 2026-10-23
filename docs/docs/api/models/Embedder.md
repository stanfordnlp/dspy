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
and their provider-specific kwargs are removed. For a provider with another protocol or
authentication flow, pass a callable that accepts a list of strings and returns
one embedding vector per input, for example `dspy.Embedder(model.encode)` for a
local embedding model. LM provider registration does not register an embeddings
backend. See the [migration guide](../../community/normalized-lm-api-migration.md).

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
