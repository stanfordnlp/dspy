---
sidebar_position: 10
---

# dspy.CloudflareAI

### Usage

```python
lm = dspy.CloudflareAI(model="@hf/meta-llama/meta-llama-3-8b-instruct")
```

### Constructor

The constructor initializes the base class `LM` and verifies the `api_key` and `account_id` for using Cloudflare AI API.
The following environment variables are expected to be set or passed as arguments:

- `CLOUDFLARE_ACCOUNT_ID`: Account ID for Cloudflare.
- `CLOUDFLARE_API_KEY`: API key for Cloudflare.

```python
class CloudflareAI(LM):
  def __init__(
          self,
          model: str = "@hf/meta-llama/meta-llama-3-8b-instruct",
          account_id: Optional[str] = None,
          api_key: Optional[str] = None,
          system_prompt: Optional[str] = None,
          **kwargs,
      ):
```

**Parameters:**

- `model` (_str_): Model hosted on Cloudflare. Defaults to `@hf/meta-llama/meta-llama-3-8b-instruct`.
- `account_id` (_Optional[str]_, _optional_): Account ID for Cloudflare. Defaults to None. Reads from environment variable `CLOUDFLARE_ACCOUNT_ID`.
- `api_key` (_Optional[str]_, _optional_): API key for Cloudflare. Defaults to None. Reads from environment variable `CLOUDFLARE_API_KEY`.
- `system_prompt` (_Optional[str]_, _optional_): System prompt to use for generation.

### Methods

Refer to [`dspy.OpenAI`](https://dspy-docs.vercel.app/api/language_model_clients/OpenAI) documentation.
