---
sidebar_position: 9
---

# dsp.AWSMistral, dsp.AWSAnthropic, dsp.AWSMeta

### Usage

```python
# Notes:
# 1. Install boto3 to use AWS models.
# 2. Configure your AWS credentials with the AWS CLI before using these models

# initialize the bedrock aws provider
bedrock = dspy.Bedrock(region_name="us-west-2")
# For mixtral on Bedrock
lm = dspy.AWSMistral(bedrock, "mistral.mixtral-8x7b-instruct-v0:1", **kwargs)
# For haiku on Bedrock
lm = dspy.AWSAnthropic(bedrock, "anthropic.claude-3-haiku-20240307-v1:0", **kwargs)
# For llama2 on Bedrock
lm = dspy.AWSMeta(bedrock, "meta.llama2-13b-chat-v1", **kwargs)

# initialize the sagemaker aws provider
sagemaker = dspy.Sagemaker(region_name="us-west-2")
# For mistral on Sagemaker
# Note: you need to create a Sagemaker endpoint for the mistral model first
lm = dspy.AWSMistral(sagemaker, "<YOUR_MISTRAL_ENDPOINT_NAME>", **kwargs)

```

### Constructor

The constructor initializes the base class `LM` and the `AWSModel` class.

```python
class AWSMistral(AWSModel):
    """Mistral family of models."""

    def __init__(
        self,
        aws_provider: AWSProvider,
        model: str,
        max_context_size: int = 32768,
        max_new_tokens: int = 1500,
        **kwargs
    ) -> None:
```

**Parameters:**
- `aws_provider` (AWSProvider): The aws provider to use. One of `Bedrock` or `Sagemaker`.
- `model` (_str_): Mistral AI pretrained models. Defaults to `mistral-medium-latest`.
- `max_context_size` (_Optional[int]_, _optional_): Max context size for this model. Defaults to 32768.
- `max_new_tokens` (_Optional[int]_, _optional_): Max new tokens possible for this model. Defaults to 1500.
- `**kwargs`: Additional language model arguments to pass to the API provider.

### Methods

Refer to [`dspy.OpenAI`](https://dspy-docs.vercel.app/api/language_model_clients/OpenAI) documentation.


`AWSAnthropic` and `AWSMeta` work exactly the same as `AWSMistral`.