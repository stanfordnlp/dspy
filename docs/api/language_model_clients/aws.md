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

The `AWSMistral` constructor initializes the base class `AWSModel` which itself inherits from the `LM` class.

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
- `aws_provider` (AWSProvider): The aws provider to use. One of `dspy.Bedrock` or `dspy.Sagemaker`.
- `model` (_str_): Mistral AI pretrained models. For Bedrock, this is the Model ID in https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns. For Sagemaker, this is the endpoint name.
- `max_context_size` (_Optional[int]_, _optional_): Max context size for this model. Defaults to 32768.
- `max_new_tokens` (_Optional[int]_, _optional_): Max new tokens possible for this model. Defaults to 1500.
- `**kwargs`: Additional language model arguments to pass to the API provider.

### Methods

```python
def _format_prompt(self, raw_prompt: str) -> str:
```
This function formats the prompt for the model. Refer to the model card for the specific formatting required.

<br/>

```python
def _create_body(self, prompt: str, **kwargs) -> tuple[int, dict[str, str | float]]:
```
This function creates the body of the request to the model. It takes the prompt and any additional keyword arguments and returns a tuple of the number of tokens to generate and a dictionary of keys including the prompt used to create the body of the request.

<br/>

```python
def _call_model(self, body: str) -> str:
```
This function calls the model using the provider `call_model()` function and extracts the generated text (completion) from the provider-specific response.

<br/>

The above model-specific methods are called by the `AWSModel::basic_request()` method, which is the main method for querying the model. This method takes the prompt and any additional keyword arguments and calls the `AWSModel::_simple_api_call()` which then delegates to the model-specific `_create_body()` and `_call_model()` methods to create the body of the request, call the model and extract the generated text.


Refer to [`dspy.OpenAI`](https://dspy-docs.vercel.app/api/language_model_clients/OpenAI) documentation for information on the `LM` base class functionality.

<br/>

`AWSAnthropic` and `AWSMeta` work exactly the same as `AWSMistral`.