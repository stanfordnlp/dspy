---
sidebar_position: 9
---

# dspy.Bedrock, dspy.Sagemaker

### Usage

The `AWSProvider` class is the base class for the AWS providers - `dspy.Bedrock` and `dspy.Sagemaker`. An instance of one of these providers is passed to the constructor when creating an instance of an AWS model class (e.g., `dspy.AWSMistral`) that is ultimately used to query the model.

```python
# Notes:
# 1. Install boto3 to use AWS models.
# 2. Configure your AWS credentials with the AWS CLI before using these models

# initialize the bedrock aws provider
bedrock = dspy.Bedrock(region_name="us-west-2")

# initialize the sagemaker aws provider
sagemaker = dspy.Sagemaker(region_name="us-west-2")
```

### Constructor

The `Bedrock` constructor initializes the base class `AWSProvider`.

```python
class Bedrock(AWSProvider):
    """This class adds support for Bedrock models."""

    def __init__(
        self,
        region_name: str,
        profile_name: Optional[str] = None,
        batch_n_enabled: bool = False,   # This has to be setup manually on Bedrock.
    ) -> None:
```

**Parameters:**
- `region_name` (str): The AWS region where this LM is hosted.
- `profile_name` (str, optional): boto3 credentials profile.
- `batch_n_enabled` (bool): If False, call the LM N times rather than batching.

### Methods

```python
def call_model(self, model_id: str, body: str) -> str:
```
This function implements the actual invocation of the model on AWS using the boto3 provider.

<br/>

`Sagemaker` works exactly the same as `Bedrock`.