"""AWS providers for LMs."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import backoff
from dsp.utils.settings import settings

try:
    import boto3
    from botocore.exceptions import ClientError
    ERRORS = (ClientError,)

except ImportError:
    ERRORS = (Exception,)


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/."""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )


def giveup_hdlr(details):
    """Wrapper function that decides when to give up on retry."""
    if "max retries" in details.args[0]:
        return False
    return True

class AWSProvider(ABC):
    """This abstract class adds support for AWS model providers such as Bedrock and SageMaker.
    The subclasses such as Bedrock and Sagemaker implement the abstract method _call_model and work in conjunction with the AWSModel classes.
    Usage Example:
        bedrock = dspy.Bedrock(region_name="us-west-2")
        bedrock_mixtral = dspy.AWSMistral(bedrock, "mistral.mixtral-8x7b-instruct-v0:1", **kwargs)
        bedrock_haiku = dspy.AWSAnthropic(bedrock, "anthropic.claude-3-haiku-20240307-v1:0", **kwargs)
        bedrock_llama2 = dspy.AWSMeta(bedrock, "meta.llama2-13b-chat-v1", **kwargs)

        sagemaker = dspy.Sagemaker(region_name="us-west-2")
        sagemaker_model = dspy.AWSMistral(sagemaker, "<YOUR_ENDPOINT_NAME>", **kwargs)
    """

    def __init__(
        self,
        region_name: str,
        service_name: str,
        profile_name: Optional[str] = None,
        batch_n_enabled: bool = True,
    ) -> None:
        """_summary_.

        Args:
            region_name (str, optional): The AWS region where this LM is hosted.
            service_name (str): Used in context of invoking the boto3 API.
            profile_name (str, optional): boto3 credentials profile.
            batch_n_enabled (bool): If False, call the LM N times rather than batching.
        """
        try:
            import boto3
        except ImportError as exc:
            raise ImportError('pip install boto3 to use AWS models.') from exc

        if profile_name is None:
            self.predictor = boto3.client(service_name, region_name=region_name)
        else:
            self.predictor = boto3.Session(profile_name=profile_name).client(
                service_name,
                region_name=region_name,
            )

        self.batch_n_enabled = batch_n_enabled

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return self.__class__.__name__

    @abstractmethod
    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=settings.backoff_time,
        max_tries=8,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def call_model(self, model_id: str, body: str) -> str:
        """Call the model and return the response."""

    def sanitize_kwargs(self, query_kwargs: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Ensure that input kwargs can be used by Bedrock or Sagemaker."""
        if "temperature" in query_kwargs:
            if query_kwargs["temperature"] > 0.99:
                query_kwargs["temperature"] = 0.99
            if query_kwargs["temperature"] < 0.01:
                query_kwargs["temperature"] = 0.01

        if "top_p" in query_kwargs:
            if query_kwargs["top_p"] > 0.99:
                query_kwargs["top_p"] = 0.99
            if query_kwargs["top_p"] < 0.01:
                query_kwargs["top_p"] = 0.01

        n = -1
        if not self.batch_n_enabled:
            n = query_kwargs.pop('n', 1)
            query_kwargs["num_generations"] = n

        return n, query_kwargs


class Bedrock(AWSProvider):
    """This class adds support for Bedrock models."""

    def __init__(
        self,
        region_name: str,
        profile_name: Optional[str] = None,
        batch_n_enabled: bool = False,   # This has to be setup manually on Bedrock.
    ) -> None:
        """_summary_.

        Args:
            region_name (str): The AWS region where this LM is hosted.
            profile_name (str, optional): boto3 credentials profile.
        """
        super().__init__(region_name, "bedrock-runtime", profile_name, batch_n_enabled)

    def call_model(self, model_id: str, body: str) -> str:
        return self.predictor.invoke_model(
            modelId=model_id,
            body=body,
            accept="application/json",
            contentType="application/json",
        )


class Sagemaker(AWSProvider):
    """This class adds support for Sagemaker models."""

    def __init__(
        self,
        region_name: str,
        profile_name: Optional[str] = None,
    ) -> None:
        """_summary_.

        Args:
            region_name (str, optional): The AWS region where this LM is hosted.
            profile_name (str, optional): boto3 credentials profile.
        """
        super().__init__(region_name, "runtime.sagemaker", profile_name)

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=settings.backoff_time,
        max_tries=8,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def call_model(self, model_id: str, body: str) -> str:
        return self.predictor.invoke_endpoint(
            EndpointName=model_id,
            Body=body,
            Accept="application/json",
            ContentType="application/json",
        )
