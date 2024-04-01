"""AWS providers for LMs."""

from abc import ABC, abstractmethod
from typing import Optional


class AWSProvider(ABC):
    """This abstract class adds support for AWS model providers such as Bedrock and SageMaker."""

    def __init__(
        self,
        region_name: str,
        service_name: str,
        profile_name: Optional[str] = None,
    ) -> None:
        """_summary_.

        Args:
            region_name (str, optional): The AWS region where this LM is hosted.
            service_name (str): Used in context of invoking the boto3 API.
            profile_name (str, optional): boto3 credentials profile.
        """
        import boto3  # pylint: disable=import-outside-toplevel

        if profile_name is None:
            self.predictor = boto3.client(service_name, region_name=region_name)
        else:
            self.predictor = boto3.Session(profile_name=profile_name).client(
                service_name,
                region_name=region_name,
            )

    @abstractmethod
    def call_model(self, model_id: str, body: str) -> str:
        """Call the model and return the response."""


class Bedrock(AWSProvider):
    """This class adds support for Bedrock models."""

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
        super().__init__(region_name, "bedrock-runtime", profile_name)

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

    def call_model(self, model_id: str, body: str) -> str:
        return self.predictor.invoke_endpoint(
            EndpointName=model_id,
            Body=body,
            Accept="application/json",
            ContentType="application/json",
        )
