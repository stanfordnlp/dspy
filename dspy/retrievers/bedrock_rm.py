"""Amazon Bedrock Knowledge Base retriever for DSPy."""

import os

from dspy.dsp.utils import dotdict
from dspy.primitives.prediction import Prediction
from dspy.retrievers.retrieve import Retrieve


def _get_source_uri(result: dict) -> str:
    """Extract source URI from a retrieval result, handling all location types."""
    location = result.get("location", {})
    loc_type = location.get("type", "")
    if loc_type == "S3" or "s3Location" in location:
        return location.get("s3Location", {}).get("uri", "")
    elif loc_type == "WEB" or "webLocation" in location:
        return location.get("webLocation", {}).get("url", "")
    elif "confluenceLocation" in location:
        return location.get("confluenceLocation", {}).get("url", "")
    elif "salesforceLocation" in location:
        return location.get("salesforceLocation", {}).get("url", "")
    elif "sharePointLocation" in location:
        return location.get("sharePointLocation", {}).get("url", "")
    elif "customDocumentLocation" in location:
        return location.get("customDocumentLocation", {}).get("id", "")
    # Fallback to metadata._source_uri (for agentic results)
    return result.get("metadata", {}).get("_source_uri", "")


class BedrockRM(Retrieve):
    """A retrieval module that uses Amazon Bedrock Knowledge Bases.

    Supports Amazon Bedrock Managed Knowledge Bases (no vector store needed).

    Args:
        knowledge_base_id: The ID of the Bedrock Knowledge Base.
        region_name: AWS region. Defaults to AWS_REGION env var or us-east-1.
        k: Number of top passages to retrieve. Defaults to 3.

    Examples:
        ```python
        import dspy

        retriever = BedrockRM(knowledge_base_id="ABCDEFGHIJ")
        dspy.configure(rm=retriever)

        retrieve = dspy.Retrieve(k=5)
        results = retrieve("What is retrieval augmented generation?").passages
        ```
    """

    def __init__(
        self,
        knowledge_base_id: str | None = None,
        region_name: str | None = None,
        k: int = 3,
        use_agentic_retrieval: bool | None = None,
    ):
        self.knowledge_base_id = knowledge_base_id or os.environ.get("KNOWLEDGE_BASE_ID")
        self.region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
        self.use_agentic_retrieval = (
            use_agentic_retrieval
            if use_agentic_retrieval is not None
            else os.environ.get("USE_AGENTIC_RETRIEVAL", "true").lower() != "false"
        )
        self._client = None
        super().__init__(k=k)

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError:
                raise ImportError("boto3 is required to use BedrockRM. Install it with `pip install boto3>=1.43.2`")
            self._client = boto3.client(
                "bedrock-agent-runtime",
                region_name=self.region_name,
                config=Config(user_agent_extra="dspy/bedrock-kb"),
            )
        return self._client

    def forward(self, query_or_queries: str | list[str], k: int | None = None, **kwargs) -> Prediction:
        """Retrieve passages from Amazon Bedrock Knowledge Base.

        Args:
            query_or_queries: The query or queries to search for.
            k: Number of top passages to retrieve. Defaults to self.k.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        k = k if k is not None else self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]

        client = self._get_client()
        all_passages = []

        for query in queries:
            retrieval_config = self._build_retrieval_config(k)

            # Try agentic retrieval first
            if self.use_agentic_retrieval:
                try:
                    response = client.agentic_retrieve_stream(
                        knowledgeBaseId=self.knowledge_base_id,
                        messages=[{"content": {"text": query}, "role": "user"}],
                        retrievers=[
                            {
                                "configuration": {
                                    "knowledgeBase": {
                                        "knowledgeBaseId": self.knowledge_base_id,
                                        "retrievalOverrides": {"maxNumberOfResults": k},
                                    }
                                }
                            }
                        ],
                        agenticRetrieveConfiguration={
                            "foundationModelType": "MANAGED",
                            "rerankingModelType": "MANAGED",
                        },
                    )
                    passages = []
                    for event in response.get("stream", []):
                        if "result" in event and "results" in event["result"]:
                            for result in event["result"]["results"]:
                                text = result.get("content", {}).get("text", "")
                                source = _get_source_uri(result)
                                score = result.get("score", 0.0)
                                passages.append(
                                    dotdict(
                                        {
                                            "long_text": text,
                                            "source": source,
                                            "score": score,
                                        }
                                    )
                                )
                    if passages:
                        all_passages.extend(passages)
                        continue
                except Exception:
                    pass  # Fall through to plain retrieve

            try:
                response = client.retrieve(
                    knowledgeBaseId=self.knowledge_base_id,
                    retrievalQuery={"text": query},
                    retrievalConfiguration=retrieval_config,
                )
                results = response.get("retrievalResults", [])
                for result in results:
                    text = result.get("content", {}).get("text", "")
                    source = _get_source_uri(result)
                    score = result.get("score", 0.0)
                    all_passages.append(
                        dotdict(
                            {
                                "long_text": text,
                                "source": source,
                                "score": score,
                            }
                        )
                    )
            except Exception as e:
                raise RuntimeError(f"Error retrieving from Bedrock KB: {e}") from e

        return all_passages

    def _build_retrieval_config(self, k: int) -> dict:
        """Build the retrieval configuration based on KB type."""
        return {"managedSearchConfiguration": {"numberOfResults": k}}
