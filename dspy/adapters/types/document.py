from typing import Any, Literal

import pydantic

from dspy.adapters.types.base_type import Type


class Document(Type):
    """A document type for providing content that can be cited by language models.

    This type represents documents that can be passed to language models for citation-enabled
    responses, particularly useful with Anthropic's Citations API. Documents include the content
    and metadata that helps the LM understand and reference the source material.

    Attributes:
        data: The text content of the document
        title: Optional title for the document (used in citations)
        media_type: MIME type of the document content (defaults to "text/plain")
        metadata: Optional additional metadata about the document
        context: Optional context information about the document

    Example:
        ```python
        import dspy
        from dspy.signatures import Signature

        class AnswerWithSources(Signature):
            '''Answer questions using provided documents with citations.'''
            documents: list[dspy.Document] = dspy.InputField()
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()
            citations: dspy.Citations = dspy.OutputField()

        # Create documents
        docs = [
            dspy.Document(
                data="The Earth orbits the Sun in an elliptical path.",
                title="Basic Astronomy Facts"
            ),
            dspy.Document(
                data="Water boils at 100°C at standard atmospheric pressure.",
                title="Physics Fundamentals",
                metadata={"author": "Dr. Smith", "year": 2023}
            )
        ]

        # Use with a citation-supporting model
        lm = dspy.LM("anthropic/claude-opus-4-1-20250805")
        predictor = dspy.Predict(AnswerWithSources)
        result = predictor(documents=docs, question="What temperature does water boil?", lm=lm)
        print(result.citations)
        ```
    """

    data: str
    title: str | None = None
    media_type: Literal["text/plain", "application/pdf"] = "text/plain"
    metadata: dict[str, Any] | None = None
    context: str | None = None

    def format(self) -> list[dict[str, Any]]:
        """Format document for LM consumption.

        Returns:
            A list containing the document block in the format expected by citation-enabled language models.
        """
        return {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": self.media_type,
                "data": self.data
            },
            "citations": {"enabled": True},
            "title": self.title
        }



    @classmethod
    def description(cls) -> str:
        """Description of the document type for use in prompts."""
        return (
            "A document containing text content that can be referenced and cited. "
            "Include the full text content and optionally a title for proper referencing."
        )

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any):
        if isinstance(data, cls):
            return data

        # Handle case where data is just a string (data only)
        if isinstance(data, str):
            return {"data": data}

        # Handle case where data is a dict
        elif isinstance(data, dict):
            return data

        raise ValueError(f"Received invalid value for `dspy.Document`: {data}")

    def __str__(self) -> str:
        """String representation showing title and content length."""
        title_part = f"'{self.title}': " if self.title else ""
        return f"Document({title_part}{len(self.data)} chars)"
