from typing import Any, Literal

import pydantic

from dspy._vendor.lm15.types import document
from dspy.adapters.types.base_type import Type
from dspy.lm15 import DocumentPart, TextPart
from dspy.utils.annotation import experimental


@experimental(version="3.0.4")
class Document(Type):
    """A document type for providing content that can be cited by language models.

    This type represents documents that can be passed to language models for citation-enabled
    responses, particularly useful with Anthropic's Citations API. Documents include the content
    and metadata that helps the LM understand and reference the source material.

    Attributes:
        data: The text content of the document
        title: Optional title for the document (used in citations)
        media_type: MIME type of the document content (defaults to "text/plain")
        context: Optional context information about the document

    Examples:
        ```python
        import dspy
        from dspy.signatures import Signature
        from dspy.experimental import Document, Citations

        class AnswerWithSources(Signature):
            '''Answer questions using provided documents with citations.'''
            documents: list[Document] = dspy.InputField()
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()
            citations: Citations = dspy.OutputField()

        # Create documents
        docs = [
            Document(
                data="The Earth orbits the Sun in an elliptical path.",
                title="Basic Astronomy Facts"
            ),
            Document(
                data="Water boils at 100°C at standard atmospheric pressure.",
                title="Physics Fundamentals",
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
    context: str | None = None

    def format(self) -> list[TextPart | DocumentPart]:
        """Render the document as lm15 content parts.

        Plain-text documents become text the model reads directly, framed by
        their title and context. PDFs become one document part. Provider-side
        citation marking is not part of lm15's document part yet, so citations
        are produced by parsing the model's answer rather than a native API.
        """
        header = []
        if self.title:
            header.append(f"Title: {self.title}")
        if self.context:
            header.append(f"Context: {self.context}")
        parts = [TextPart("\n".join(header) + "\n")] if header else []
        if self.media_type == "application/pdf":
            parts.append(document(data=self.data, media_type=self.media_type))
        else:
            parts.append(TextPart(self.data))
        return parts

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

        raise ValueError(f"Received invalid value for `Document`: {data}")

    def __str__(self) -> str:
        """String representation showing title and content length."""
        title_part = f"'{self.title}': " if self.title else ""
        return f"Document({title_part}{len(self.data)} chars)"
