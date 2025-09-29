from typing import Any, Optional

import pydantic

from dspy.adapters.types.base_type import Type
from dspy.utils.annotation import experimental


@experimental(version="3.0.4")
class Citations(Type):
    """Citations extracted from an LM response with source references.

    This type represents citations returned by language models that support
    citation extraction, particularly Anthropic's Citations API through LiteLLM.
    Citations include the quoted text and source information.

    Example:
        ```python
        import os
        import dspy
        from dspy.signatures import Signature
        from dspy.experimental import Citations, Document
        os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"

        class AnswerWithSources(Signature):
            '''Answer questions using provided documents with citations.'''
            documents: list[Document] = dspy.InputField()
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()
            citations: Citations = dspy.OutputField()

        # Create documents to provide as sources
        docs = [
            Document(
                data="The Earth orbits the Sun in an elliptical path.",
                title="Basic Astronomy Facts"
            ),
            Document(
                data="Water boils at 100°C at standard atmospheric pressure.",
                title="Physics Fundamentals",
                metadata={"author": "Dr. Smith", "year": 2023}
            )
        ]

        # Use with a model that supports citations like Claude
        lm = dspy.LM("anthropic/claude-opus-4-1-20250805")
        predictor = dspy.Predict(AnswerWithSources)
        result = predictor(documents=docs, question="What temperature does water boil?", lm=lm)

        for citation in result.citations.citations:
            print(citation.format())
        ```
    """

    class Citation(Type):
        """Individual citation with character location information."""
        type: str = "char_location"
        cited_text: str
        document_index: int
        document_title: str | None = None
        start_char_index: int
        end_char_index: int
        supported_text: str | None = None

        def format(self) -> dict[str, Any]:
            """Format citation as dictionary for LM consumption.

            Returns:
                A dictionary in the format expected by citation APIs.
            """
            citation_dict = {
                "type": self.type,
                "cited_text": self.cited_text,
                "document_index": self.document_index,
                "start_char_index": self.start_char_index,
                "end_char_index": self.end_char_index
            }

            if self.document_title:
                citation_dict["document_title"] = self.document_title

            if self.supported_text:
                citation_dict["supported_text"] = self.supported_text

            return citation_dict

    citations: list[Citation]

    @classmethod
    def from_dict_list(cls, citations_dicts: list[dict[str, Any]]) -> "Citations":
        """Convert a list of dictionaries to a Citations instance.

        Args:
            citations_dicts: A list of dictionaries, where each dictionary should have 'cited_text' key
                and 'document_index', 'start_char_index', 'end_char_index' keys.

        Returns:
            A Citations instance.

        Example:
            ```python
            citations_dict = [
                {
                    "cited_text": "The sky is blue",
                    "document_index": 0,
                    "document_title": "Weather Guide",
                    "start_char_index": 0,
                    "end_char_index": 15,
                    "supported_text": "The sky was blue yesterday."
                }
            ]
            citations = Citations.from_dict_list(citations_dict)
            ```
        """
        citations = [cls.Citation(**item) for item in citations_dicts]
        return cls(citations=citations)

    @classmethod
    def description(cls) -> str:
        """Description of the citations type for use in prompts."""
        return (
            "Citations with quoted text and source references. "
            "Include the exact text being cited and information about its source."
        )

    def format(self) -> list[dict[str, Any]]:
        """Format citations as a list of dictionaries."""
        return [citation.format() for citation in self.citations]

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any):
        if isinstance(data, cls):
            return data

        # Handle case where data is a list of dicts with citation info
        if isinstance(data, list) and all(
            isinstance(item, dict) and "cited_text" in item for item in data
        ):
            return {"citations": [cls.Citation(**item) for item in data]}

        # Handle case where data is a dict
        elif isinstance(data, dict):
            if "citations" in data:
                # Handle case where data is a dict with "citations" key
                citations_data = data["citations"]
                if isinstance(citations_data, list):
                    return {
                        "citations": [
                            cls.Citation(**item) if isinstance(item, dict) else item
                            for item in citations_data
                        ]
                    }
            elif "cited_text" in data:
                # Handle case where data is a single citation dict
                return {"citations": [cls.Citation(**data)]}

        raise ValueError(f"Received invalid value for `Citations`: {data}")

    def __iter__(self):
        """Allow iteration over citations."""
        return iter(self.citations)

    def __len__(self):
        """Return the number of citations."""
        return len(self.citations)

    def __getitem__(self, index):
        """Allow indexing into citations."""
        return self.citations[index]

    @classmethod
    def is_streamable(cls) -> bool:
        """Whether the Citations type is streamable."""
        return True

    @classmethod
    def parse_stream_chunk(cls, chunk) -> Optional["Citations"]:
        """
        Parse a stream chunk into Citations.

        Args:
            chunk: A stream chunk from the LM.

        Returns:
            A Citations object if the chunk contains citation data, None otherwise.
        """
        try:
            # Check if the chunk has citation data in provider_specific_fields
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "provider_specific_fields") and delta.provider_specific_fields:
                    citation_data = delta.provider_specific_fields.get("citation")
                    if citation_data:
                        return cls.from_dict_list([citation_data])
        except Exception:
            pass
        return None


    @classmethod
    def parse_lm_response(cls, response: str | dict[str, Any]) -> Optional["Citations"]:
        """Parse a LM response into Citations.

        Args:
            response: A LM response that may contain citation data.

        Returns:
            A Citations object if citation data is found, None otherwise.
        """
        if isinstance(response, dict):
            # Check if the response contains citations in the expected format
            if "citations" in response:
                citations_data = response["citations"]
                if isinstance(citations_data, list):
                    return cls.from_dict_list(citations_data)

        return None
