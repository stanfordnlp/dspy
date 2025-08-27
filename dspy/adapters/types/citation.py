from typing import Any

import pydantic

from dspy.adapters.types.base_type import Type


class Citations(Type):
    """Citations extracted from an LM response with source references.

    This type represents citations returned by language models that support
    citation extraction, particularly Anthropic's Citations API through LiteLLM.
    Citations include the quoted text and source information.

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

        # Create documents to provide as sources
        docs = [
            dspy.Document(
                content="The Earth orbits the Sun in an elliptical path.",
                title="Basic Astronomy Facts"
            ),
            dspy.Document(
                content="Water boils at 100°C at standard atmospheric pressure.",
                title="Physics Fundamentals",
                metadata={"author": "Dr. Smith", "year": 2023}
            )
        ]

        # Use with a model that supports citations like Claude
        lm = dspy.LM("anthropic/claude-3-5-sonnet-20241022")
        predictor = dspy.Predict(AnswerWithSources, lm=lm)
        result = predictor(documents=docs, question="What temperature does water boil?")

        for citation in result.citations.citations:
            print(citation.format())
        ```
    """

    class Citation(Type):
        """Individual citation with text and source information."""
        text: str
        source: str | dict[str, Any] | None = None
        start: int | None = None
        end: int | None = None

        def format(self) -> str:
            """Format citation as a readable string.

            Returns:
                A formatted citation string showing the quoted text and source.
            """
            source_info = ""
            if self.source:
                if isinstance(self.source, str):
                    source_info = f" ({self.source})"
                elif isinstance(self.source, dict):
                    title = self.source.get("title", "")
                    url = self.source.get("url", "")
                    if title and url:
                        source_info = f" ({title}: {url})"
                    elif title:
                        source_info = f" ({title})"
                    elif url:
                        source_info = f" ({url})"

            return f'"{self.text}"{source_info}'

    citations: list[Citation]

    @classmethod
    def from_dict_list(cls, citations_dicts: list[dict[str, Any]]) -> "Citations":
        """Convert a list of dictionaries to a Citations instance.

        Args:
            citations_dicts: A list of dictionaries, where each dictionary should have 'text' key
                and optionally 'source', 'start', and 'end' keys.

        Returns:
            A Citations instance.

        Example:
            ```python
            citations_dict = [
                {"text": "The sky is blue", "source": "Weather Guide"},
                {"text": "Water boils at 100°C", "source": {"title": "Physics Book", "url": "http://example.com"}}
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
        return [
            {
                "text": citation.text,
                "source": citation.source,
                "start": citation.start,
                "end": citation.end,
            }
            for citation in self.citations
        ]

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any):
        if isinstance(data, cls):
            return data

        # Handle case where data is a list of dicts with citation info
        if isinstance(data, list) and all(
            isinstance(item, dict) and "text" in item for item in data
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
            elif "text" in data:
                # Handle case where data is a single citation dict
                return {"citations": [cls.Citation(**data)]}

        raise ValueError(f"Received invalid value for `dspy.Citations`: {data}")

    def __iter__(self):
        """Allow iteration over citations."""
        return iter(self.citations)

    def __len__(self):
        """Return the number of citations."""
        return len(self.citations)

    def __getitem__(self, index):
        """Allow indexing into citations."""
        return self.citations[index]
