from typing import Any, Optional

import pydantic

from dspy.adapters.types.base_type import Type
from dspy.lm15 import CitationPart, Response
from dspy.utils.annotation import experimental


@experimental(version="3.0.4")
class Citations(Type):
    """Citations extracted from an LM response with source references.

    This type represents citations returned by language models that support
    citation extraction, particularly Anthropic's Citations API through LiteLLM.
    Citations include the quoted text and source information.

    Examples:
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
        """One citation: the quoted text plus whatever the source reported.

        Native citations arrive as lm15 `CitationPart`s carrying text, title
        and URL. Character offsets and document indexes are optional because
        not every provider reports them.
        """

        type: str = "char_location"
        cited_text: str
        document_index: int | None = None
        document_title: str | None = None
        url: str | None = None
        start_char_index: int | None = None
        end_char_index: int | None = None
        supported_text: str | None = None

        def format(self) -> dict[str, Any]:
            """The citation as a JSON object, the shape the model is asked to write."""
            return {key: value for key, value in self.__dict__.items() if value is not None}

        @classmethod
        def from_part(cls, part: CitationPart) -> "Citations.Citation":
            return cls(cited_text=part.text or part.title or part.url or "", document_title=part.title, url=part.url)

    citations: list[Citation]

    @classmethod
    def from_dict_list(cls, citations_dicts: list[dict[str, Any]]) -> "Citations":
        """Convert a list of dictionaries to a Citations instance.

        Args:
            citations_dicts: A list of dictionaries, where each dictionary should have 'cited_text' key
                and 'document_index', 'start_char_index', 'end_char_index' keys.

        Returns:
            A Citations instance.

        Examples:
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

    def format(self) -> str:
        """Citations as a JSON list, the shape `parse_value` reads back."""
        import json

        return json.dumps([citation.format() for citation in self.citations], ensure_ascii=False)

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any):
        if isinstance(data, cls):
            return data

        # Handle case where data is a list of dicts with citation info
        if isinstance(data, list) and all(isinstance(item, dict) and "cited_text" in item for item in data):
            return {"citations": [cls.Citation(**item) for item in data]}

        # Handle case where data is a dict
        elif isinstance(data, dict):
            if "citations" in data:
                # Handle case where data is a dict with "citations" key
                citations_data = data["citations"]
                if isinstance(citations_data, list):
                    return {
                        "citations": [
                            cls.Citation(**item) if isinstance(item, dict) else item for item in citations_data
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
    def adapt_to_native_lm_feature(cls, signature, field_name, lm, lm_kwargs):
        # Native citations need a provider-side opt-in on document parts, which
        # lm15's DocumentPart cannot express yet. The field therefore stays in
        # the prompt and is parsed from the answer; citation parts a provider
        # returns anyway are still read in `parse_lm_response`.
        return signature

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
            # Listener chunks carry a streamed lm15 citation part under provider_specific_fields.
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "provider_specific_fields") and delta.provider_specific_fields:
                    citation_data = delta.provider_specific_fields.get("citation")
                    if citation_data:
                        part = CitationPart(
                            text=citation_data.get("text"), title=citation_data.get("title"), url=citation_data.get("url"),
                        )
                        return cls(citations=[cls.Citation.from_part(part)])
        except Exception:
            pass
        return None

    @classmethod
    def parse_lm_response(cls, response: Response) -> Optional["Citations"]:
        """Read the citation parts of an lm15 `Response` into Citations."""
        parts = response.citations
        if not parts:
            return None
        return cls(citations=[cls.Citation.from_part(part) for part in parts])
