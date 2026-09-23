"""
Chain of Density (CoD) Module for DSPy

This module implements the Chain of Density summarization technique from:
"From Sparse to Dense: GPT-4 Summarization with Chain of Density Prompting"
(Adams et al., 2023) - https://arxiv.org/abs/2309.04269

Chain of Density generates increasingly entity-dense summaries through iterative
refinement, starting with an entity-sparse summary and progressively incorporating
missing salient entities without increasing the summary length.
"""

from typing import Any, Callable

from dspy.clients.base_lm import BaseLM
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures import InputField, OutputField
from dspy.signatures.signature import Signature


class InitialSummary(Signature):
    """Generate an initial entity-sparse summary of the article.

    The summary should be comprehensive but use general language rather than
    specific entities. Focus on the main topic and key points without including
    too many specific names, numbers, or technical terms.
    """

    article: str = InputField(desc="The article or document to summarize")
    target_length: int = InputField(desc="Target length of the summary in words (approximately)")
    summary: str = OutputField(desc="An initial entity-sparse summary that covers the main topic broadly")


class IdentifyMissingEntities(Signature):
    """Identify 1-3 informative entities missing from the current summary.

    A Missing Entity is:
    - Relevant: to the main story/topic
    - Specific: descriptive yet concise (5 words or fewer)
    - Novel: not already present in the current summary
    - Faithful: present in the original article
    """

    article: str = InputField(desc="The original article or document")
    current_summary: str = InputField(desc="The current summary to analyze")
    missing_entities: list[str] = OutputField(
        desc="1-3 informative entities from the article that are missing from the summary"
    )


class DensifySummary(Signature):
    """Write a denser summary incorporating the missing entities.

    Create a new summary that:
    1. Includes ALL information from the previous summary
    2. Incorporates the missing entities naturally
    3. Maintains approximately the same length as the previous summary
    4. Uses compression, fusion, and rewriting to make room for new entities

    The new summary should be more informative while remaining coherent and readable.
    """

    article: str = InputField(desc="The original article or document")
    previous_summary: str = InputField(desc="The previous summary to densify")
    missing_entities: list[str] = InputField(desc="The entities to incorporate into the new summary")
    target_length: int = InputField(desc="Target length of the summary in words (should match previous summary)")
    denser_summary: str = OutputField(
        desc="A denser summary that incorporates the missing entities while maintaining the same length"
    )


class DensificationStep(Signature):
    """Perform one step of Chain of Density: identify missing entities and create a denser summary.

    This combines entity identification and summary densification into a single step for efficiency.

    Process:
    1. Identify 1-3 informative entities missing from the current summary
    2. Rewrite the summary to incorporate these entities without increasing length
    3. Use compression, fusion, and abstraction to make room for new information
    """

    article: str = InputField(desc="The original article or document")
    previous_summary: str = InputField(desc="The previous summary to densify")
    target_length: int = InputField(desc="Target length of the summary in words (should match previous summary)")
    missing_entities: list[str] = OutputField(
        desc="1-3 informative entities from the article that were missing and are now included"
    )
    denser_summary: str = OutputField(
        desc="A denser summary that incorporates the missing entities while maintaining the same length"
    )


class ChainOfDensity(Module):
    """
    Chain of Density (CoD) summarization module.

    Generates increasingly entity-dense summaries through iterative refinement.
    Starting with an entity-sparse summary, it progressively incorporates missing
    salient entities without increasing the summary length.

    Based on: "From Sparse to Dense: GPT-4 Summarization with Chain of Density Prompting"
    (Adams et al., 2023)

    Args:
        num_steps: Number of densification iterations (default: 5)
        target_length: Target summary length in words (default: 80)
        return_intermediate: Whether to return all intermediate summaries (default: True).
            When False, `summaries` and `entity_density` only contain the selected summary.
        use_chain_of_thought: Whether to use CoT for densification steps (default: True)

    Example:
        ```python
        import dspy

        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        cod = dspy.ChainOfDensity(num_steps=5, target_length=80)

        article = '''
        Apple Inc. announced its latest iPhone 14 today at a press event in Cupertino.
        CEO Tim Cook highlighted the new features including improved camera capabilities,
        satellite connectivity for emergencies, and a longer battery life. The device
        will be available starting September 16th at a price of $799 for the base model.
        '''

        result = cod(article=article)

        print(result.final_summary)
        # Access all intermediate summaries
        for i, summary in enumerate(result.summaries):
            print(f"Step {i}: {summary}")
        ```
    """

    def __init__(
        self,
        num_steps: int = 5,
        target_length: int = 80,
        return_intermediate: bool = True,
        use_chain_of_thought: bool = True,
        **config: Any,
    ):
        super().__init__()

        if num_steps < 0:
            raise ValueError(f"num_steps must be non-negative, got {num_steps}.")

        self.num_steps = num_steps
        self.target_length = target_length
        self.return_intermediate = return_intermediate
        self.use_chain_of_thought = use_chain_of_thought

        predictor_cls = ChainOfThought if use_chain_of_thought else Predict
        self.initial_summarizer = predictor_cls(InitialSummary, **config)
        self.densifier = predictor_cls(DensificationStep, **config)

    def forward(self, article: str, **kwargs) -> Prediction:
        """
        Generate a chain of increasingly dense summaries.

        Args:
            article: The text to summarize
            **kwargs: Additional arguments passed to predictors

        Returns:
            Prediction with:
                - final_summary: The selected (by default, most dense) summary
                - summaries: List of summaries from sparse to dense
                - entities_added: List of entities added at each step
                - entity_density: Approximate entity density metrics, aligned with `summaries`
        """
        initial = self.initial_summarizer(article=article, target_length=self.target_length, **kwargs)
        summaries = [initial.summary]
        entities_added = []

        for _ in range(self.num_steps):
            step = self.densifier(
                article=article,
                previous_summary=summaries[-1],
                target_length=self.target_length,
                **kwargs,
            )
            summaries.append(step.denser_summary)
            entities_added.append(step.missing_entities)

        return self._build_prediction(summaries, entities_added)

    async def aforward(self, article: str, **kwargs) -> Prediction:
        """Async version of forward."""
        initial = await self.initial_summarizer.acall(article=article, target_length=self.target_length, **kwargs)
        summaries = [initial.summary]
        entities_added = []

        for _ in range(self.num_steps):
            step = await self.densifier.acall(
                article=article,
                previous_summary=summaries[-1],
                target_length=self.target_length,
                **kwargs,
            )
            summaries.append(step.denser_summary)
            entities_added.append(step.missing_entities)

        return self._build_prediction(summaries, entities_added)

    def _select_step(self, summaries: list[str], entities_added: list[list[str]]) -> int:
        """Return the index (into the full chain of summaries) of the summary to use as `final_summary`."""
        return len(summaries) - 1

    def _extra_fields(self, selected_step: int) -> dict[str, Any]:
        """Additional fields to include in the returned Prediction."""
        return {}

    def _build_prediction(self, summaries: list[str], entities_added: list[list[str]]) -> Prediction:
        # Selection always runs on the full chain so that `return_intermediate=False` cannot change which summary
        # is picked; trimming for the output happens afterwards.
        selected_step = self._select_step(summaries, entities_added)
        if not isinstance(selected_step, int) or not 0 <= selected_step < len(summaries):
            raise ValueError(f"Selected step must be an integer in [0, {len(summaries) - 1}], got {selected_step!r}.")

        final_summary = summaries[selected_step]
        if self.return_intermediate:
            returned_summaries = summaries
        else:
            returned_summaries = [final_summary]

        return Prediction(
            final_summary=final_summary,
            summaries=returned_summaries,
            entities_added=entities_added,
            entity_density=self._calculate_density(returned_summaries),
            **self._extra_fields(selected_step),
        )

    def _calculate_density(self, summaries: list[str]) -> list[dict]:
        """
        Calculate approximate entity density for each summary.

        This is a simplified calculation - for more accurate entity counting,
        consider using spaCy or similar NER tools.
        """
        densities = []
        for summary in summaries:
            words = summary.split()
            word_count = len(words)

            # Simple heuristic: count capitalized words as potential entities
            # (This is a rough approximation)
            potential_entities = sum(1 for word in words if word[0].isupper() and len(word) > 1)

            density = potential_entities / word_count if word_count > 0 else 0

            densities.append(
                {
                    "word_count": word_count,
                    "potential_entities": potential_entities,
                    "density": round(density, 4),
                }
            )

        return densities


class ChainOfDensityWithPreference(ChainOfDensity):
    """
    Extended Chain of Density that selects the best summary based on a preference model.

    According to the original paper, humans prefer summaries that are moderately dense
    (around step 3-4 out of 5), balancing informativeness with readability.

    This variant allows specifying a preferred density level or using a custom
    selection function. Selection is applied for both sync (`__call__`) and async
    (`acall`) invocations, and always operates on the full chain of summaries, even
    when `return_intermediate=False`.

    Args:
        preferred_step: Which step's summary to return as final (0=sparse, num_steps=densest)
                       If None, returns the last summary. Use 3-4 for balanced density.
                       Values larger than `num_steps` are clamped to the densest summary.
        selection_fn: Optional custom function to select the best summary. Takes precedence
                     over `preferred_step`. It always receives the full chain of summaries.
                     Signature: (summaries: list[str], entities: list[list[str]]) -> int
        **kwargs: Arguments passed to ChainOfDensity

    Example:
        ```python
        # Get a moderately dense summary (step 3 out of 5)
        cod = ChainOfDensityWithPreference(
            num_steps=5,
            preferred_step=3,
            target_length=80
        )
        result = cod(article=article)
        print(result.final_summary)  # Returns step 3 summary
        print(result.preferred_step)  # Returns 3
        ```
    """

    def __init__(
        self,
        preferred_step: int | None = None,
        selection_fn: Callable[[list[str], list[list[str]]], int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if preferred_step is not None and preferred_step < 0:
            raise ValueError(f"preferred_step must be non-negative, got {preferred_step}.")
        self.preferred_step = preferred_step
        self.selection_fn = selection_fn

    def _select_step(self, summaries: list[str], entities_added: list[list[str]]) -> int:
        if self.selection_fn is not None:
            return self.selection_fn(list(summaries), list(entities_added))
        if self.preferred_step is not None:
            return min(self.preferred_step, len(summaries) - 1)
        return len(summaries) - 1

    def _extra_fields(self, selected_step: int) -> dict[str, Any]:
        return {"preferred_step": selected_step}


# Convenience function for quick usage
def chain_of_density(
    article: str,
    num_steps: int = 5,
    target_length: int = 80,
    lm: BaseLM | None = None,
) -> Prediction:
    """
    Convenience function to generate a Chain of Density summary.

    Args:
        article: The text to summarize
        num_steps: Number of densification iterations
        target_length: Target summary length in words
        lm: Optional language model to use

    Returns:
        Prediction with final_summary and intermediate summaries

    Example:
        ```python
        result = chain_of_density(article, num_steps=5)
        print(result.final_summary)
        ```
    """
    cod = ChainOfDensity(num_steps=num_steps, target_length=target_length)

    if lm is not None:
        cod.set_lm(lm)

    return cod(article=article)


__all__ = [
    "ChainOfDensity",
    "ChainOfDensityWithPreference",
    "chain_of_density",
    "InitialSummary",
    "DensifySummary",
    "DensificationStep",
    "IdentifyMissingEntities",
]
