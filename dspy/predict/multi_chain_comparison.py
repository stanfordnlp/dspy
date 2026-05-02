from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.signatures import InputField, OutputField
from dspy.signatures.signature import ensure_signature


class MultiChainComparison(Module):
    """Compares multiple chain-of-thought reasoning attempts and produces a refined answer.

    Given ``M`` reasoning attempts (typically from ``ChainOfThought`` with high
    temperature), this module asks the LM to review all attempts holistically
    and produce a corrected rationale and final answer. This is useful for
    self-consistency style pipelines where you generate several candidate
    solutions and want the model to pick or synthesize the best one.

    Example:
        >>> import dspy
        >>> cot = dspy.ChainOfThought("question -> answer", temperature=0.7, n=3)
        >>> compare = dspy.MultiChainComparison("question -> answer", M=3)
        >>> completions = cot(question="What is 12 * 15?").completions
        >>> result = compare(completions, question="What is 12 * 15?")
    """

    def __init__(self, signature, M=3, temperature=0.7, **config):  # noqa: N803
        """Initializes the MultiChainComparison module.

        Args:
            signature: The task signature defining input and output fields.
            M: The number of reasoning attempts to compare. Must match the
                number of completions passed to ``forward``.
            temperature: Sampling temperature for the comparison LM call.
            **config: Additional configuration passed to the underlying
                ``Predict`` module.
        """
        super().__init__()

        self.M = M
        signature = ensure_signature(signature)

        *_, self.last_key = signature.output_fields.keys()

        for idx in range(M):
            signature = signature.append(
                f"reasoning_attempt_{idx+1}",
                InputField(
                    prefix=f"Student Attempt #{idx+1}:",
                    desc="${reasoning attempt}",
                ),
            )

        signature = signature.prepend(
            "rationale",
            OutputField(
                prefix="Accurate Reasoning: Thank you everyone. Let's now holistically",
                desc="${corrected reasoning}",
            ),
        )

        self.predict = Predict(signature, temperature=temperature, **config)

    def forward(self, completions, **kwargs):
        """Compares reasoning attempts and returns a refined prediction.

        Extracts the rationale and answer from each completion, formats them
        as numbered "student attempts", and asks the LM to synthesize a
        corrected rationale and final answer.

        Args:
            completions: A list of ``M`` completion dicts, each containing
                a ``rationale`` (or ``reasoning``) key and the output field.
            **kwargs: Additional input field values for the signature.

        Returns:
            A ``Prediction`` with a ``rationale`` field containing the
            corrected reasoning and the original output field.

        Raises:
            AssertionError: If ``len(completions)`` does not equal ``M``.
        """
        attempts = []

        for c in completions:
            rationale = c.get("rationale", c.get("reasoning")).strip().split("\n")[0].strip()
            answer = str(c[self.last_key]).strip().split("\n")[0].strip()
            attempts.append(
                f"«I'm trying to {rationale} I'm not sure but my prediction is {answer}»",
            )

        assert (
            len(attempts) == self.M
        ), f"The number of attempts ({len(attempts)}) doesn't match the expected number M ({self.M}). Please set the correct value for M when initializing MultiChainComparison."

        kwargs = {
            **{f"reasoning_attempt_{idx+1}": attempt for idx, attempt in enumerate(attempts)},
            **kwargs,
        }
        return self.predict(**kwargs)
