from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.signatures import InputField, OutputField
from dspy.signatures.signature import ensure_signature


class MultiChainComparison(Module):
    """Compare multiple chain-of-thought reasoning attempts and produce a refined answer.

    ``MultiChainComparison`` implements a "judge" pattern: it collects ``M`` independent
    reasoning attempts (typically produced by :class:`~dspy.ChainOfThought` with
    higher temperature) and asks the language model to evaluate all of them
    holistically before generating a final, more accurate response.

    The module dynamically extends the provided signature by appending ``M``
    input fields (one per reasoning attempt) and prepending a ``rationale``
    output field that captures the corrected reasoning.

    Args:
        signature: The input/output signature describing the task. Can be a
            string shorthand (e.g., ``"question -> answer"``) or a
            :class:`~dspy.Signature` class.
        M (int): The number of reasoning attempts to compare. Defaults to 3.
        temperature (float): The sampling temperature for the underlying
            :class:`~dspy.Predict` call. Defaults to 0.7.
        **config: Additional keyword arguments forwarded to :class:`~dspy.Predict`.

    Examples:

    ```python
    import dspy

    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    cot = dspy.ChainOfThought("question -> answer", temperature=0.7, n=3)
    compare = dspy.MultiChainComparison("question -> answer", M=3)

    completions = cot(question="What is 23 * 47?").completions
    result = compare(completions, question="What is 23 * 47?")
    print(result.answer)
    ```
    """

    def __init__(self, signature, M=3, temperature=0.7, **config):  # noqa: N803
        super().__init__()

        self.M = M
        signature = ensure_signature(signature)

        *_, self.last_key = signature.output_fields.keys()

        for idx in range(M):
            signature = signature.append(
                f"reasoning_attempt_{idx + 1}",
                InputField(
                    prefix=f"Student Attempt #{idx + 1}:",
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
        """Run the multi-chain comparison on a set of reasoning attempts.

        Each completion is expected to contain a ``rationale`` (or ``reasoning``)
        field and the final output field from the signature. The method formats
        these into numbered "student attempts" and passes them to the underlying
        :class:`~dspy.Predict` module for holistic evaluation.

        Args:
            completions: A sequence of completion dictionaries, each containing
                at least a ``rationale`` (or ``reasoning``) key and the final
                output field defined by the signature. Typically obtained from
                ``dspy.ChainOfThought(...).completions``.
            **kwargs: Additional input fields required by the signature
                (e.g., ``question``).

        Returns:
            A :class:`~dspy.Prediction` with a ``rationale`` field containing the
            corrected reasoning and the original output fields from the signature.

        Raises:
            AssertionError: If the number of completions does not equal ``M``.
        """
        attempts = []

        for c in completions:
            rationale = c.get("rationale", c.get("reasoning")).strip().split("\n")[0].strip()
            answer = str(c[self.last_key]).strip().split("\n")[0].strip()
            attempts.append(
                f"«I'm trying to {rationale} I'm not sure but my prediction is {answer}»",
            )

        assert len(attempts) == self.M, (
            f"The number of attempts ({len(attempts)}) doesn't match the expected number M ({self.M}). Please set the correct value for M when initializing MultiChainComparison."
        )

        kwargs = {
            **{f"reasoning_attempt_{idx + 1}": attempt for idx, attempt in enumerate(attempts)},
            **kwargs,
        }
        return self.predict(**kwargs)
