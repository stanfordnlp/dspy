from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.signatures import InputField, OutputField
from dspy.signatures.signature import ensure_signature


class MultiChainComparison(Module):
    """Compare multiple chain-of-thought reasoning attempts and produce a refined answer.

    This module implements the multi-chain comparison technique: given ``M`` candidate
    completions (each containing a ``rationale`` or ``reasoning`` field and an answer), it
    asks the language model to evaluate all attempts together and return an improved,
    holistic rationale along with the final answer.

    The workflow is:

    1. Generate ``M`` candidate completions for the same input (e.g. by calling
       ``dspy.ChainOfThought`` with ``n=M`` or by collecting predictions from
       separate calls).
    2. Pass the list of completions to this module together with the original inputs.
    3. The module formats each completion as a student attempt and delegates to an
       internal ``dspy.Predict`` whose signature includes the original fields plus
       the ``M`` reasoning attempts.

    Args:
        signature: The task signature (e.g. ``"question -> answer"``). Must contain at
            least one output field.
        M: The number of chain-of-thought completions to compare. Defaults to 3.
        temperature: Sampling temperature forwarded to the underlying ``Predict``.
            Defaults to 0.7.
        **config: Additional keyword arguments forwarded to the underlying ``Predict``.

    Examples:
        ```python
        import dspy

        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        # 1) Define a simple QA signature.
        class BasicQA(dspy.Signature):
            \"\"\"Answer questions with short factoid answers.\"\"\"
            question = dspy.InputField()
            answer = dspy.OutputField(desc="often between 1 and 5 words")

        # 2) Generate M=3 candidate completions.
        cot = dspy.ChainOfThought(BasicQA, n=3, temperature=0.7)
        completions = cot(question="What is the color of the sky?").completions

        # 3) Compare and refine.
        compare = dspy.MultiChainComparison(BasicQA, M=3)
        result = compare(completions, question="What is the color of the sky?")
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
        """Compare the candidate completions and return a refined prediction.

        Each completion must contain a ``rationale`` (or ``reasoning``) field and the
        last output field defined in the signature. The method formats these into
        student-attempt strings, passes them to the underlying ``Predict``, and
        returns the refined prediction.

        Args:
            completions: A sequence of ``M`` ``dspy.Prediction`` objects (or
                dict-like objects) produced by a chain-of-thought module.
            **kwargs: The original input field values required by the signature
                (e.g. ``question="..."``).

        Returns:
            dspy.Prediction: A prediction containing the ``rationale`` and the
            final output field(s) defined in the signature.

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
