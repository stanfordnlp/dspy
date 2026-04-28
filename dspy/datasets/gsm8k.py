import random

import tqdm


class GSM8K:
    """Loader for the GSM8K grade-school math word-problem benchmark.

    Downloads the Hugging Face ``gsm8k`` (``main`` config) dataset and turns
    each row into a ``dspy.Example`` with three fields:

    - ``question``: the natural-language word problem.
    - ``gold_reasoning``: the chain-of-thought rationale included in the
      original answer (everything before the final ``####`` marker).
    - ``answer``: the integer final answer as a string.

    The constructor splits the official train set into a fixed train/dev
    pair (200 train + 300 dev examples after a seeded shuffle) and exposes
    the entire official test set.

    Attributes:
        train: First 200 shuffled training examples.
        dev: Next 300 shuffled training examples (used as a held-out dev set
            because the GSM8K release exposes only train and test splits).
        test: The official GSM8K test split, shuffled deterministically.
        do_shuffle: ``False`` so that downstream consumers (for example,
            ``dspy.Dataset`` subclasses) preserve this loader's ordering.

    Examples:
        >>> from dspy.datasets.gsm8k import GSM8K
        >>> data = GSM8K()  # doctest: +SKIP
        >>> example = data.train[0]  # doctest: +SKIP
        >>> example.question, example.answer  # doctest: +SKIP
    """

    def __init__(self):
        self.do_shuffle = False

        from datasets import load_dataset

        dataset = load_dataset("gsm8k", "main")

        hf_official_train = dataset["train"]
        hf_official_test = dataset["test"]
        official_train = []
        official_test = []

        for example in tqdm.tqdm(hf_official_train):
            question = example["question"]

            answer = example["answer"].strip().split()
            assert answer[-2] == "####"

            gold_reasoning = " ".join(answer[:-2])
            answer = str(int(answer[-1].replace(",", "")))

            official_train.append({"question": question, "gold_reasoning": gold_reasoning, "answer": answer})

        for example in tqdm.tqdm(hf_official_test):
            question = example["question"]

            answer = example["answer"].strip().split()
            assert answer[-2] == "####"

            gold_reasoning = " ".join(answer[:-2])
            answer = str(int(answer[-1].replace(",", "")))

            official_test.append({"question": question, "gold_reasoning": gold_reasoning, "answer": answer})

        rng = random.Random(0)
        rng.shuffle(official_train)

        rng = random.Random(0)
        rng.shuffle(official_test)

        trainset = official_train[:200]
        devset = official_train[200:500]
        testset = official_test[:]

        import dspy

        trainset = [dspy.Example(**x).with_inputs("question") for x in trainset]
        devset = [dspy.Example(**x).with_inputs("question") for x in devset]
        testset = [dspy.Example(**x).with_inputs("question") for x in testset]

        self.train = trainset
        self.dev = devset
        self.test = testset


def parse_integer_answer(answer, only_first_line=True):
    """Extract the final integer answer from a free-form model response.

    The function walks the response from left to right looking for the last
    whitespace-separated token that contains a digit, drops everything after
    a decimal point, strips non-digit characters, and parses the result as an
    integer. If no digits can be recovered, ``0`` is returned so that callers
    can compare against gold answers without raising.

    Args:
        answer: The model output to parse. Will be coerced to a string by
            callers.
        only_first_line: If ``True`` (the default), restrict parsing to the
            first line of ``answer``. Useful when the model emits a final
            answer on the first line followed by additional commentary.

    Returns:
        The parsed integer, or ``0`` if no number could be extracted.

    Examples:
        >>> parse_integer_answer("The answer is 42.")
        42
        >>> parse_integer_answer("no digits here")
        0
    """
    try:
        if only_first_line:
            answer = answer.strip().split("\n")[0]

        # find the last token that has a number in it
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][-1]
        answer = answer.split(".")[0]
        answer = "".join([c for c in answer if c.isdigit()])
        answer = int(answer)

    except (ValueError, IndexError):
        answer = 0

    return answer


def gsm8k_metric(gold, pred, trace=None):
    """Exact-match metric for GSM8K predictions.

    Both ``gold.answer`` and ``pred.answer`` are normalized through
    :func:`parse_integer_answer` before comparison, which makes the metric
    tolerant of free-form model outputs (for example, ``"The answer is 42"``)
    and stringified gold answers.

    Args:
        gold: A ``dspy.Example`` (or any object with an ``answer`` attribute)
            holding the reference answer.
        pred: A ``dspy.Prediction`` (or any object with an ``answer``
            attribute) holding the model's prediction.
        trace: Unused; accepted so this function matches the signature
            expected by ``dspy.Evaluate`` and the teleprompters.

    Returns:
        ``1`` if the parsed integer answers match, ``0`` otherwise.
    """
    return int(parse_integer_answer(str(gold.answer))) == int(parse_integer_answer(str(pred.answer)))
