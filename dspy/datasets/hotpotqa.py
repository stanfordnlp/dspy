import random

from dspy.datasets.dataset import Dataset


class HotPotQA(Dataset):
    """Loader for the HotPotQA multi-hop question-answering benchmark.

    Pulls the ``hotpot_qa`` dataset (``fullwiki`` config) from Hugging Face
    and exposes its train/dev/test splits through the standard
    ``dspy.Dataset`` interface. Only "hard" examples from the official train
    split are kept; by default a 75/25 partition of those hard examples is
    used as ``train``/``dev``, and the official validation split is exposed
    as ``test``. Setting ``unofficial_dev=False`` disables the synthetic dev
    split so that ``dev`` is ``None``.

    Args:
        *args: Forwarded to ``Dataset.__init__``.
        only_hard_examples: Must be ``True`` (default). The flag exists so
            that future support for easy examples can be added without
            changing the call sites; passing ``False`` raises an
            ``AssertionError`` because dev consistency with the official
            HotPotQA dev split would be broken.
        keep_details: Controls which fields are retained on training
            examples. Accepts:

            - ``True``: keep ``id``, ``question``, ``answer``, ``type``,
              ``supporting_facts``, and ``context``.
            - ``"dev_titles"`` (default): keep ``question``, ``answer``, and
              the supporting-fact titles only on the synthetic dev split.
            - any other value: keep only ``question`` and ``answer``.
        unofficial_dev: If ``True`` (default), use the last 25% of the hard
            train split as a synthetic dev set. If ``False``, leave ``dev``
            as ``None`` and rely solely on the official test split.
        **kwargs: Forwarded to ``Dataset.__init__`` (for example,
            ``train_size``, ``dev_size``, ``input_keys``).

    Examples:
        >>> from dspy.datasets.hotpotqa import HotPotQA
        >>> data = HotPotQA(train_size=16, dev_size=200, test_size=0)  # doctest: +SKIP
        >>> example = data.train[0]  # doctest: +SKIP
        >>> example.question  # doctest: +SKIP
    """

    def __init__(
        self,
        *args,
        only_hard_examples=True,
        keep_details="dev_titles",
        unofficial_dev=True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert only_hard_examples, (
            "Care must be taken when adding support for easy examples."
            "Dev must be all hard to match official dev, but training can be flexible."
        )

        from datasets import load_dataset

        hf_official_train = load_dataset("hotpot_qa", "fullwiki", split="train")
        hf_official_dev = load_dataset("hotpot_qa", "fullwiki", split="validation")

        official_train = []
        for raw_example in hf_official_train:
            if raw_example["level"] == "hard":
                if keep_details is True:
                    keys = ["id", "question", "answer", "type", "supporting_facts", "context"]
                elif keep_details == "dev_titles":
                    keys = ["question", "answer", "supporting_facts"]
                else:
                    keys = ["question", "answer"]

                example = {k: raw_example[k] for k in keys}

                if "supporting_facts" in example:
                    example["gold_titles"] = set(example["supporting_facts"]["title"])
                    del example["supporting_facts"]

                official_train.append(example)

        rng = random.Random(0)
        rng.shuffle(official_train)

        self._train = official_train[: len(official_train) * 75 // 100]

        if unofficial_dev:
            self._dev = official_train[len(official_train) * 75 // 100 :]
        else:
            self._dev = None

        for example in self._train:
            if keep_details == "dev_titles":
                del example["gold_titles"]

        test = []
        for raw_example in hf_official_dev:
            assert raw_example["level"] == "hard"
            example = {k: raw_example[k] for k in ["id", "question", "answer", "type", "supporting_facts"]}
            if "supporting_facts" in example:
                example["gold_titles"] = set(example["supporting_facts"]["title"])
                del example["supporting_facts"]
            test.append(example)

        self._test = test


if __name__ == "__main__":
    from dspy.dsp.utils import dotdict

    data_args = dotdict(train_seed=1, train_size=16, eval_seed=2023, dev_size=200 * 5, test_size=0)
    dataset = HotPotQA(**data_args)

    print(dataset)
    print(dataset.train[0].question)
    print(dataset.train[15].question)

    print(len(dataset.train), len(dataset.dev), len(dataset.test))
