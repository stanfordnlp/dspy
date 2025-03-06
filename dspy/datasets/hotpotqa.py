import random

from dspy.datasets.dataset import Dataset


class HotPotQA(Dataset):
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

        hf_official_train = load_dataset("hotpot_qa", "fullwiki", split="train", trust_remote_code=True)
        hf_official_dev = load_dataset("hotpot_qa", "fullwiki", split="validation", trust_remote_code=True)

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

    print(dataset.dev[0].question)
    print(dataset.dev[340].question)
    print(dataset.dev[937].question)

"""
What was the population of the city where Woodward Avenue ends in 2010?
Where did the star , who is also an executive producer, of the Mick begin her carrer? 
16 1000 0
Both London and German have seen attacks during war, there was one specific type of attack that Germany called the blitz, what did London call a similar attack?
Pre-Madonna was a collection of demos by the singer who was a leading presence during the emergence of what network?
Alan Mills composed the classic folk song that tells the story of what? 
"""
