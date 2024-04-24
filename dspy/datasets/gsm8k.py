from datasets import load_dataset

from dspy.primitives.example import Example

from .dataset import Dataset


class GSM8K(Dataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.load()

    def load(self) -> None:
        # Load dataset from HuggingFace
        train_ds, test_ds = load_dataset("gsm8k", "main", split=["train", "test"])

        # Process Samples into Examples
        def process_sample(sample) -> Example:
            question = sample["question"]
            answer = sample["answer"].strip().split()

            gold_reasoning = " ".join(answer[:-2])
            answer = str(int(answer[-1].replace(",", "")))

            return Example(question=question, gold_reasoning=gold_reasoning, answer=answer).with_inputs("question")

        train_examples = [process_sample(sample) for sample in train_ds]
        test_examples = [process_sample(sample) for sample in test_ds]

        # Split Data
        self.train_examples = train_examples[:200]
        self.dev_examples = train_examples[200:500]
        self.test_examples = test_examples


def parse_integer_answer(answer, only_first_line=True):
    try:
        if only_first_line:
            answer = answer.strip().split("\n")[0]

        # find the last token that has a number in it
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][-1]
        answer = answer.split(".")[0]
        answer = "".join([c for c in answer if c.isdigit()])
        answer = int(answer)

    except (ValueError, IndexError):
        # print(answer)
        answer = 0

    return answer


def gsm8k_metric(gold, pred, trace=None):
    return int(parse_integer_answer(str(gold.answer))) == int(parse_integer_answer(str(pred.answer)))
