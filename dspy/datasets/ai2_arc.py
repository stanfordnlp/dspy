import re
from typing import Literal

import tqdm

import dspy


class AI2ARC:
    """AI2 Reasoning Challenge (ARC) Dataset.
    
    The ARC dataset contains multiple-choice science questions at a grade-school level.
    It consists of two subsets: ARC-Challenge (harder questions) and ARC-Easy (easier questions).
    
    Args:
        subset: Either "challenge" or "easy" to specify which subset to load
    """

    def __init__(self, subset: Literal["challenge", "easy"] = "challenge"):
        if subset not in ["challenge", "easy"]:
            raise ValueError("subset must be either 'challenge' or 'easy'")

        self.subset = subset

        from datasets import load_dataset

        dataset_name = "allenai/ai2_arc"
        config_name = f"ARC-{subset.title()}"

        try:
            hf_dataset = load_dataset(dataset_name, config_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load {config_name} from {dataset_name}: {e}")

        official_train = self._process_split(hf_dataset["train"])
        official_dev = self._process_split(hf_dataset["validation"])
        official_test = self._process_split(hf_dataset["test"])

        self.train = [dspy.Example(**x).with_inputs("question", "choices") for x in official_train]
        self.dev = [dspy.Example(**x).with_inputs("question", "choices") for x in official_dev]
        self.test = [dspy.Example(**x).with_inputs("question", "choices") for x in official_test]

    def _process_split(self, split_data):
        """Process a data split and convert to DSPy format."""
        processed_data = []

        for example in tqdm.tqdm(split_data, desc=f"Processing {self.subset} split"):
            choices_text = []
            choice_labels = example["choices"]["label"]
            choice_texts = example["choices"]["text"]

            for label, text in zip(choice_labels, choice_texts, strict=False):
                choices_text.append(f"({label}) {text}")

            processed_example = {
                "id": example["id"],
                "question": example["question"],
                "choices": "\n".join(choices_text),
                "choices_list": choice_texts,
                "choice_labels": choice_labels,
                "answer": example["answerKey"],
                "answer_text": self._get_answer_text(example)
            }

            processed_data.append(processed_example)

        return processed_data

    def _get_answer_text(self, example):
        """Extract the answer text corresponding to the correct answer key."""
        answer_key = example["answerKey"]
        choice_labels = example["choices"]["label"]
        choice_texts = example["choices"]["text"]

        try:
            answer_index = choice_labels.index(answer_key)
            return choice_texts[answer_index]
        except ValueError:
            # If answer key not found in labels, return the key itself
            return answer_key


def ai2_arc_metric(gold, pred, trace=None):
    """Metric function for AI2 ARC dataset.
    
    Args:
        gold: Gold example with 'answer' field
        pred: Predicted example with 'answer' field
        trace: Optional trace (unused)
        
    Returns:
        bool: True if prediction matches gold answer
    """
    pred_answer = _parse_arc_answer(pred.answer)
    gold_answer = _parse_arc_answer(gold.answer)

    if len(pred_answer) == 1 and pred_answer in ["A", "B", "C", "D"]:
        return pred_answer == gold_answer

    for letter in ["A", "B", "C", "D"]:
        if f"({letter})" in pred_answer or f"{letter})" in pred_answer:
            return letter == gold_answer

    return False


def _parse_arc_answer(answer_text):
    """Parse answer from model output to extract the letter choice.
    
    Args:
        answer_text: Raw model output
        
    Returns:
        str: Extracted answer letter (A, B, C, or D), or the original text if not found
    """
    answer_text = str(answer_text).strip().upper()

    patterns = [
        r"\(([ABCD])\)",  # (A), (B), etc.
        r"([ABCD])\)",    # A), B), etc.
        r"^([ABCD])$",    # Just A, B, C, D at start of line
        r"answer is ([ABCD])",  # "answer is A"
        r"choice ([ABCD])",     # "choice A"
    ]

    for pattern in patterns:
        match = re.search(pattern, answer_text.upper())
        if match:
            return match.group(1)

    # If no pattern found, return the first letter if it's A, B, C, or D
    first_char = answer_text.upper()[0] if answer_text else ""
    if first_char in ["A", "B", "C", "D"]:
        return first_char

    return answer_text
