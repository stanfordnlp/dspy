import dspy
from dspy.datasets import HotPotQA
from dspy.evaluate import Evaluate
import pandas as pd
import os
from .base_task import BaseTask
import string


def preprocess_text(text):
    # Remove all punctuation
    text = "".join(char for char in text if char not in string.punctuation)
    # Convert to lower case
    text = text.lower()
    # Remove 'peace!' from the end if it exists
    if text.endswith("peace!"):
        text = text[
            :-6
        ].strip()  # Remove the last 'peace!' and strip any trailing spaces
    return text


def calculate_recall(correct_words, pred_words):
    correct_set = set(correct_words)
    pred_set = set(pred_words)
    # Calculate the number of correct words recalled in the prediction
    common_words = correct_set.intersection(pred_set)
    if len(correct_set) == 0:
        return True  # Avoid division by zero if correct answer is somehow empty
    recall = len(common_words) / len(correct_set)
    return recall >= 2 / 3


def check_conditions(example, pred, trace=None, debug=False):
    category = example.category
    answer = pred.answer

    preprocessed_pred_answer = preprocess_text(pred.answer)
    preprocessed_correct_answer = preprocess_text(example.answer)

    # Check for exact match after preprocessing
    if preprocessed_pred_answer != preprocessed_correct_answer:
        if debug:
            print(
                f"Exact match failed. Expected: '{preprocessed_correct_answer}', Got: '{preprocessed_pred_answer}'"
            )
        return False

    # When the answer is a place, the response should contain no punctuation
    if category == "place":
        if any(char in answer for char in ",.?!;:"):
            return False
        else:
            if debug:
                print(
                    f"Place. When the answer is a place, the response should contain no punctuation {answer}"
                )
            return True

    # When the answer is a date, the response should end with "Peace!"
    elif category == "date":
        if not answer.endswith("Peace!"):
            return False
        else:
            if debug:
                print(
                    f"Date. When the answer is a date, the response should end with Peace! {answer}"
                )
            return True

    # When the answer is a person, the response should be entirely in lowercase
    elif category == "person":
        if answer != answer.lower():
            return False
        else:
            if debug:
                print(
                    f"Answer. When the answer is a person, the response should be entirely in lowercase {answer}"
                )
            return True

    # When the answer is none of the above categories, the response should be in all caps and not end with "Peace!"
    else:
        if answer != answer.upper() or answer.endswith("Peace!"):
            return False
        else:
            if debug:
                print(
                    f"Other category. the response should be in all caps and not end with Peace! {answer}"
                )
            return True


class MultiHop(dspy.Module):
    def __init__(self, passages_per_hop):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = dspy.ChainOfThought("context ,question->search_query")
        self.generate_answer = dspy.ChainOfThought("context ,question->answer")

    def forward(self, question):
        context = []
        for hop in range(2):
            query = self.generate_query(context=context, question=question).search_query
            context += self.retrieve(query).passages
        return dspy.Prediction(
            context=context,
            answer=self.generate_answer(context=context, question=question).answer,
        )


class GenerateAnswerInstruction(dspy.Signature):
    """When the answer is a person, respond entirely in lowercase.  When the answer is a place, ensure your response contains no punctuation.  When the answer is a date, end your response with “Peace!”.  Never end your response with "Peace!" under other circumstances.  When the answer is none of the above categories respond in all caps."""

    context = dspy.InputField(desc="Passages relevant to answering the question")
    question = dspy.InputField(desc="Question we want an answer to")
    answer = dspy.OutputField(desc="Answer to the question")


class MultiHopHandwritten(dspy.Module):
    def __init__(self, passages_per_hop):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = dspy.ChainOfThought("context ,question->search_query")
        self.generate_answer = dspy.ChainOfThought(GenerateAnswerInstruction)

    def forward(self, question):
        context = []
        for hop in range(2):
            query = self.generate_query(context=context, question=question).search_query
            context += self.retrieve(query).passages
        return dspy.Prediction(
            context=context,
            answer=self.generate_answer(context=context, question=question).answer,
        )


class HotPotQAConditionalTask(BaseTask):
    def __init__(self):
        # Read in the conditional HotpotQA dataset from nfl_datasets as a csv from nfl_datasets/conditional_hotpotqa

        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the absolute paths to the datasets
        hotpotqa_train_path = os.path.join(
            script_dir, "../datasets/hotpotqa_conditional/hotpot_train.csv"
        )
        hotpotqa_dev_path = os.path.join(
            script_dir, "../datasets/hotpotqa_conditional/hotpot_dev.csv"
        )
        hotpotqa_test_path = os.path.join(
            script_dir, "../datasets/hotpotqa_conditional/hotpot_test.csv"
        )

        # Read the datasets
        hotpotqa_train = pd.read_csv(hotpotqa_train_path)
        hotpotqa_dev = pd.read_csv(hotpotqa_dev_path)
        hotpotqa_test = pd.read_csv(hotpotqa_test_path)

        combined_train = pd.concat([hotpotqa_train, hotpotqa_dev], ignore_index=True)

        # Load and configure the datasets.
        self.trainset = [
            dspy.Example(
                question=row["question"],
                answer=row["answer"],
                category=row["answer category"],
            ).with_inputs("question")
            for index, row in combined_train.iterrows()
        ]
        self.testset = [
            dspy.Example(
                question=row["question"],
                answer=row["answer"],
                category=row["answer category"],
            ).with_inputs("question")
            for index, row in hotpotqa_test.iterrows()
        ]

        # Set up metrics
        NUM_THREADS = 16

        # TODO: set up metrics
        self.metric = check_conditions

        kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=15)

        self.set_splits(TRAIN_NUM=100, DEV_NUM=100, TEST_NUM=100)

    def get_program(self):
        return MultiHop(passages_per_hop=3)

    def get_metric(self):
        return self.metric

    def get_default_max_bootstrapped_demos(self):
        return 4

    def get_default_max_labeled_demos(self):
        return 0

    def get_max_tokens(self):
        return 600
