import textwrap
import pytest
import re
import dspy
from dsp.modules import LM
from dspy.teleprompt.signature_opt_bayesian import BayesianSignatureOptimizer
from dspy.utils.dummies import DummyLM
from dspy import Example


# Define a simple metric function for testing
def simple_metric(example, prediction, trace=None):
    # Simplified metric for testing: true if prediction matches expected output
    return example.output == prediction.output


# Some example data
capitals = {
    "Germany": "Berlin",
    "France": "Paris",
    "Denmark": "Copenhagen",
    "Sweden": "Stockholm",
    "Norway": "Oslo",
}
# Not used for training data
extra_capitals = {
    "Spain": "Madrid",
    "Portugal": "Lisbon",
    "Italy": "Rome",
}

# Example training and validation sets
trainset = [
    Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
    Example(
        input="What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!"
    ).with_inputs("input"),
] + [
    Example(input=f"What is the capital of {country}?", output=capital).with_inputs(
        "input"
    )
    for country, capital in capitals.items()
]


class ConditionalLM(LM):
    def __init__(self):
        super().__init__("conditional-lm")

    def basic_request(self, prompt, n=1, **kwargs):
        # If we are in the "optimization" stage, we don't say much.
        if prompt.endswith("Observations:"):
            answer = " (*silence*)"
        elif prompt.endswith("Proposed Instruction:"):
            answer = " Input: "
        elif prompt.endswith("Proposed Prefix For Output Field:"):
            answer = " Output: "
        elif prompt.endswith("Summary:"):
            answer = " summarizing..."
        else:
            pairs = re.findall(r"Input: (.*)\nOutput: (.*)", prompt)

            print("PROMPT:", prompt)
            print("PAIRS:", pairs)

            last = re.search(r"Input: (.*)\nReasoning: (.*)$", prompt)
            current_question = last.group(1)

            if match := re.match(r"What is the capital of (.*?)\?", current_question):
                country = match.group(1)
                # If we had a previous example of a question about a capital, the model
                # has learned the format, and will answer with question correctly.
                if any("capital" in question for question, _ in pairs):
                    answer = (capitals | extra_capitals)[country]
                # Otherwise, it is confused and will answer with the country's name.
                else:
                    answer = country
            # For other questions, the model will answer with the last word of the question.
            else:
                answer = current_question.split()[-1]

            answer = "think deeply.\nOutput: " + answer

        RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
        print("=== DummyLM ===")
        print(prompt, end="")
        print(f"{RED}{answer}{RESET}")
        print("===")

        dummy_response = {"choices": []}
        for _ in range(n):
            dummy_response["choices"].append(
                {
                    "text": answer,
                    "finish_reason": "done",
                }
            )

        # Simulate processing and storing the request and response.
        history_entry = {
            "prompt": prompt,
            "response": dummy_response,
            "kwargs": kwargs,
            "raw_kwargs": kwargs,
        }
        self.history.append(history_entry)

        return dummy_response

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        response = self.basic_request(prompt, **kwargs)
        return [choice["text"] for choice in response["choices"]]

    def get_convo(self, index):
        """get the prompt + anwer from the ith message"""
        return (
            self.history[index]["prompt"]
            + " "
            + self.history[index]["response"]["choices"][0]["text"]
        )


def test_bayesian_signature_optimizer_initialization():
    optimizer = BayesianSignatureOptimizer(
        metric=simple_metric, n=10, init_temperature=1.4, verbose=True, track_stats=True
    )
    assert optimizer.metric == simple_metric, "Metric not correctly initialized"
    assert optimizer.n == 10, "Incorrect 'n' parameter initialization"
    assert (
        optimizer.init_temperature == 1.4
    ), "Initial temperature not correctly initialized"
    assert optimizer.verbose is True, "Verbose flag not correctly initialized"
    assert optimizer.track_stats is True, "Track stats flag not correctly initialized"


class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        # SignatureOptimizer doesn't work with dspy.Predict
        self.predictor = dspy.ChainOfThought(signature)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


def test_signature_optimizer_optimization_process():
    lm = ConditionalLM()
    dspy.settings.configure(lm=lm)

    student = SimpleModule(signature="input -> output")

    optimizer = BayesianSignatureOptimizer(
        metric=simple_metric,
        n=10,
        init_temperature=1.4,
        verbose=False,
        track_stats=False,
    )

    # Adjustments: Include required parameters for the compile method
    optimized_student = optimizer.compile(
        student=student,
        devset=trainset,
        optuna_trials_num=10,
        max_bootstrapped_demos=3,
        max_labeled_demos=5,
        eval_kwargs={"num_threads": 1, "display_progress": False},
    )

    assert len(optimized_student.predictor.demos) == 5


def test_signature_optimizer_bad_lm():
    dspy.settings.configure(
        lm=DummyLM([f"Optimized instruction {i}" for i in range(30)])
    )
    student = SimpleModule(signature="input -> output")
    optimizer = BayesianSignatureOptimizer(
        metric=simple_metric,
        n=10,
        init_temperature=1.4,
        verbose=False,
        track_stats=False,
    )

    # Krista: when the code tries to generate bootstrapped examples, the examples are generated using DummyLM,
    # which only outputs "Optimized instruction i" this means that none of the bootstrapped examples are successful,
    # and therefore the set of examples that we're using to generate new prompts is empty
    with pytest.raises(ValueError):
        _optimized_student = optimizer.compile(
            student=student,
            devset=trainset,
            optuna_trials_num=10,
            max_bootstrapped_demos=3,
            max_labeled_demos=5,
            eval_kwargs={"num_threads": 1, "display_progress": False},
        )


def test_optimization_and_output_verification():
    # Make a language model that is always right, except on the last
    # example in the train set.
    lm = ConditionalLM()
    dspy.settings.configure(lm=lm)

    optimizer = BayesianSignatureOptimizer(
        metric=simple_metric,
        n=10,
        init_temperature=1.4,
        verbose=False,
        track_stats=True,
    )

    student = SimpleModule("input -> output")

    # Compile the student with the optimizer
    optimized_student = optimizer.compile(
        student=student,
        devset=trainset,
        optuna_trials_num=4,
        max_bootstrapped_demos=2,
        max_labeled_demos=3,
        eval_kwargs={"num_threads": 1, "display_progress": False},
    )

    # Simulate calling the optimized student with a new input
    test_input = "What is the capital of Spain?"
    prediction = optimized_student(input=test_input)

    print("CORRECT ANSWER")
    print(lm.get_convo(-1))

    assert prediction.output == "Madrid"

    assert lm.get_convo(-1) == textwrap.dedent(
        """\
        Input:

        ---

        Follow the following format.

        Input: ${input}
        Reasoning: Let's think step by step in order to ${produce the output}. We ...
        Output: ${output}

        ---

        Input: What is the capital of Norway?
        Reasoning: Let's think step by step in order to think deeply.
        Output: Oslo

        ---

        Input: What is the capital of Sweden?
        Reasoning: Let's think step by step in order to think deeply.
        Output: Stockholm

        ---

        Input: What is the capital of France?
        Output: Paris

        ---

        Input: What is the capital of Spain?
        Reasoning: Let's think step by step in order to think deeply.
        Output: Madrid"""
    )
