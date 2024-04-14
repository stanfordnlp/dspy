import re
from typing import Union

from dsp.modules import LM


# This testing module was moved in PR #735 to patch Arize Phoenix logging
class DummyLM(LM):
    """Dummy language model for unit testing purposes."""

    def __init__(self, answers: Union[list[str], dict[str, str]], follow_examples: bool = False):
        """Initializes the dummy language model.

        Parameters:
        - answers: A list of strings or a dictionary with string keys and values.
        - follow_examples: If True, and the prompt contains an example exactly equal to the prompt,
                           the dummy model will return the next string in the list for each request.
        If a list is provided, the dummy model will return the next string in the list for each request.
        If a dictionary is provided, the dummy model will return the value corresponding to the key that matches the prompt.
        """
        super().__init__("dummy-model")
        self.provider = "dummy"
        self.answers = answers
        self.follow_examples = follow_examples

    def basic_request(self, prompt, n=1, **kwargs) -> dict[str, list[dict[str, str]]]:
        """Generates a dummy response based on the prompt."""
        dummy_response = {"choices": []}
        for _ in range(n):
            answer = None

            if self.follow_examples:
                prefix = prompt.split("\n")[-1]
                _instructions, _format, *examples, _output = prompt.split("\n---\n")
                examples_str = "\n".join(examples)
                possible_answers = re.findall(prefix + r"\s*(.*)", examples_str)
                if possible_answers:
                    # We take the last answer, as the first one is just from
                    # the "Follow the following format" section.
                    answer = possible_answers[-1]
                    print(f"DummyLM got found previous example for {prefix} with value {answer=}")
                else:
                    print(f"DummyLM couldn't find previous example for {prefix=}")

            if answer is None:
                if isinstance(self.answers, dict):
                    answer = next((v for k, v in self.answers.items() if k in prompt), None)
                else:
                    if len(self.answers) > 0:
                        answer = self.answers[0]
                        self.answers = self.answers[1:]

            if answer is None:
                answer = "No more responses"

            # Mimic the structure of a real language model response.
            dummy_response["choices"].append(
                {
                    "text": answer,
                    "finish_reason": "simulated completion",
                },
            )

            RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
            print("=== DummyLM ===")
            print(prompt, end="")
            print(f"{RED}{answer}{RESET}")
            print("===")

        # Simulate processing and storing the request and response.
        history_entry = {
            "prompt": prompt,
            "response": dummy_response,
            "kwargs": kwargs,
            "raw_kwargs": kwargs,
        }
        self.history.append(history_entry)

        return dummy_response

    def __call__(self, prompt, _only_completed=True, _return_sorted=False, **kwargs):
        """Retrieves dummy completions."""
        response = self.basic_request(prompt, **kwargs)
        choices = response["choices"]

        # Filter choices and return text completions.
        return [choice["text"] for choice in choices]

    def get_convo(self, index) -> str:
        """Get the prompt + anwer from the ith message."""
        return self.history[index]["prompt"] + " " + self.history[index]["response"]["choices"][0]["text"]
