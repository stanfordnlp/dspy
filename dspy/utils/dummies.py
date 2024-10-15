import random
import re
from collections import defaultdict
from typing import Union

import numpy as np

from dsp.modules import LM as DSPLM
from dsp.utils.utils import dotdict
from dspy.adapters.chat_adapter import field_header_pattern, format_fields
from dspy.clients.lm import LM


class DSPDummyLM(DSPLM):
    """Dummy language model for unit testing purposes subclassing DSP LM class."""

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

            RED, _, RESET = "\033[91m", "\033[92m", "\033[0m"
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


class DummyLM(LM):
    """
    Dummy language model for unit testing purposes.

    Three modes of operation:

    ## 1. List of dictionaries

    If a list of dictionaries is provided, the dummy model will return the next dictionary
    in the list for each request, formatted according to the `format_fields` function.
    from the chat adapter.

    ```python
        lm = DummyLM([{"answer": "red"}, {"answer": "blue"}])
        dspy.settings.configure(lm=lm)
        predictor("What color is the sky?")
        # Output: "[[## answer ##]]\nred"
        predictor("What color is the sky?")
        # Output: "[[## answer ##]]\nblue"
    ```

    ## 2. Dictionary of dictionaries

    If a dictionary of dictionaries is provided, the dummy model will return the value
    corresponding to the key which is contained with the final message of the prompt,
    formatted according to the `format_fields` function from the chat adapter.

    ```python
        lm = DummyLM({"What color is the sky?": {"answer": "blue"}})
        dspy.settings.configure(lm=lm)
        predictor("What color is the sky?")
        # Output: "[[## answer ##]]\nblue"
    ```

    ## 3. Follow examples

    If `follow_examples` is set to True, and the prompt contains an example input exactly equal to the prompt,
    the dummy model will return the output from that example.

    ```python
        lm = DummyLM([{"answer": "red"}], follow_examples=True)
        dspy.settings.configure(lm=lm)
        predictor("What color is the sky?, demos=dspy.Example(input="What color is the sky?", output="blue"))
        # Output: "[[## answer ##]]\nblue"
    ```

    """

    def __init__(self, answers: Union[list[dict[str, str]], dict[str, dict[str, str]]], follow_examples: bool = False):
        super().__init__("dummy", "chat", 0.0, 1000, True)
        self.answers = answers
        if isinstance(answers, list):
            self.answers = iter(answers)
        self.follow_examples = follow_examples

    def _use_example(self, messages):
        # find all field names
        fields = defaultdict(int)
        for message in messages:
            if "content" in message:
                if ma := field_header_pattern.match(message["content"]):
                    fields[message["content"][ma.start() : ma.end()]] += 1
        # find the fields which are missing from the final turns
        max_count = max(fields.values())
        output_fields = [field for field, count in fields.items() if count != max_count]

        # get the output from the last turn that has the output fields as headers
        final_input = messages[-1]["content"].split("\n\n")[0]
        for input, output in zip(reversed(messages[:-1]), reversed(messages)):
            if any(field in output["content"] for field in output_fields) and final_input in input["content"]:
                return output["content"]

    def __call__(self, prompt=None, messages=None, **kwargs):
        # Build the request.
        outputs = []
        for _ in range(kwargs.get("n", 1)):
            messages = messages or [{"role": "user", "content": prompt}]
            kwargs = {**self.kwargs, **kwargs}

            if self.follow_examples:
                outputs.append(self._use_example(messages))
            elif isinstance(self.answers, dict):
                outputs.append(
                    next(
                        (format_fields(v) for k, v in self.answers.items() if k in messages[-1]["content"]),
                        "No more responses",
                    )
                )
            else:
                outputs.append(format_fields(next(self.answers, {"answer": "No more responses"})))

            # Logging, with removed api key & where `cost` is None on cache hit.
            kwargs = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
            entry = dict(prompt=prompt, messages=messages, kwargs=kwargs)
            entry = dict(**entry, outputs=outputs, usage=0)
            entry = dict(**entry, cost=0)
            self.history.append(entry)

        return outputs

    def get_convo(self, index):
        """Get the prompt + anwer from the ith message."""
        return self.history[index]["messages"], self.history[index]["outputs"]


def dummy_rm(passages=()) -> callable:
    if not passages:

        def inner(query: str, *, k: int, **kwargs):
            assert False, "No passages defined"

        return inner
    max_length = max(map(len, passages)) + 100
    vectorizer = DummyVectorizer(max_length)
    passage_vecs = vectorizer(passages)

    def inner(query: str, *, k: int, **kwargs):
        assert k <= len(passages)
        query_vec = vectorizer([query])[0]
        scores = passage_vecs @ query_vec
        largest_idx = (-scores).argsort()[:k]
        # return dspy.Prediction(passages=[passages[i] for i in largest_idx])
        return [dotdict(dict(long_text=passages[i])) for i in largest_idx]

    return inner


class DummyVectorizer:
    """Simple vectorizer based on n-grams."""

    def __init__(self, max_length=100, n_gram=2):
        self.max_length = max_length
        self.n_gram = n_gram
        self.P = 10**9 + 7  # A large prime number
        random.seed(123)
        self.coeffs = [random.randrange(1, self.P) for _ in range(n_gram)]

    def _hash(self, gram):
        """Hashes a string using a polynomial hash function."""
        h = 1
        for coeff, c in zip(self.coeffs, gram):
            h = h * coeff + ord(c)
            h %= self.P
        return h % self.max_length

    def __call__(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for text in texts:
            grams = [text[i : i + self.n_gram] for i in range(len(text) - self.n_gram + 1)]
            vec = [0] * self.max_length
            for gram in grams:
                vec[self._hash(gram)] += 1
            vecs.append(vec)

        vecs = np.array(vecs, dtype=np.float32)
        vecs -= np.mean(vecs, axis=1, keepdims=True)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10  # Added epsilon to avoid division by zero
        return vecs
