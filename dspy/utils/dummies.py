import random
from collections import defaultdict
from typing import Any

import numpy as np

from dspy.adapters.chat_adapter import ChatAdapter, FieldInfoWithName, field_header_pattern
from dspy.clients.lm import LM
from dspy.dsp.utils.utils import dotdict
from dspy.signatures.field import OutputField
from dspy.utils.callback import with_callbacks


class DummyLM(LM):
    """Dummy language model for unit testing purposes.

    Three modes of operation:

    Mode 1: List of dictionaries

    If a list of dictionaries is provided, the dummy model will return the next dictionary
    in the list for each request, formatted according to the `format_field_with_value` function.

    Example:

    ```
    lm = DummyLM([{"answer": "red"}, {"answer": "blue"}])
    dspy.settings.configure(lm=lm)
    predictor("What color is the sky?")
    # Output:
    # [[## answer ##]]
    # red
    predictor("What color is the sky?")
    # Output:
    # [[## answer ##]]
    # blue
    ```

    Mode 2: Dictionary of dictionaries

    If a dictionary of dictionaries is provided, the dummy model will return the value
    corresponding to the key which is contained with the final message of the prompt,
    formatted according to the `format_field_with_value` function from the chat adapter.

    ```
    lm = DummyLM({"What color is the sky?": {"answer": "blue"}})
    dspy.settings.configure(lm=lm)
    predictor("What color is the sky?")
    # Output:
    # [[## answer ##]]
    # blue
    ```

    Mode 3: Follow examples

    If `follow_examples` is set to True, and the prompt contains an example input exactly equal to the prompt,
    the dummy model will return the output from that example.

    ```
    lm = DummyLM([{"answer": "red"}], follow_examples=True)
    dspy.settings.configure(lm=lm)
    predictor("What color is the sky?, demos=dspy.Example(input="What color is the sky?", output="blue"))
    # Output:
    # [[## answer ##]]
    # blue
    ```

    """

    def __init__(self, answers: list[dict[str, str]] | dict[str, dict[str, str]], follow_examples: bool = False):
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
        for input, output in zip(reversed(messages[:-1]), reversed(messages), strict=False):
            if any(field in output["content"] for field in output_fields) and final_input in input["content"]:
                return output["content"]

    @with_callbacks
    def __call__(self, prompt=None, messages=None, **kwargs):
        def format_answer_fields(field_names_and_values: dict[str, Any]):
            return ChatAdapter().format_field_with_value(
                fields_with_values={
                    FieldInfoWithName(name=field_name, info=OutputField()): value
                    for field_name, value in field_names_and_values.items()
                }
            )

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
                        (format_answer_fields(v) for k, v in self.answers.items() if k in messages[-1]["content"]),
                        "No more responses",
                    )
                )
            else:
                outputs.append(format_answer_fields(next(self.answers, {"answer": "No more responses"})))

            # Logging, with removed api key & where `cost` is None on cache hit.
            kwargs = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
            entry = {"prompt": prompt, "messages": messages, "kwargs": kwargs}
            entry = {**entry, "outputs": outputs, "usage": 0}
            entry = {**entry, "cost": 0}
            self.history.append(entry)
            self.update_global_history(entry)

        return outputs

    async def acall(self, prompt=None, messages=None, **kwargs):
        return self.__call__(prompt=prompt, messages=messages, **kwargs)

    def get_convo(self, index):
        """Get the prompt + answer from the ith message."""
        return self.history[index]["messages"], self.history[index]["outputs"]


def dummy_rm(passages=()) -> callable:
    if not passages:

        def inner(query: str, *, k: int, **kwargs):
            raise ValueError("No passages defined")

        return inner
    max_length = max(map(len, passages)) + 100
    vectorizer = DummyVectorizer(max_length)
    passage_vecs = vectorizer(passages)

    def inner(query: str, *, k: int, **kwargs):
        assert k <= len(passages)
        query_vec = vectorizer([query])[0]
        scores = passage_vecs @ query_vec
        largest_idx = (-scores).argsort()[:k]

        return [dotdict(long_text=passages[i]) for i in largest_idx]

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
        for coeff, c in zip(self.coeffs, gram, strict=False):
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
