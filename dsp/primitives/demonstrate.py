from typing import Any, Callable

import numpy as np

import dsp
from dsp.utils import EM, F1, DPR_normalize, dotdict, has_answer, normalize_text


class Example(dotdict):
    """A primitive datatype for representing an example"""

    demos: list[Any]

    def __init__(self, *args, **kwargs):
        assert len(args) <= 1
        super().__init__()

        if args:
            assert len(args) == 1
            self.update(args[0])

        self.update(**kwargs)

    def copy(self, **kwargs):
        the_copy = Example(**{**dict(self), **kwargs})

        return the_copy

    def without(self, *keys):
        """Removes the provided keys from the example and returns a copy"""
        keys = set(keys)
        return Example({k: v for k, v in self.items() if k not in keys})

    def demos_at(self, fn):
        """Returns a copy of the example with the demos stage transformed by the provided function"""

        def at(example):
            try:
                return fn(example).without("augmented")
            except Exception:
                return {}

        demos = [example.copy(**at(example)) for example in self.demos]
        return self.copy(demos=demos)


# def annotate(*transformations):
#     """Returns an Augment function that applies the provided transformations to the Examples"""

#     def do_augment(train, k=None, return_all=False):
#         rdemos = []
#         ademos = []

#         for example in train:  # tqdm.tqdm
#             raw_example = dsp.Example(example)

#             if (k is not None) and len(ademos) >= k:
#                 example = None

#             for f in transformations:
#                 if example is None:
#                     break

#                 example = f(example)

#             if example is not None:
#                 example.augmented = True
#                 ademos.append(example)
#             else:
#                 raw_example.augmented = False
#                 rdemos.append(raw_example)

#         if return_all:
#             return ademos + rdemos

#         return ademos

#     return do_augment


# def sample(train: list[Example], k: int):
#     """Sample k examples from train."""
#     rng = random.Random(dsp.settings.branch_idx)
#     shuffled_train = [dsp.Example(example) for example in train]
#     rng.shuffle(shuffled_train)

#     return shuffled_train[:k]


# def all_but(train: list[Example], x: Example) -> list[Example]:
#     """Removes the example x from the train set by comparing the question and history."""

#     output = [
#         y
#         for y in train
#         if not set.intersection(
#             set(x.get("history", []) + [x.question]),
#             set(y.get("history", []) + [y.question]),
#         )
#     ]

#     return output


def passage_match(passages: list[str], answers: list[str]) -> bool:
    """Returns True if any of the passages contains the answer."""
    return any(passage_has_answers(psg, answers) for psg in passages)


def answer_match(prediction, answers, frac=1.0):
    # pred = example.prediction
    # answers = example.answers

    if frac >= 1.0:
        return EM(prediction, answers)

    return F1(prediction, answers) >= frac


def passage_has_answers(passage: str, answers: list[str]) -> bool:
    """Returns True if the passage contains the answer."""
    return has_answer(
        tokenized_answers=[DPR_normalize(normalize_text(ans)) for ans in answers],
        text=normalize_text(passage),
    )


def cast_naive_get_only_question_text(inp_example: Example) -> Example:
    """
    Extracts question as a field to vectorize with Vectorizer object. `question` field is used.
    """
    return inp_example.copy(text_to_vectorize=inp_example.question)


def cast_naive_get_question_and_answer(inp_example: Example) -> Example:
    """
    Extracts question and answer as fields to vectorize with Vectorizer object.
    `question` and `answer` fields are used. They will be concatenated with the word "Answer"
    between.
    """
    text_to_vectorize = (
        inp_example.question.strip() + " Answer: " + inp_example.answer.strip()
    )
    return inp_example.copy(text_to_vectorize=text_to_vectorize)


def knn(
    train: list[Example],
    cast: Callable[[Example], Example] = cast_naive_get_only_question_text,
    **knn_args,
) -> Callable[[Example, int], list[Example]]:
    """
    A function that vectorizes train data using `dsm.settings.vectorizer`, then build an ANN/KNN
    index to search similar questions among `train` samples.

    Args:
        train: a bunch of questions to put in index & search later
        cast: function that contructs text before vectorization. By default,
            it uses only question. Check `cast_naive_get_question_and_answer` for more details.
        n_probe: number of closest IVF-clusters to check for neighbours.
            Doesn't affect bruteforce-based search.
        knn_args: check `create_faiss_index` function for details on ANN/KNN arguments.
    Returns: function to search similar Examples from `train` in FAISS-index.
    """
    from dsp.utils.ann_utils import create_faiss_index

    train_casted_to_vectorize = [cast(cur_elem) for cur_elem in train]

    vectorizer: BaseSentenceVectorizer = dsp.settings.vectorizer
    all_vectors = vectorizer(train_casted_to_vectorize).astype(np.float32)

    index = create_faiss_index(
        emb_dim=all_vectors.shape[1], n_objects=len(train), **knn_args,
    )
    index.train(all_vectors)
    index.add(all_vectors)

    def inner_knn_search(inp_example: Example, k: int) -> list[Example]:
        inp_example_vector = vectorizer([cast(inp_example)])
        _, nearest_samples_idxs = index.search(inp_example_vector, k)
        train_sampled = [train[cur_idx] for cur_idx in nearest_samples_idxs[0]]
        return train_sampled

    return inner_knn_search
