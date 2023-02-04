import random
from typing import Callable, List

import numpy as np

import dsp
from dsp.utils import (EM, DPR_normalize, create_faiss_index, dotdict,
                       has_answer, normalize_text)


class Example(dotdict):
    def __init__(self, *args, **kwargs):
        assert len(args) <= 1
        super().__init__()

        if len(args):
            assert len(args) == 1
            self.update(args[0])

        self.update(**kwargs)

    def copy(self, **kwargs):
        the_copy = Example(**{**dict(self), **kwargs})

        return the_copy

    def without(self, *keys):
        keys = set(keys)
        return Example({k: v for k, v in self.items() if k not in keys})

    def demos_at(self, fn):
        def at(d):
            try:
                return fn(d).without('augmented')
            except:
                return {}

        demos = [d.copy(**at(d)) for d in self.demos]
        return self.copy(demos=demos)

    # def __str__(self):
    #     return self.get('question', self.get('query'))


def annotate(*transformations):
    def do_augment(train, k=None, return_all=False):
        rdemos = []
        ademos = []

        for example in (train):  # tqdm.tqdm
            raw_example = dsp.Example(example)

            if k and len(ademos) >= k:
                example = None

            for f in transformations:
                if example is None:
                    break

                example = f(example)

            if example is not None:
                example.augmented = True
                ademos.append(example)
            else:
                raw_example.augmented = False
                rdemos.append(raw_example)

        if return_all:
            return ademos + rdemos

        return ademos

    return do_augment


def sample(train, k: int):
    branch_idx = hasattr(dsp.settings, 'branch_idx') and isinstance(
        dsp.settings.branch_idx, int) and dsp.settings.branch_idx
    branch_idx = branch_idx or 0
    # print(f"branch_idx = {branch_idx}")

    rng = random.Random(branch_idx)
    shuffled_train = list(train)
    rng.shuffle(shuffled_train)

    subset = shuffled_train[:k]
    subset = [dsp.Example(x) for x in subset]  # augmented=False

    return subset


def all_but(train, x):
    # output = [y for y in train if y.question != x.question]

    output = [y for y in train
              if not set.intersection(set(x.get('history', []) + [x.question]), set(y.get('history', []) + [y.question]))]
    # print(len(output))

    return output


def passage_match(passages, answers):
    # passages = example.passages
    # answers = example.answers
    return any(passage_has_answers(psg, answers) for psg in passages)


def answer_match(prediction, answers):
    # pred = example.prediction
    # answers = example.answers
    return EM(prediction, answers)


def passage_has_answers(passage, answers):
    return has_answer([DPR_normalize(normalize_text(ans)) for ans in answers], normalize_text(passage))


def cast_naive_get_only_question_text(inp_example: Example) -> Example:
    return inp_example.copy(text_to_vectorize=inp_example.question)


def cast_naive_get_question_and_answer(inp_example: Example) -> Example:
    text_to_vectorize = inp_example.question.strip() + " Answer: " + inp_example.answer.strip()
    return inp_example.copy(text_to_vectorize=text_to_vectorize)


def vectorize_naive_get_field(inp_examples: List[Example]) -> np.ndarray:
    embeddings = [cur_example.vectorized.reshape(1, -1) for cur_example in inp_examples]
    embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
    return embeddings


def knn(
    train: List[Example],
    cast: Callable[[Example], Example] = cast_naive_get_only_question_text,
    vectorize: Callable[[List[Example]], np.ndarray] = vectorize_naive_get_field,
    vectorize_train_bs: int = 128,
    **knn_args
) -> Callable[[Example, int], List[Example]]:
    need_add_one = len(train) % vectorize_train_bs != 0
    n_batches = len(train) // vectorize_train_bs + int(need_add_one)
    emb_dim = vectorize(train[:1]).shape[1]
    all_vectors = np.empty((len(train), emb_dim), dtype=np.float32)
    # batched vectorization for train Examples
    for batch_idx in range(n_batches):
        batch_start_idx = batch_idx * vectorize_train_bs
        batch_end_idx = (batch_idx + 1) * vectorize_train_bs
        cur_batch = train[batch_start_idx: batch_end_idx]
        cur_batch = [cast(cur_elem) for cur_elem in cur_batch]
        cur_batch_vectors = vectorize(cur_batch)
        all_vectors[batch_start_idx:batch_end_idx, :] = cur_batch_vectors

    index = create_faiss_index(emb_dim=emb_dim, n_objects=len(train), **knn_args)
    index.train(all_vectors)
    index.add(all_vectors)

    def inner_knn_search(inp_example: Example, k: int) -> List[Example]:
        inp_example_vector = vectorize([inp_example])
        _, nearest_samples_idxs = index.search(inp_example_vector, k)
        train_sampled = [train[cur_idx] for cur_idx in nearest_samples_idxs[0]]
        return train_sampled

    return inner_knn_search
