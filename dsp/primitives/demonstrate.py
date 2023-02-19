import random
from typing import Callable, List

import numpy as np

import dsp
from dsp.utils import EM, F1, DPR_normalize, dotdict, has_answer, normalize_text



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
    branch_idx = (
        hasattr(dsp.settings, 'branch_idx')
        and isinstance(dsp.settings.branch_idx, int)
        and dsp.settings.branch_idx
    )
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

    output = [
        y for y in train
        if not set.intersection(
            set(x.get('history', []) + [x.question]),
            set(y.get('history', []) + [y.question])
        )
    ]
    # print(len(output))

    return output


def passage_match(passages, answers):
    # passages = example.passages
    # answers = example.answers
    return any(passage_has_answers(psg, answers) for psg in passages)

def answer_match(prediction, answers, frac=1.0):
    # pred = example.prediction
    # answers = example.answers

    if frac >= 1.0:
        return EM(prediction, answers)
    
    return F1(prediction, answers) >= frac


def passage_has_answers(passage, answers):
    return has_answer(
        tokenized_answers=[DPR_normalize(normalize_text(ans)) for ans in answers],
        text=normalize_text(passage)
    )


def cast_naive_get_only_question_text(inp_example: Example) -> Example:
    '''
    Extracts question as a field to vectorize with Vectorizer object. `question` field is used.
    '''
    return inp_example.copy(text_to_vectorize=inp_example.question)


def cast_naive_get_question_and_answer(inp_example: Example) -> Example:
    '''
    Extracts question and answer as fields to vectorize with Vectorizer object.
    `question` and `answer` fields are used. They will be concatenated with the word "Answer"
    between.
    '''
    text_to_vectorize = inp_example.question.strip() + " Answer: " + inp_example.answer.strip()
    return inp_example.copy(text_to_vectorize=text_to_vectorize)


def knn(
    train: List[Example],
    cast: Callable[[Example], Example] = cast_naive_get_only_question_text,
    **knn_args
) -> Callable[[Example, int], List[Example]]:
    '''
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
    '''
    from dsp.utils.ann_utils import create_faiss_index
    train_casted_to_vectorize = [cast(cur_elem) for cur_elem in train]

    vectorizer: "BaseSentenceVectorizer" = dsp.settings.vectorizer
    all_vectors = vectorizer(train_casted_to_vectorize).astype(np.float32)

    index = create_faiss_index(emb_dim=all_vectors.shape[1], n_objects=len(train), **knn_args)
    index.train(all_vectors)
    index.add(all_vectors)

    def inner_knn_search(inp_example: Example, k: int) -> List[Example]:
        inp_example_vector = vectorizer([cast(inp_example)])
        _, nearest_samples_idxs = index.search(inp_example_vector, k)
        train_sampled = [train[cur_idx] for cur_idx in nearest_samples_idxs[0]]
        return train_sampled

    return inner_knn_search
