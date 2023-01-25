import dsp
import random

from dsp.utils import dotdict, has_answer, DPR_normalize, normalize_text, EM


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

        for example in (train): # tqdm.tqdm
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
    branch_idx = hasattr(dsp.settings, 'branch_idx') and isinstance(dsp.settings.branch_idx, int) and dsp.settings.branch_idx
    branch_idx = branch_idx or 0
    # print(f"branch_idx = {branch_idx}")

    rng = random.Random(branch_idx)
    shuffled_train = list(train)
    rng.shuffle(shuffled_train)

    subset = shuffled_train[:k]
    subset = [dsp.Example(x) for x in subset] # augmented=False

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
