import random
import uuid

from dsp.utils import dotdict
from dspy import Example


class Dataset:
    def __init__(self, train_seed=0, train_size=None, eval_seed=0, dev_size=None, test_size=None, input_keys=[]):
        self.train_size = train_size
        self.train_seed = train_seed
        self.dev_size = dev_size
        self.dev_seed = eval_seed
        self.test_size = test_size
        self.test_seed = eval_seed
        self.input_keys = input_keys

        self.do_shuffle = True

        self.name = self.__class__.__name__

    def reset_seeds(self, train_seed=None, train_size=None, eval_seed=None, dev_size=None, test_size=None):
        self.train_size = train_size if train_size is not None else self.train_size
        self.train_seed = train_seed if train_seed is not None else self.train_seed
        self.dev_size = dev_size if dev_size is not None else self.dev_size
        self.dev_seed = eval_seed if eval_seed is not None else self.dev_seed
        self.test_size = test_size if test_size is not None else self.test_size
        self.test_seed = eval_seed if eval_seed is not None else self.test_seed

        if hasattr(self, '_train_'):
            del self._train_
        
        if hasattr(self, '_dev_'):
            del self._dev_
        
        if hasattr(self, '_test_'):
            del self._test_

    @property
    def train(self):
        if not hasattr(self, '_train_'):
            self._train_ = self._shuffle_and_sample('train', self._train, self.train_size, self.train_seed)

        return self._train_

    @property
    def dev(self):
        if not hasattr(self, '_dev_'):
            self._dev_ = self._shuffle_and_sample('dev', self._dev, self.dev_size, self.dev_seed)

        return self._dev_
    
    @property
    def test(self):
        if not hasattr(self, '_test_'):
            self._test_ = self._shuffle_and_sample('test', self._test, self.test_size, self.test_seed)

        return self._test_

    def _shuffle_and_sample(self, split, data, size, seed=0):
        '''
            The setting (seed=s, size=N) is always a subset
            of the setting (seed=s, size=M) for N < M.
        '''

        data = list(data)

        # Shuffle the data irrespective of the requested size.
        base_rng = random.Random(seed)

        if self.do_shuffle:
            base_rng.shuffle(data)

        data = data[:size]
        output = []

        for example in data:
            example_obj = Example(**example, dspy_uuid=str(uuid.uuid4()), dspy_split=split)
            if self.input_keys:
                example_obj = example_obj.with_inputs(*self.input_keys)
            output.append(example_obj)
        # TODO: NOTE: Ideally we use these uuids for dedup internally, for demos and internal train/val splits.
        # Now, some tasks (like convQA and Colors) have overlapping examples. Here, we should allow the user to give us
        # a uuid field that would respect this in some way. This means that we need a more refined concept that
        # uuid (each example is unique) and more like a group_uuid.

        # rng = random.Random(seed)
        # rng.shuffle(data)

        return output
    
    @classmethod
    def prepare_by_seed(cls, train_seeds=[1,2,3,4,5], train_size=16, dev_size=1000,
                        divide_eval_per_seed=True, eval_seed=2023, **kwargs):
        
        data_args = dotdict(train_size=train_size, eval_seed=eval_seed, dev_size=dev_size, test_size=0, **kwargs)
        dataset = cls(**data_args)

        eval_set = dataset.dev
        eval_sets, train_sets = [], []

        examples_per_seed = dev_size // len(train_seeds) if divide_eval_per_seed else dev_size
        eval_offset = 0

        for train_seed in train_seeds:
            data_args.train_seed = train_seed
            dataset.reset_seeds(**data_args)

            eval_sets.append(eval_set[eval_offset:eval_offset+examples_per_seed])
            train_sets.append(dataset.train)

            assert len(eval_sets[-1]) == examples_per_seed, len(eval_sets[-1])
            assert len(train_sets[-1]) == train_size, len(train_sets[-1])
            
            if divide_eval_per_seed:
                eval_offset += examples_per_seed

        return dotdict(train_sets=train_sets, eval_sets=eval_sets)
