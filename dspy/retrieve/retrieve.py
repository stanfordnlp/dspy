import dsp
import random

from dspy.predict.parameter import Parameter
from dspy.primitives.prediction import Prediction


class Retrieve(Parameter):
    def __init__(self, k=3):
        self.stage = random.randbytes(8).hex()
        self.k = k
    
    def reset(self):
        pass
    
    def dump_state(self):
        state_keys = ["k"]
        return {k: getattr(self, k) for k in state_keys}

    def load_state(self, state):
        for name, value in state.items():
            setattr(self, name, value)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, query_or_queries):
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries

        passages = dsp.retrieveEnsemble(queries, k=self.k)
        return Prediction(passages=passages)
    

# TODO: Consider doing Prediction.from_completions with the individual sets of passages (per query) too.