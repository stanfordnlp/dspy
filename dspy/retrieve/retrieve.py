import dsp
import random

from dspy.predict.parameter import Parameter
from dspy.primitives.prediction import Prediction


class Retrieve(Parameter):
    name = "Search"
    input_variable = "query"
    desc = "takes a search query and returns one or more potentially relevant passages from a corpus"

    def __init__(self, k=3, **config):
        self.stage = random.randbytes(8).hex()
        self.k = k
        self.reset()

        self.rm = config.get("rm")

    def reset(self):
        self.rm = None

    def dump_state(self):
        state_keys = ["k", "rm"]
        return {k: getattr(self, k) for k in state_keys}

    def load_state(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, query_or_queries, **kwargs):
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [query.strip().split('\n')[0].strip() for query in queries]

        # Get the right RM to use.
        rm = kwargs.pop("rm", None) or self.rm or dsp.settings.rm

        # print(queries)
        # TODO: Consider removing any quote-like markers that surround the query too.

        with dsp.settings.context(rm=rm):
            passages = dsp.retrieveEnsemble(queries, k=self.k)

        return Prediction(passages=passages)



# TODO: Consider doing Prediction.from_completions with the individual sets of passages (per query) too.
