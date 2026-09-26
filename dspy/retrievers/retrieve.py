import random

from dspy.predict.parameter import Parameter
from dspy.primitives.prediction import Prediction
from dspy.utils.callback import with_callbacks


def single_query_passage(passages):
    passages_dict = {key: [] for key in list(passages[0].keys())}
    for docs in passages:
        for key, value in docs.items():
            passages_dict[key].append(value)
    if "long_text" in passages_dict:
        passages_dict["passages"] = passages_dict.pop("long_text")
    return Prediction(**passages_dict)


class Retrieve(Parameter):
    name = "Search"
    input_variable = "query"
    desc = "takes a search query and returns one or more potentially relevant passages from a corpus"

    def __init__(self, k=3, callbacks=None):
        self.stage = random.randbytes(8).hex()
        self.k = k
        self.callbacks = callbacks or []

    def reset(self):
        pass

    def dump_state(self, json_mode=True):
        # `json_mode` is part of the Parameter.dump_state contract:
        # BaseModule.dump_state forwards it to every named parameter, so a
        # module that holds a Retrieve could not be dumped or saved without it
        # (#10454). Retrieve's state is a plain int `k`, identical in JSON and
        # pickle form, so the flag is accepted and needs no branch.
        state_keys = ["k"]
        return {k: getattr(self, k) for k in state_keys}

    def load_state(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    @with_callbacks
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        query: str,
        k: int | None = None,
        **kwargs,
    ) -> list[str] | Prediction | list[Prediction]:
        k = k if k is not None else self.k

        import dspy

        if not dspy.settings.rm:
            raise AssertionError("No RM is loaded.")

        passages = dspy.settings.rm(query, k=k, **kwargs)

        from collections.abc import Iterable
        if not isinstance(passages, Iterable):
            # it's not an iterable yet; make it one.
            # TODO: we should unify the type signatures of dspy.Retriever
            passages = [passages]
        passages = [psg.long_text for psg in passages]

        return Prediction(passages=passages)

# TODO: Consider doing Prediction.from_completions with the individual sets of passages (per query) too.
