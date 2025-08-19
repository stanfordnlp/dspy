import random
from dataclasses import dataclass
from typing import Any

from dspy.predict.parameter import Parameter
from dspy.primitives.prediction import Prediction
from dspy.utils.callback import with_callbacks


@dataclass
class Document:
    page_content: str
    metadata: dict[str, Any]
    type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
            "type": self.type,
        }


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

    def dump_state(self):
        state_keys = ["k"]
        return {k: getattr(self, k) for k in state_keys}

    def load_state(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    @with_callbacks
    def __call__(self, *args, **kwargs) -> Prediction | list[Document]:
        return self.forward(*args, **kwargs)

    def forward(
        self,
        query: str,
        k: int | None = None,
        return_documents: bool = False,
        **kwargs,
    ) -> Prediction | list[Document]:
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

        docs: list[Document] = []
        for psg in passages:
            if isinstance(psg, Document):
                docs.append(psg)
            elif isinstance(psg, dict):
                page_content = psg.get("page_content", psg.get("long_text", ""))
                # support text OR long_text
                metadata = psg.get("metadata", {})
                _type = psg.get("type", "Document")
                docs.append(
                    Document(page_content=page_content, metadata=metadata, type=_type)
                )
            elif isinstance(psg, str):
                docs.append(Document(page_content=psg, metadata={}, type="Document"))

        if return_documents:
            return docs

        passages = [psg.page_content for psg in docs]

        return Prediction(passages=passages)


# TODO: Consider doing Prediction.from_completions with the individual sets of passages (per query) too.
