import logging

import pytest

import dspy
from dsp.modules.dummy_lm import DSPDummyLM
from dspy.datasets import HotPotQA

try:
    from llama_index.core import Settings, VectorStoreIndex
    from llama_index.core.base.base_retriever import BaseRetriever
    from llama_index.core.embeddings.mock_embed_model import MockEmbedding
    from llama_index.core.readers.string_iterable import StringIterableReader

    from dspy.retrieve.llama_index_rm import LlamaIndexRM

except ImportError:
    logging.info("Optional dependency llama-index is not installed - skipping LlamaIndexRM tests.")


@pytest.fixture()
def rag_setup() -> dict:
    """Builds the necessary fixtures to test LI"""
    pytest.importorskip("llamaindex")
    dataset = HotPotQA(train_seed=1, train_size=8, eval_seed=2023, dev_size=4, test_size=0)
    trainset = [x.with_inputs("question") for x in dataset.train]
    devset = [x.with_inputs("question") for x in dataset.dev]
    ragset = [f"Question: {x.question} Answer: {x.answer}" for x in dataset.train]
    dummyset = {x.question: x.answer for x in dataset.train}

    Settings.embed_model = MockEmbedding(8)
    docs = StringIterableReader().load_data(texts=ragset)
    index = VectorStoreIndex.from_documents(documents=docs)
    retriever = index.as_retriever()
    rm = LlamaIndexRM(retriever)

    return {
        "index": index,
        "retriever": retriever,
        "rm": rm,
        "lm": DSPDummyLM(answers=dummyset),
        "trainset": trainset,
        "devset": devset,
    }


def test_lirm_as_rm(rag_setup):
    """Test the retriever as retriever method"""
    pytest.importorskip("llamaindex")
    retriever = rag_setup.get("retriever")
    test_res_li = retriever.retrieve("At My Window was released by which American singer-songwriter?")
    rm = rag_setup.get("rm")
    test_res_dspy = rm.forward("At My Window was released by which American singer-songwriter?")

    assert isinstance(retriever, BaseRetriever), "Ensuring that the retriever is a LI Retriever object"
    assert isinstance(test_res_li, list), "Ensuring results are a list from LI Retriever"

    assert isinstance(rm, dspy.Retrieve), "Ensuring the RM is a retriever object from dspy"
    assert isinstance(test_res_dspy, list), "Ensuring the results are a list from the DSPy retriever"

    assert len(test_res_li) == len(test_res_dspy), "Rough equality check of the results"
