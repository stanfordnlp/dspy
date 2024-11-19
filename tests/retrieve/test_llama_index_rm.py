import logging

import pytest

import dspy
from dspy.datasets import HotPotQA
from dspy.utils.dummies import DummyLM

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
    pytest.importorskip("llama_index")
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
        "lm": DummyLM(answers=dummyset),
        "trainset": trainset,
        "devset": devset,
    }


def test_lirm_as_rm(rag_setup):
    """Test the retriever as retriever method"""
    pytest.importorskip("llama_index")
    retriever = rag_setup.get("retriever")
    test_res_li = retriever.retrieve("At My Window was released by which American singer-songwriter?")
    rm = rag_setup.get("rm")
    test_res_dspy = rm.forward("At My Window was released by which American singer-songwriter?")

    assert isinstance(retriever, BaseRetriever), "Ensuring that the retriever is a LI Retriever object"
    assert isinstance(test_res_li, list), "Ensuring results are a list from LI Retriever"

    assert isinstance(rm, dspy.Retrieve), "Ensuring the RM is a retriever object from dspy"
    assert isinstance(test_res_dspy, list), "Ensuring the results are a list from the DSPy retriever"

    assert len(test_res_li) == len(test_res_dspy), "Rough equality check of the results"


def test_save_load_llama_index_rag(rag_setup, tmp_path):
    pytest.importorskip("llama_index")

    class RAG(dspy.Module):
        def __init__(self):
            super().__init__()
            self.retriever = dspy.Retrieve(k=3)
            self.cot = dspy.ChainOfThought("question, context -> answer")

    rag = RAG()
    rag.retriever.k = 4

    file_path = tmp_path / "rag.json"
    rag.save(file_path)
    loaded_rag = RAG()
    # Before loading, the retriever k should be 3.
    assert loaded_rag.retriever.k == 3
    # After loading, the retriever k should be 4.
    loaded_rag.load(file_path)
    assert loaded_rag.retriever.k == 4
