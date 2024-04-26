import pytest
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.readers.string_iterable import StringIterableReader

import dspy
from dsp.modules.dummy_lm import DummyLM
from dspy.datasets import HotPotQA
from dspy.evaluate import Evaluate
from dspy.retrieve.llama_index_rm import LlamaIndexRM
from dspy.teleprompt import BootstrapFinetune


class MockQnA(dspy.Signature):
    """Answer questions from given context"""

    context = dspy.InputField(desc="Context to answer questions with")
    question = dspy.InputField(desc="Question to answer")
    answer = dspy.OutputField(desc="Answer to the question")


class MockModule(dspy.Module):
    """Mock pipeline for testing"""

    def __init__(self):
        super().__init__()

        self.retrieve = dspy.Retrieve()
        self.generate_answer = dspy.Predict(MockQnA)

    def forward(self, question) -> dspy.Example:
        context = self.retrieve(question).passages

        answer = self.generate_answer(
            context=context,
            question=question,
        ).answer

        return dspy.Prediction(context=context, question=question, answer=answer)


def validate_context_and_answer(example, pred, trace=None):
    """Copied this from the intro.ipynb"""
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM


@pytest.fixture()
def rag_setup() -> dict:
    """Builds the necessary fixtures to test LI"""
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

    tp = BootstrapFinetune(metric=validate_context_and_answer)

    return {
        "index": index,
        "retriever": retriever,
        "rm": rm,
        "lm": DummyLM(answers=dummyset),
        "tp": tp,
        "trainset": trainset,
        "devset": devset,
    }


def test_lirm_as_rm(rag_setup):
    """Test the retriever as retriever method"""

    retriever = rag_setup.get("retriever")
    test_res_li = retriever.retrieve("At My Window was released by which American singer-songwriter?")
    rm = rag_setup.get("rm")
    test_res_dspy = rm.forward("At My Window was released by which American singer-songwriter?")

    assert isinstance(retriever, BaseRetriever), "Ensuring that the retriever is a LI Retriever object"
    assert isinstance(test_res_li, list), "Ensuring results are a list from LI Retriever"

    assert isinstance(rm, dspy.Retrieve), "Ensuring the RM is a retriever object from dspy"
    assert isinstance(test_res_dspy, list), "Ensuring the results are a list from the DSPy retriever"

    assert len(test_res_li) == len(
        test_res_dspy
    ), "Ensuring that the results are the same length, a rough equality check of the results"
