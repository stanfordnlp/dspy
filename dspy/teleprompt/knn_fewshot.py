import types

from dspy.clients import Embedder
from dspy.predict.knn import KNN
from dspy.primitives import Example
from dspy.teleprompt import BootstrapFewShot

from .teleprompt import Teleprompter


class KNNFewShot(Teleprompter):
    def __init__(self, k: int, trainset: list[Example], vectorizer: Embedder, **few_shot_bootstrap_args):
        """
        KNNFewShot is an optimizer that uses an in-memory KNN retriever to find the k nearest neighbors
        in a trainset at test time. For each input example in a forward call, it identifies the k most
        similar examples from the trainset and attaches them as demonstrations to the student module.

        Args:
            k: The number of nearest neighbors to attach to the student model.
            trainset: The training set to use for few-shot prompting.
            vectorizer: The `Embedder` to use for vectorization
            **few_shot_bootstrap_args: Additional arguments for the `BootstrapFewShot` optimizer.

        Example:
            >>> import dspy
            >>> from sentence_transformers import SentenceTransformer
            >>>
            >>> qa = dspy.ChainOfThought("question -> answer")
            >>> trainset = [dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"), ...]
            >>> knn_few_shot = KNNFewShot(k=3, trainset=trainset, vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode))
            >>> compiled_qa = knn_few_shot.compile(qa)
            >>> compiled_qa("What is the capital of Belgium?")
        """
        self.KNN = KNN(k, trainset, vectorizer=vectorizer)
        self.few_shot_bootstrap_args = few_shot_bootstrap_args

    def compile(self, student, *, teacher=None):
        student_copy = student.reset_copy()

        def forward_pass(_, **kwargs):
            knn_trainset = self.KNN(**kwargs)
            few_shot_bootstrap = BootstrapFewShot(**self.few_shot_bootstrap_args)
            compiled_program = few_shot_bootstrap.compile(
                student,
                teacher=teacher,
                trainset=knn_trainset,
            )
            return compiled_program(**kwargs)

        student_copy.forward = types.MethodType(forward_pass, student_copy)
        return student_copy
