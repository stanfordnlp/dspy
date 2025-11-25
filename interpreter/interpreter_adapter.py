from dspy import Adapter
import dspy
from dspy.adapters import ChatAdapter
from dspy.clients.lm import LM
from typing import Any, Dict, List, Type
from dspy.signatures.signature import Signature, make_signature
import json
from dspy.utils import download



class Interpreter(ChatAdapter):
    def __init__(self, interpreter_model: LM, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(interpreter_model, LM):
            raise ValueError("interpreter_model must be an instance of LM")
        self.interpreter_model = interpreter_model

    def format(
        self, signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]
    ) -> list[dict[str, Any]]:
        # We are going to change the instructions of the signature before calling super.format(...)
        print("Old instructions:")
        print(signature.instructions)
        new_intructions= "Answer the question using the provided context and question."
        signature.instructions = new_intructions
        print("New instructions:")
        print(signature.instructions)
        return super().format(signature, demos, inputs)

class QA(dspy.Signature):
    """Answer the question using the provided context and question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

def main():
    lm = dspy.LM("gpt-5-nano")
    dspy.settings.configure(lm=lm, adapter = Interpreter(interpreter_model=lm))


    # download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_examples.jsonl")

    with open("ragqa_arena_tech_examples.jsonl") as f:
        raw = [json.loads(line) for line in f]

    # Each d is like {"question": "...", "response": "...", "gold_doc_ids": [...]}
    dataset = [dspy.Example(**d).with_inputs("question") for d in raw]

    print(dataset[0])
    pred = dspy.Predict(QA)
    
    # print(pred(question="What is the capital of France?", context="The capital of France is Paris."))


    
if __name__ == "__main__":
    main()


