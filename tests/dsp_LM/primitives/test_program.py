import dspy
from dspy.primitives.program import Module, set_attribute_by_name  # Adjust the import based on your file structure
from dspy.utils import DSPDummyLM


class HopModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict1 = dspy.Predict("question -> query")
        self.predict2 = dspy.Predict("query -> answer")

    def forward(self, question):
        query = self.predict1(question=question).query
        return self.predict2(query=query)


def test_forward():
    program = HopModule()
    dspy.settings.configure(lm=DSPDummyLM({"What is 1+1?": "let me check", "let me check": "2"}))
    result = program(question="What is 1+1?").answer
    assert result == "2"
