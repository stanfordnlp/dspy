from dspy.teleprompt.teleprompt import Teleprompter
from dspy.primitives import Module
import dspy
import random
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2


class MyAgent(Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ReAct("question -> answer", tools=[self.random_function])

    def random_function(self):
        return random.random()


optimizer = MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light")
compiled_agent = optimizer.compile(student=MyAgent(), trainset=trainset, teacher=teacher, valset=valset)
print(compiled_agent.random_function())
