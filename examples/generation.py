from pydantic import BaseModel, Field

import dspy
from dspy.functional import TypedPredictor
from dspy.teleprompt import LabeledFewShot

turbo = dspy.OpenAI(model='gpt-3.5-turbo')
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)

class SyntheticFact(BaseModel):
    fact: str = Field(..., description="a statement")
    varacity: bool = Field(..., description="is the statement true or false")

class ExampleSignature(dspy.Signature):
    """Generate an example of a synthetic fact."""
    fact: SyntheticFact = dspy.OutputField()

generator = TypedPredictor(ExampleSignature)
examples = generator(config=dict(n=10))

# If you have examples and want more
existing_examples = [
    dspy.Example(fact="The sky is blue", varacity=True),
    dspy.Example(fact="The sky is green", varacity=False),
]
trained = LabeledFewShot().compile(student=generator, trainset=existing_examples)

augmented_examples = trained(config=dict(n=10))
