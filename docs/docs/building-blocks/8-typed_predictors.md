# Typed Predictors

In DSPy Signatures, we have `InputField` and `OutputField` that define the nature of inputs and outputs of the field. However, the inputs and output to these fields are always `str`-typed, which requires input and output string processing.

Pydantic `BaseModel` is a great way to enforce type constraints on the fields, but it is not directly compatible with the `dspy.Signature`. Typed Predictors resolves this as a way to enforce the type constraints on the inputs and outputs of the fields in a `dspy.Signature`.

## Executing Typed Predictors

Using Typed Predictors is not too different than any other module with the minor additions of type hints to signature attributes and using a special Predictor module instead of `dspy.Predict`. Let's take a look at a simple example to understand this.

### Defining Input and Output Models

Let's take a simple task as an example i.e. given the `context` and `query`, the LLM should return an `answer` and `confidence_score`. Let's define our `Input` and `Output` models via pydantic.

```python
from pydantic import BaseModel, Field

class Input(BaseModel):
    context: str = Field(description="The context for the question")
    query: str = Field(description="The question to be answered")

class Output(BaseModel):
    answer: str = Field(description="The answer for the question")
    confidence: float = Field(ge=0, le=1, description="The confidence score for the answer")
```

As you can see, we can describe the attributes by defining a simple Signature that takes in the input and returns the output.

### Creating Typed Predictor

A Typed Predictor needs a Typed Signature, which extends a `dspy.Signature` with the addition of specifying "field type".

```python
class QASignature(dspy.Signature):
    """Answer the question based on the context and query provided, and on the scale of 10 tell how confident you are about the answer."""

    input: Input = dspy.InputField()
    output: Output = dspy.OutputField()
```

Now that we have the `QASignature`, let's define a Typed Predictor that executes this Signature while conforming to the type constraints.

```python
predictor = dspy.TypedPredictor(QASignature)
```

Similar to other modules, we pass the `QASignature` to `dspy.TypedPredictor` which enforces the typed constraints.

And similarly to `dspy.Predict`, we can also use a "string signature", which we type as:
```python
predictor = dspy.TypedPredictor("input:Input -> output:Output")
```

### I/O in Typed Predictors

Now let's test out the Typed Predictor by providing some sample input to the predictor and verifying the output type. We can create an `Input` instance and pass it to the predictor to get a dictionary of the output. 

```python
doc_query_pair = Input(
    context="The quick brown fox jumps over the lazy dog",
    query="What does the fox jumps over?",
)

prediction = predictor(input=doc_query_pair)
```

Let's see the output and its type.

```python
answer = prediction.output.answer
confidence_score = prediction.output.confidence

print(f"Prediction: {prediction}\n\n")
print(f"Answer: {answer}, Answer Type: {type(answer)}")
print(f"Confidence Score: {confidence_score}, Confidence Score Type: {type(confidence_score)}")
```

## Typed Chain of Thoughts with `dspy.TypedChainOfThought`

Extending the analogous comparison of `TypedPredictor` to `dspy.Predict`, we create `TypedChainOfThought`, the typed counterpart of `dspy.ChainOfThought`:

```python
cot_predictor = dspy.TypedChainOfThought(QASignature)

doc_query_pair = Input(
    context="The quick brown fox jumps over the lazy dog",
    query="What does the fox jumps over?",
)

prediction = cot_predictor(input=doc_query_pair)
```

## Typed Predictors as Decorators

While the `dspy.TypedPredictor` and `dspy.TypedChainOfThought` provide a convenient way to use typed predictors, you can also use them as decorators to enforce type constraints on the inputs and outputs of the function. This relies on the internal definitions of the Signature class and its function arguments, outputs, and docstrings.

```python
@dspy.predictor
def answer(doc_query_pair: Input) -> Output:
    """Answer the question based on the context and query provided, and on the scale of 0-1 tell how confident you are about the answer."""
    pass

@dspy.cot
def answer(doc_query_pair: Input) -> Output:
    """Answer the question based on the context and query provided, and on the scale of 0-1 tell how confident you are about the answer."""
    pass

prediction = answer(doc_query_pair=doc_query_pair)
```

## Composing Functional Typed Predictors in `dspy.Module`

If you're creating DSPy pipelines via `dspy.Module`, then you can simply use Functional Typed Predictors by creating these class methods and using them as decorators. Here is an example of using functional typed predictors to create a `SimplifiedBaleen` pipeline:

```python
class SimplifiedBaleen(FunctionalModule):
    def __init__(self, passages_per_hop=3, max_hops=1):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.max_hops = max_hops

    @cot
    def generate_query(self, context: list[str], question) -> str:
        """Write a simple search query that will help answer a complex question."""
        pass

    @cot
    def generate_answer(self, context: list[str], question) -> str:
        """Answer questions with short factoid answers."""
        pass

    def forward(self, question):
        context = []

        for _ in range(self.max_hops):
            query = self.generate_query(context=context, question=question)
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        answer = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=answer)
```

## Optimizing Typed Predictors

Typed predictors can be optimized on the Signature instructions through the `optimize_signature` optimizer. Here is an example of this optimization on the `QASignature`:

```python
import dspy
from dspy.evaluate import Evaluate
from dspy.evaluate.metrics import answer_exact_match
from dspy.teleprompt.signature_opt_typed import optimize_signature

turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=4000)
gpt4 = dspy.OpenAI(model='gpt-4', max_tokens=4000)
dspy.settings.configure(lm=turbo)

evaluator = Evaluate(devset=devset, metric=answer_exact_match, num_threads=10, display_progress=True)

result = optimize_signature(
    student=dspy.TypedPredictor(QASignature),
    evaluator=evaluator,
    initial_prompts=6,
    n_iterations=100,
    max_examples=30,
    verbose=True,
    prompt_model=gpt4,
)
```
