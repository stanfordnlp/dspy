# Typed Predictors

In DSPy Signatures, we have `InputField` and `OutputField` that define the nature of inputs and outputs of the field. However the inputs and output to these fields is always string, you can seperately process these inputs and outputs in complicated scenarios but the inherent type is always `str`.

Pydantic `BaseModel` is a great way to enforce type constraints on the fields, but it is not directly compatible with the `dspy.Signature`. This is where Typed Predictors come in. They are a way to enforce the type constraints on the inputs and outputs of the fields in a `dspy.Signature`.

## Executing Typed Predictors

Using Typed Predictors is not too different than any other module, infact aside from adding type hints to signature attributes and using a special Predictor module instead of `dspy.Predict` there is nothing else to do. Let's take a look at a simple example to understand this.

### Defining Input and Output Models

Let's take an simple task as example i.e. given the `context` and `query` the LLM should give me an `answer` and `confidence_score`. The task could be modelled better but this is just for illustration purposes. Let's define our `Input` and `Output` models via pydantic.

```python
from pydantic import BaseModel, Field

class Input(BaseModel):
    context: str = Field(..., description="The context for the question")
    query: str = Field(..., description="The question to be answered")

class Output(BaseModel):
    answer: str = Field(..., description="The answer for the question")
    factual_: float = Field(..., description="The confidence score for the answer")
```

As you can see this is where you can provide description to the attributes now. Now that we have the input and output models, let's define a simple Signature that takes in the input and returns the output.

### Creating Typed Predictor

A Typed Predictor needs a Typed Signature which is not any different than a normal `dspy.Signature` everything is the same except here you provide type of each field as well.

```python
class QASignature(dspy.Signature):
    """Answer the question based on the context and query provided, and on the scale of 10 tell how confident you are about the answer."""

    input: Input = dspy.InputField()
    output: Output = dspy.OutputField()
```

Now that we have the `QASignature`, let's define a Typed Predictor that let's use execute this Signature while conforming to the type constraints.

```python
predictor = dspy.TypedPredictor(QASignature)
```

Just how we pass the Signature to other modules we pass the `QASignature` to `dspy.TypedPredictor`, where typed constraints are inforced.

### I/O in Typed Predictors

Now that we have the Typed Predictor let's test it out by providing some sample input to the predictor and see the output and it's type. We can create a `Input` instance and pass it to the predictor to get a dictionary of the output. 

```python
doc_query_pair = Input(
    context="The quick brown fox jumps over the lazy dog",
    query="What does the fox jumps over?",
)

prediction = predictor(input=doc_query_pair)
```

Now that we have the prediction, we can see the output and it's type.

```python
answer = prediction['answer']
confidence_score = prediction['confidence_score']

print(f"Prediction: {prediction}\n\n")
print(f"Answer: {answer}, Answer Type: {type(answer)}")
print(f"Confidence Score: {confidence_score}, Confidence Score Type: {type(confidence_score)}")
```

## Typed Chain of Thoughts with `dspy.TypedChainOfThought`

If `TypedPredictor` is the typed counterpart of `dspy.Predict` then `TypedChainOfThought` is the typed counterpart of `dspy.ChainOfThought`. It adds a Chain of Thoughts `dspy.OutputField` to the `dspy.TypedPredictor` module by prepending it to the Signature.

```python
cot_predictor = dspy.TypedChainOfThought(QASignature)

doc_query_pair = Input(
    context="The quick brown fox jumps over the lazy dog",
    query="What does the fox jumps over?",
)

prediction = cot_predictor(input=doc_query_pair)
```

## Typed Predictors as Decorators

While the `dspy.TypedPredictor` and `dspy.TypedChainOfThought` provide a convinient way to use typed predictors, you can also use their as decorators to enforce type constraints on the inputs and outputs of the function. Good thing is that you won't need to explicitly define a Signature class because it's created internally based on function arguments, outputs and docstring.

```
# Function name is output key

@dspy.predictor
def qa_function(doc_query_pair: Input) -> Output:
    """Answer the question based on the context and query provided, and on the scale of 10 tell how confident you are about the answer."""
    pass

@dspy.cot
def qa_function(doc_query_pair: Input) -> Output:
    """Answer the question based on the context and query provided, and on the scale of 10 tell how confident you are about the answer."""
    pass
```

## Composing Functional Typed Predictors in `dspy.Module`

If you're creating DSPy pipelines via `dspy.Module` then you can simply use Functional Typed Predictors by creating these class methods and using them as decorators. Here is an example of using functional typed predictors to create a `SimplifiedBaleen` pipeline:

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

Typed predictors can be optimized using `optimize_signature` optimizer which optimizes the instructions of the Signature. Here is an example of using `optimize_signature` to optimize the `QASignature`:

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