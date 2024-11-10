---
sidebar_position: 2
---

# [02] Multi-Hop Question Answering

A single search query is often not enough for complex QA tasks. For instance, an example within `HotPotQA` includes a question about the birth city of the writer of "Right Back At It Again". A search query often identifies the author correctly as "Jeremy McKinnon", but lacks the capability to compose the intended answer in determining when he was born.

The standard approach for this challenge in retrieval-augmented NLP literature is to build multi-hop search systems, like GoldEn (Qi et al., 2019) and Baleen (Khattab et al., 2021). These systems read the retrieved results and then generate additional queries to gather additional information when necessary before arriving to a final answer. Using DSPy, we can easily simulate such systems in a few lines of code.

## Configuring LM and RM

We'll start by setting up the language model (LM) and retrieval model (RM), which **DSPy** supports through multiple [LM](/category/language-model-clients) and [RM](/category/retrieval-model-clients) APIs and [local models hosting](/category/local-language-model-clients).

In this notebook, we'll work with GPT-3.5 (`gpt-3.5-turbo`) and the `ColBERTv2` retriever (a free server hosting a Wikipedia 2017 "abstracts" search index containing the first paragraph of each article from this [2017 dump](https://hotpotqa.github.io/wiki-readme.html)). We configure the LM and RM within DSPy, allowing DSPy to internally call the respective module when needed for generation or retrieval. 

```python
import dspy

turbo = dspy.OpenAI(model='gpt-3.5-turbo')
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)
```

## Loading the Dataset

For this tutorial, we make use of the mentioned `HotPotQA` dataset, a collection of complex question-answer pairs typically answered in a multi-hop fashion. We can load this dataset provided by DSPy through the `HotPotQA` class:

```python
from dspy.datasets import HotPotQA

# Load the dataset.
dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)

# Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

len(trainset), len(devset)
```
**Output:**
```text
(20, 50)
```

## Building Signature

Now that we have the data loaded let's start defining the signatures for sub-tasks of out Baleen pipeline.

We'll start by creating the `GenerateAnswer` signature that'll take `context` and `question` as input and give `answer` as output.

```python
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
```

Unlike usual QA pipelines, we have an intermediate question-generation step in Baleen for which we'll need to define a new Signature for the "hop" behavior: inputting some context and a question to generate a search query to find missing information. 

```python
class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()
```

!!! info
    We could have written `context = GenerateAnswer.signature.context` to avoid duplicating the description of the context field.

Now that we have the necessary signatures in place, we can start building the Baleen pipeline!

## Building the Pipeline

So, let's define the program itself `SimplifiedBaleen`. There are many possible ways to implement this, but we'll keep this version down to the key elements.

```python
from dsp.utils import deduplicate

class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)
```

As we can see, the `__init__` method defines a few key sub-modules:

- **generate_query**: For each hop, we will have one `dspy.ChainOfThought` predictor with the `GenerateSearchQuery` signature.
- **retrieve**: This module will conduct the search using the generated queries over our defined ColBERT RM search index via the `dspy.Retrieve` module.
- **generate_answer**: This `dspy.Predict` module will be used with the `GenerateAnswer` signature to produce the final answer.

The `forward` method uses these sub-modules in simple control flow.

1. First, we'll loop up to `self.max_hops` times.
2. In each iteration, we'll generate a search query using the predictor at `self.generate_query[hop]`.
3. We'll retrieve the top-k passages using that query.
4. We'll add the (deduplicated) passages to our `context` accumulator.
5. After the loop, we'll use `self.generate_answer` to produce an answer.
6. We'll return a prediction with the retrieved `context` and predicted `answer`.

## Executing the Pipeline

Let's execute this program in its zero-shot (uncompiled) setting. 

This doesn't necessarily imply the performance will be bad but rather that we're bottlenecked directly by the reliability of the underlying LM to understand our sub-tasks from minimal instructions. Often, this is perfectly fine when using the most expensive/powerful models (e.g., GPT-4) on the easiest and most standard tasks (e.g., answering simple questions about popular entities).

```python
# Ask any question you like to this simple RAG program.
my_question = "How many storeys are in the castle that David Gregory inherited?"

# Get the prediction. This contains `pred.context` and `pred.answer`.
uncompiled_baleen = SimplifiedBaleen()  # uncompiled (i.e., zero-shot) program
pred = uncompiled_baleen(my_question)

# Print the contexts and the answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")
print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")
```
**Output:**
```text
Question: How many storeys are in the castle that David Gregory inherited?
Predicted Answer: five
Retrieved Contexts (truncated): ['David Gregory (physician) | David Gregory (20 December 1625 – 1720) was a Scottish physician and inventor. His surname is sometimes spelt as Gregorie, the original Scottish spelling. He inherited Kinn...', 'The Boleyn Inheritance | The Boleyn Inheritance is a novel by British author Philippa Gregory which was first published in 2006. It is a direct sequel to her previous novel "The Other Boleyn Girl," an...', 'Gregory of Gaeta | Gregory was the Duke of Gaeta from 963 until his death. He was the second son of Docibilis II of Gaeta and his wife Orania. He succeeded his brother John II, who had left only daugh...', 'Kinnairdy Castle | Kinnairdy Castle is a tower house, having five storeys and a garret, two miles south of Aberchirder, Aberdeenshire, Scotland. The alternative name is Old Kinnairdy....', 'Kinnaird Head | Kinnaird Head (Scottish Gaelic: "An Ceann Àrd" , "high headland") is a headland projecting into the North Sea, within the town of Fraserburgh, Aberdeenshire on the east coast of Scotla...', 'Kinnaird Castle, Brechin | Kinnaird Castle is a 15th-century castle in Angus, Scotland. The castle has been home to the Carnegie family, the Earl of Southesk, for more than 600 years....']
```

We can inspect the last **three** calls to the LM (i.e., generating the first hop's query, generating the second hop's query, and generating the answer) using:

```python
turbo.inspect_history(n=3)
```

## Optimizing the Pipeline

However, a zero-shot approach quickly falls short for more specialized tasks, novel domains/settings, and more efficient (or open) models. 

To address this, **DSPy** offers compilation. Let's compile our multi-hop (`SimplifiedBaleen`) program. 

Let's first define our validation logic for compilation: 

- The predicted answer matches the gold answer.
- The retrieved context contains the gold answer.
- None of the generated queries is rambling (i.e., none exceeds 100 characters in length).
- None of the generated queries is roughly repeated (i.e., none is within 0.8 or higher F1 score of earlier queries).

```python
def validate_context_and_answer_and_hops(example, pred, trace=None):
    if not dspy.evaluate.answer_exact_match(example, pred): return False
    if not dspy.evaluate.answer_passage_match(example, pred): return False

    hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]

    if max([len(h) for h in hops]) > 100: return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False

    return True
```

We'll use one of the most basic teleprompters in **DSPy**, namely, `BootstrapFewShot` to optimize the predictors in pipeline with few-shot examples.

```python
from dspy.teleprompt import BootstrapFewShot

teleprompter = BootstrapFewShot(metric=validate_context_and_answer_and_hops)
compiled_baleen = teleprompter.compile(SimplifiedBaleen(), teacher=SimplifiedBaleen(passages_per_hop=2), trainset=trainset)
```

## Evaluating the Pipeline

Let's now define our evaluation function and compare the performance of the uncompiled and compiled Baleen pipelines. While this devset does not serve as a completely reliable benchmark, it is instructive to use for this tutorial. 

```python
from dspy.evaluate.evaluate import Evaluate

# Define metric to check if we retrieved the correct documents
def gold_passages_retrieved(example, pred, trace=None):
    gold_titles = set(map(dspy.evaluate.normalize_text, example["gold_titles"]))
    found_titles = set(
        map(dspy.evaluate.normalize_text, [c.split(" | ")[0] for c in pred.context])
    )
    return gold_titles.issubset(found_titles)

# Set up the `evaluate_on_hotpotqa` function. We'll use this many times below.
evaluate_on_hotpotqa = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)

uncompiled_baleen_retrieval_score = evaluate_on_hotpotqa(uncompiled_baleen, metric=gold_passages_retrieved, display=False)

compiled_baleen_retrieval_score = evaluate_on_hotpotqa(compiled_baleen, metric=gold_passages_retrieved)

print(f"## Retrieval Score for uncompiled Baleen: {uncompiled_baleen_retrieval_score}")
print(f"## Retrieval Score for compiled Baleen: {compiled_baleen_retrieval_score}")
```
**Output:**
```text
## Retrieval Score for uncompiled Baleen: 36.0
## Retrieval Score for compiled Baleen: 60.0
```

Excellent! There might be something to this compiled, multi-hop program then.

Earlier, we said simple programs are not very effective at finding all evidence required for answering each question. Is this resolved by the adding some greater prompting techniques in the forward function of `SimplifiedBaleen`? Does compiling programs improve performance?

While in our tutorial we demonstrate our findings, the answer for these questions will not always be obvious. However, DSPy makes it extremely easy to try out the many diverse approaches with minimal effort. 

Now that you've seen a example of how to build a simple yet powerful pipeline, it's time for you to build one yourself!
