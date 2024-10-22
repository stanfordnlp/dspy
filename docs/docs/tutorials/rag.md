---
sidebar_position: 1
---

# [01] RAG: Retrieval-Augmented Generation

Retrieval-augmented generation (RAG) is an approach that allows LLMs to tap into a large corpus of knowledge from sources and query its knowledge store to find relevant passages/content and produce a well-refined response.

RAG ensures LLMs can dynamically utilize real-time knowledge even if not originally trained on the subject and give thoughtful answers. However, with this nuance comes greater complexities in setting up refined RAG pipelines. To reduce these intricacies, we turn to **DSPy**, which offers a seamless approach to setting up prompting pipelines!

## Configuring LM and RM

We'll start by setting up the language model (LM) and retrieval model (RM), which **DSPy** supports through multiple [LM](/building-blocks/1-language_models.md) and [RM](/deep-dive/retrieval_models_clients/Azure.md) APIs and [local models hosting](/deep-dive/language_model_clients/local_models/HFClientTGI).

In this notebook, we'll work with GPT-3.5 (`gpt-3.5-turbo`) and the `ColBERTv2` retriever (a free server hosting a Wikipedia 2017 "abstracts" search index containing the first paragraph of each article from this [2017 dump](https://hotpotqa.github.io/wiki-readme.html)). We configure the LM and RM within DSPy, allowing DSPy to internally call the respective module when needed for generation or retrieval. 

```python
import dspy

turbo = dspy.OpenAI(model='gpt-3.5-turbo')
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)
```


## Loading the Dataset

For this tutorial, we make use of the `HotPotQA` dataset, a collection of complex question-answer pairs typically answered in a multi-hop fashion. We can load this dataset provided by DSPy through the `HotPotQA` class:

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

## Building Signatures

Now that we have the data loaded, let's start defining the [signatures](/building-blocks/2-signatures) for the sub-tasks of our pipeline.

We can identify our simple input `question` and output `answer`, but since we are building out a RAG pipeline, we wish to utilize some contextual information from our ColBERT corpus. So let's define our signature: `context, question --> answer`.

```python
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
```

We include small descriptions for the `context` and `answer` fields to define more robust guidelines on what the model will receive and should generate. 

## Building the Pipeline

We will build our RAG pipeline as a [DSPy module](/building-blocks/3-modules) which will require two methods:

* The `__init__` method will simply declare the sub-modules it needs: `dspy.Retrieve` and `dspy.ChainOfThought`. The latter is defined to implement our `GenerateAnswer` signature.
* The `forward` method will describe the control flow of answering the question using the modules we have: Given a question, we'll search for the top-3 relevant passages and then feed them as context for answer generation.


```python
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)
```

## Optimizing the Pipeline

##### Compiling the RAG program

Having defined this program, let's now **compile** it. [Compiling a program](/building-blocks/6-optimizers) will update the parameters stored in each module. In our setting, this is primarily in the form of collecting and selecting good demonstrations for inclusion within the prompt(s).

Compiling depends on three things:

1. **A training set.** We'll just use our 20 question–answer examples from `trainset` above.
1. **A metric for validation.** We'll define a simple `validate_context_and_answer` that checks that the predicted answer is correct and that the retrieved context actually contains the answer.
1. **A specific teleprompter.** The **DSPy** compiler includes a number of **teleprompters** that can optimize your programs.

```python
from dspy.teleprompt import BootstrapFewShot

# Validation logic: check that the predicted answer is correct.
# Also check that the retrieved context does actually contain that answer.
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM

# Set up a basic teleprompter, which will compile our RAG program.
teleprompter = BootstrapFewShot(metric=validate_context_and_answer)

# Compile!
compiled_rag = teleprompter.compile(RAG(), trainset=trainset)
```

!!! info
    **Teleprompters:** Teleprompters are powerful optimizers that can take any program and learn to bootstrap and select effective prompts for its modules. Hence the name which means "prompting at a distance".

    Different teleprompters offer various tradeoffs in terms of how much they optimize cost versus quality, etc. We will used a simple default `BootstrapFewShot` in the example above.

    _If you're into analogies, you could think of this as your training data, your loss function, and your optimizer in a standard DNN supervised learning setup. Whereas SGD is a basic optimizer, there are more sophisticated (and more expensive!) ones like Adam or RMSProp._

## Executing the Pipeline

Now that we've compiled our RAG program, let's try it out.

```python
# Ask any question you like to this simple RAG program.
my_question = "What castle did David Gregory inherit?"

# Get the prediction. This contains `pred.context` and `pred.answer`.
pred = compiled_rag(my_question)

# Print the contexts and the answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")
print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")
```

Excellent. How about we inspect the last prompt for the LM?

```python
turbo.inspect_history(n=1)
```
**Output:**
```text
Answer questions with short factoid answers.

---

Question: At My Window was released by which American singer-songwriter?
Answer: John Townes Van Zandt

Question: "Everything Has Changed" is a song from an album released under which record label ?
Answer: Big Machine Records
...(truncated)
```

Even though we haven't written any of this detailed demonstrations, we see that DSPy was able to bootstrap this 3,000 token prompt for 3-shot retrieval-augmented generation with hard negative passages and uses Chain-of-Thought reasoning within an extremely simply-written program.

This illustrates the power of composition and learning. Of course, this was just generated by a particular teleprompter, which may or may not be perfect in each setting. As you'll see in DSPy, there is a large but systematic space of options you have to optimize and validate with respect to your program's quality and cost.

You can also easily inspect the learned objects themselves.

```python
for name, parameter in compiled_rag.named_predictors():
    print(name)
    print(parameter.demos[0])
    print()
```

## Evaluating the Pipeline

We can now evaluate our `compiled_rag` program on the dev set. Of course, this tiny set is _not_ meant to be a reliable benchmark, but it'll be instructive to use it for illustration.

Let's evaluate the accuracy (exact match) of the predicted answer.

```python
from dspy.evaluate.evaluate import Evaluate

# Set up the `evaluate_on_hotpotqa` function. We'll use this many times below.
evaluate_on_hotpotqa = Evaluate(devset=devset, num_threads=1, display_progress=False, display_table=5)

# Evaluate the `compiled_rag` program with the `answer_exact_match` metric.
metric = dspy.evaluate.answer_exact_match
evaluate_on_hotpotqa(compiled_rag, metric=metric)
```
**Output:**
```text
Average Metric: 22 / 50  (44.0): 100%|██████████| 50/50 [00:00<00:00, 116.45it/s]
Average Metric: 22 / 50  (44.0%)

44.0
```

## Evaluating the Retrieval

It may also be instructive to look at the accuracy of retrieval. While there are multiple ways to do this, we can simply check whether the retrieved passages contain the answer.

We can make use of our dev set which includes the gold titles that should be retrieved.

```python
def gold_passages_retrieved(example, pred, trace=None):
    gold_titles = set(map(dspy.evaluate.normalize_text, example['gold_titles']))
    found_titles = set(map(dspy.evaluate.normalize_text, [c.split(' | ')[0] for c in pred.context]))

    return gold_titles.issubset(found_titles)

compiled_rag_retrieval_score = evaluate_on_hotpotqa(compiled_rag, metric=gold_passages_retrieved)
```
**Output:**
```text
Average Metric: 13 / 50  (26.0): 100%|██████████| 50/50 [00:00<00:00, 671.76it/s]Average Metric: 13 / 50  (26.0%)
```

Although this simple `compiled_rag` program is able to answer a decent fraction of the questions correctly (on this tiny set, over 40%), the quality of retrieval is much lower.

This potentially suggests that the LM is often relying on the knowledge it memorized during training to answer questions. To address this weak retrieval, let's explore a second program that involves more advanced search behavior.
