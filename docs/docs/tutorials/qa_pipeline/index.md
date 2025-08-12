
# Building a Simple Question-Answering Pipeline with DSPy

Have you ever wanted to quickly prototype a question-answering system without diving into complex prompt engineering? **DSPy** makes it  easy.

In this post, we’ll walk through:

* Creating a small Q\&A dataset
* Splitting it into training and validation sets
* Defining a **Signature** in DSPy
* Configuring a language model
* Optimizing with **BootstrapFewShot**
* Testing the results

By the end, you’ll have a **working Q\&A pipeline** in less than 100 lines of Python.

---

## Step 1: Install DSPy

If you haven’t already:

```bash
pip install dspy
```

You’ll also need an OpenAI API key (or Azure equivalent).

---

## Step 2: Create Your Dataset

We’ll start with a small set of question–answer pairs. In real projects, you’d have hundreds or thousands, but for our example 10 will do.

```python
qa_pairs = [
    ("What is the capital of Japan?", "Tokyo"),
    ("Which element has the atomic number 1?", "Hydrogen"),
    ("Who discovered penicillin?", "Alexander Fleming"),
    ("What is the fastest land animal?", "Cheetah"),
    ("Which planet is known as the Red Planet?", "Mars"),
    ("What is the square root of 64?", "8"),
    ("In which year did the Titanic sink?", "1912"),
    ("What is the hardest natural substance?", "Diamond"),
    ("Who developed the theory of relativity?", "Albert Einstein"),
    ("What is the largest ocean on Earth?", "Pacific Ocean")
]
```

---

## Step 3: Convert to DSPy Examples & Split

We’ll convert each Q\&A into a `dspy.Example` and split them into 20% training / 80% validation.

```python
import dspy
import math

dataset = [
    dspy.Example(question=q, answer=a).with_inputs("question")
    for q, a in qa_pairs
]

split_index = math.ceil(len(dataset) * 0.2)
trainset = dataset[:split_index]
valset = dataset[split_index:]

print(f"Training size: {len(trainset)}, Validation size: {len(valset)}")
```

---

## Step 4: Define a Signature

Signatures in DSPy describe **what your model will do** — in this case, take a question and return a short factual answer.

```python
class QA(dspy.Signature):
    """Answer the given question accurately."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="A short, factual answer.")
```

---

## Step 5: Configure Your Language Model

You can swap in any LM you have access to — OpenAI, Azure OpenAI, etc.

```python
lm = dspy.LM("openai/gpt-4o-mini", api_key="YOUR_OPENAI_KEY")
dspy.configure(lm=lm)
```

---

## Step 6: Create a Predictor Module

```python
class QAPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(QA)

    def forward(self, question):
        return self.predict(question=question)
```

---

## Step 7: Define Your Evaluation Metric

We’ll keep it simple: exact match (case-insensitive).
This is a **separate function** so it’s easy to swap for a semantic similarity check later.

```python
def exact_match_metric(gold, pred, trace=None):
    """
    Returns True if the predicted answer matches the gold answer (case-insensitive).
    """
    return pred.answer.strip().lower() == gold.answer.strip().lower()
```

---

## Step 8: Optimize with BootstrapFewShot

This will select the **best 3 examples** from the training set to include in the prompt.

```python
from dspy.teleprompt import BootstrapFewShot

pipeline = QAPredictor()
optimizer = BootstrapFewShot(
    metric=exact_match_metric,
    max_bootstrapped_demos=3
)

optimized_pipeline = optimizer.compile(pipeline, trainset=trainset)
```

---

## Step 9: Test on the Validation Set

```python
print("\nValidation results:")
for example in valset:
    prediction = optimized_pipeline(question=example.question)
    print(f"Q: {example.question}")
    print(f"Predicted: {prediction.answer} | Gold: {example.answer}")
    print("-" * 40)
```

---

## Why This Matters

* **No manual prompt crafting** — you just describe your task in the Signature.
* **Automatic example selection** — `BootstrapFewShot` finds the best few-shot examples.
* **Modular design** — you can swap in new datasets, models, or metrics easily.

---

## Next Steps

* Try a **semantic similarity metric** using `sentence-transformers`.
* Expand your dataset to cover a specific domain (finance, medical, etc.).
* Experiment with different LMs to see accuracy differences.

---
## Full script

```

import dspy
import math

# Step 1: Create a dataset of Q&A pairs

qa_pairs = [
    ("What is the capital of Japan?", "Tokyo"),
    ("Which element has the atomic number 1?", "Hydrogen"),
    ("Who discovered penicillin?", "Alexander Fleming"),
    ("What is the fastest land animal?", "Cheetah"),
    ("Which planet is known as the Red Planet?", "Mars"),
    ("What is the square root of 64?", "8"),
    ("In which year did the Titanic sink?", "1912"),
    ("What is the hardest natural substance?", "Diamond"),
    ("Who developed the theory of relativity?", "Albert Einstein"),
    ("What is the largest ocean on Earth?", "Pacific Ocean")
]

# Step 2: Convert into DSPy Examples

dataset = [
    dspy.Example(question=q, answer=a).with_inputs("question")
    for q, a in qa_pairs
]

# Step 3: Train/validation split (20% / 80%)

split_index = math.ceil(len(dataset) * 0.2)  # 20% train
trainset = dataset[:split_index]
valset = dataset[split_index:]

print(f"Total examples: {len(dataset)}")
print(f"Training set size: {len(trainset)}")
print(f"Validation set size: {len(valset)}")

# Step 4: Define a Signature

class QA(dspy.Signature):
    """Answer the given question accurately."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="A short, factual answer.")

# Step 5: Configure a Language Model

lm = dspy.LM('ollama_chat/phi3', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

# Step 6: Create a predictor module

class QAPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(QA)

    def forward(self, question):
        return self.predict(question=question)

# Step 7: Define the evaluation metric

def exact_match_metric(gold, pred, trace=None):
    """
    Returns True if the predicted answer matches the gold answer (case-insensitive).
    """
    return pred.answer.strip().lower() == gold.answer.strip().lower()

# Step 8: Optimize with BootstrapFewShot

pipeline = QAPredictor()
optimizer = dspy.teleprompt.BootstrapFewShot(
    metric=exact_match_metric,
    max_bootstrapped_demos=3
)

optimized_pipeline = optimizer.compile(pipeline, trainset=trainset)

# Step 9: Test on validation set

print("\nValidation results:")
for example in valset:
    prediction = optimized_pipeline(question=example.question)
    print(f"Q: {example.question}")
    print(f"Predicted: {prediction.answer} | Gold: {example.answer}")
    print("-" * 40)
```

## Output
```
Validation results:
Q: Who discovered penicillin?
Predicted: Alexander Fleming | Gold: Alexander Fleming
----------------------------------------
Q: What is the fastest land animal?
Predicted: Cheetah | Gold: Cheetah
----------------------------------------
Q: Which planet is known as the Red Planet?
Predicted: Mars | Gold: Mars
----------------------------------------
Q: What is the square root of 64?
Predicted: 8 | Gold: 8
----------------------------------------
Q: In which year did the Titanic sink?
Predicted: 1912 | Gold: 1912
----------------------------------------
Q: What is the hardest natural substance?
Predicted: Diamond | Gold: Diamond
----------------------------------------
Q: Who developed the theory of relativity?
Predicted: Albert Einstein | Gold: Albert Einstein
----------------------------------------
Q: What is the largest ocean on Earth?
Predicted: Pacific Ocean | Gold: Pacific Ocean
----------------------------------------
```
