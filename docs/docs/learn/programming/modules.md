---
sidebar_position: 3
---

# Modules

A **DSPy module** is a building block for programs that use LMs.

- Each built-in module abstracts a **prompting technique** (like chain of thought or ReAct). Crucially, they are generalized to handle any signature.

- A DSPy module has **learnable parameters** (i.e., the little pieces comprising the prompt and the LM weights) and can be invoked (called) to process inputs and return outputs.

- Multiple modules can be composed into bigger modules (programs). DSPy modules are inspired directly by NN modules in PyTorch, but applied to LM programs.


## How do I use a built-in module, like `dspy.Predict` or `dspy.ChainOfThought`?

Let's start with the most fundamental module, `dspy.Predict`. Internally, all other DSPy modules are built using `dspy.Predict`. We'll assume you are already at least a little familiar with [DSPy signatures](signatures.md), which are declarative specs for defining the behavior of any module we use in DSPy.

To use a module, we first **declare** it by giving it a signature. Then we **call** the module with the input arguments, and extract the output fields!

```python
sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.

# 1) Declare with a signature.
classify = dspy.Predict('sentence -> sentiment: bool')

# 2) Call with input argument(s). 
response = classify(sentence=sentence)

# 3) Access the output.
print(response.sentiment)
```
**Output:**
```text
True
```

When we declare a module, we can pass configuration keys to it.

Below, we'll pass `n=5` to request five completions. We can also pass `temperature` or `max_len`, etc.

Let's use `dspy.ChainOfThought`. In many cases, simply swapping `dspy.ChainOfThought` in place of `dspy.Predict` improves quality.

```python
question = "What's something great about the ColBERT retrieval model?"

# 1) Declare with a signature, and pass some config.
classify = dspy.ChainOfThought('question -> answer', n=5)

# 2) Call with input argument.
response = classify(question=question)

# 3) Access the outputs.
response.completions.answer
```
**Possible Output:**
```text
['One great thing about the ColBERT retrieval model is its superior efficiency and effectiveness compared to other models.',
 'Its ability to efficiently retrieve relevant information from large document collections.',
 'One great thing about the ColBERT retrieval model is its superior performance compared to other models and its efficient use of pre-trained language models.',
 'One great thing about the ColBERT retrieval model is its superior efficiency and accuracy compared to other models.',
 'One great thing about the ColBERT retrieval model is its ability to incorporate user feedback and support complex queries.']
```

Let's discuss the output object here. The `dspy.ChainOfThought` module will generally inject a `reasoning` before the output field(s) of your signature.

Let's inspect the (first) reasoning and answer!

```python
print(f"Reasoning: {response.reasoning}")
print(f"Answer: {response.answer}")
```
**Possible Output:**
```text
Reasoning: We can consider the fact that ColBERT has shown to outperform other state-of-the-art retrieval models in terms of efficiency and effectiveness. It uses contextualized embeddings and performs document retrieval in a way that is both accurate and scalable.
Answer: One great thing about the ColBERT retrieval model is its superior efficiency and effectiveness compared to other models.
```

This is accessible whether we request one or many completions.

We can also access the different completions as a list of `Prediction`s or as several lists, one for each field.

```python
response.completions[3].reasoning == response.completions.reasoning[3]
```
**Output:**
```text
True
```


## What other DSPy modules are there? How can I use them?

The others are very similar. They mainly change the internal behavior with which your signature is implemented!

1. **`dspy.Predict`**: Basic predictor. Does not modify the signature. Handles the key forms of learning (i.e., storing the instructions and demonstrations and updates to the LM).

2. **`dspy.ChainOfThought`**: Teaches the LM to think step-by-step before committing to the signature's response.

3. **`dspy.ProgramOfThought`**: Teaches the LM to output code, whose execution results will dictate the response.

4. **`dspy.ReAct`**: An agent that can use tools to implement the given signature.

5. **`dspy.MultiChainComparison`**: Can compare multiple outputs from `ChainOfThought` to produce a final prediction.


We also have some function-style modules:

6. **`dspy.majority`**: Can do basic voting to return the most popular response from a set of predictions.


!!! info "A few examples of DSPy modules on simple tasks."
    Try the examples below after configuring your `lm`. Adjust the fields to explore what tasks your LM can do well out of the box.

    === "Math"

        ```python linenums="1"
        math = dspy.ChainOfThought("question -> answer: float")
        math(question="Two dice are tossed. What is the probability that the sum equals two?")
        ```
        
        **Possible Output:**
        ```text
        Prediction(
            reasoning='When two dice are tossed, each die has 6 faces, resulting in a total of 6 x 6 = 36 possible outcomes. The sum of the numbers on the two dice equals two only when both dice show a 1. This is just one specific outcome: (1, 1). Therefore, there is only 1 favorable outcome. The probability of the sum being two is the number of favorable outcomes divided by the total number of possible outcomes, which is 1/36.',
            answer=0.0277776
        )
        ```

    === "Retrieval-Augmented Generation"

        ```python linenums="1"       
        def search(query: str) -> list[str]:
            """Retrieves abstracts from Wikipedia."""
            results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
            return [x['text'] for x in results]
        
        rag = dspy.ChainOfThought('context, question -> response')

        question = "What's the name of the castle that David Gregory inherited?"
        rag(context=search(question), question=question)
        ```
        
        **Possible Output:**
        ```text
        Prediction(
            reasoning='The context provides information about David Gregory, a Scottish physician and inventor. It specifically mentions that he inherited Kinnairdy Castle in 1664. This detail directly answers the question about the name of the castle that David Gregory inherited.',
            response='Kinnairdy Castle'
        )
        ```

    === "Classification"

        ```python linenums="1"
        from typing import Literal

        class Classify(dspy.Signature):
            """Classify sentiment of a given sentence."""
            
            sentence: str = dspy.InputField()
            sentiment: Literal['positive', 'negative', 'neutral'] = dspy.OutputField()
            confidence: float = dspy.OutputField()

        classify = dspy.Predict(Classify)
        classify(sentence="This book was super fun to read, though not the last chapter.")
        ```
        
        **Possible Output:**

        ```text
        Prediction(
            sentiment='positive',
            confidence=0.75
        )
        ```

    === "Information Extraction"

        ```python linenums="1"        
        text = "Apple Inc. announced its latest iPhone 14 today. The CEO, Tim Cook, highlighted its new features in a press release."

        module = dspy.Predict("text -> title, headings: list[str], entities_and_metadata: list[dict[str, str]]")
        response = module(text=text)

        print(response.title)
        print(response.headings)
        print(response.entities_and_metadata)
        ```
        
        **Possible Output:**
        ```text
        Apple Unveils iPhone 14
        ['Introduction', 'Key Features', "CEO's Statement"]
        [{'entity': 'Apple Inc.', 'type': 'Organization'}, {'entity': 'iPhone 14', 'type': 'Product'}, {'entity': 'Tim Cook', 'type': 'Person'}]
        ```

    === "Agents"

        ```python linenums="1"       
        def evaluate_math(expression: str) -> float:
            return dspy.PythonInterpreter({}).execute(expression)

        def search_wikipedia(query: str) -> str:
            results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
            return [x['text'] for x in results]

        react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])

        pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
        print(pred.answer)
        ```
        
        **Possible Output:**

        ```text
        5761.328
        ```


## How do I compose multiple modules into a bigger program?

DSPy is just Python code that uses modules in any control flow you like, with a little magic internally at `compile` time to trace your LM calls. What this means is that, you can just call the modules freely.

See tutorials like [multi-hop search](https://dspy.ai/tutorials/multihop_search/), whose module is reproduced below as an example.

```python linenums="1"        
class Hop(dspy.Module):
    def __init__(self, num_docs=10, num_hops=4):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought('claim, notes -> query')
        self.append_notes = dspy.ChainOfThought('claim, notes, context -> new_notes: list[str], titles: list[str]')

    def forward(self, claim: str) -> list[str]:
        notes = []
        titles = []

        for _ in range(self.num_hops):
            query = self.generate_query(claim=claim, notes=notes).query
            context = search(query, k=self.num_docs)
            prediction = self.append_notes(claim=claim, notes=notes, context=context)
            notes.extend(prediction.new_notes)
            titles.extend(prediction.titles)
        
        return dspy.Prediction(notes=notes, titles=list(set(titles)))
```

Then you can create a instance of the custom module class `Hop`, then invoke it by the `__call__` method:

```
hop = Hop()
print(hop(claim="Stephen Curry is the best 3 pointer shooter ever in the human history"))
```

## How do I track LM usage?

!!! note "Version Requirement"
    LM usage tracking is available in DSPy version 2.6.16 and later.

DSPy provides built-in tracking of language model usage across all module calls. To enable tracking:

```python
dspy.configure(track_usage=True)
```

Once enabled, you can access usage statistics from any `dspy.Prediction` object:

```python
usage = prediction_instance.get_lm_usage()
```

The usage data is returned as a dictionary that maps each language model name to its usage statistics. Here's a complete example:

```python
import dspy

# Configure DSPy with tracking enabled
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=False),
    track_usage=True
)

# Define a simple program that makes multiple LM calls
class MyProgram(dspy.Module):
    def __init__(self):
        self.predict1 = dspy.ChainOfThought("question -> answer")
        self.predict2 = dspy.ChainOfThought("question, answer -> score")

    def __call__(self, question: str) -> str:
        answer = self.predict1(question=question)
        score = self.predict2(question=question, answer=answer)
        return score

# Run the program and check usage
program = MyProgram()
output = program(question="What is the capital of France?")
print(output.get_lm_usage())
```

This will output usage statistics like:

```python
{
    'openai/gpt-4o-mini': {
        'completion_tokens': 61,
        'prompt_tokens': 260,
        'total_tokens': 321,
        'completion_tokens_details': {
            'accepted_prediction_tokens': 0,
            'audio_tokens': 0,
            'reasoning_tokens': 0,
            'rejected_prediction_tokens': 0,
            'text_tokens': None
        },
        'prompt_tokens_details': {
            'audio_tokens': 0,
            'cached_tokens': 0,
            'text_tokens': None,
            'image_tokens': None
        }
    }
}
```

When using DSPy's caching features (either in-memory or on-disk via litellm), cached responses won't count toward usage statistics. For example:

```python
# Enable caching
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=True),
    track_usage=True
)

program = MyProgram()

# First call - will show usage statistics
output = program(question="What is the capital of Zambia?")
print(output.get_lm_usage())  # Shows token usage

# Second call - same question, will use cache
output = program(question="What is the capital of Zambia?")
print(output.get_lm_usage())  # Shows empty dict: {}
```
