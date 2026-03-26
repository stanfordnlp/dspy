---
sidebar_position: 2
---

# Signatures

When we assign tasks to LMs in DSPy, we specify the behavior we need as a Signature.

**A signature is a declarative specification of input/output behavior of a DSPy module.** Signatures allow you to tell the LM _what_ it needs to do, rather than specify _how_ we should ask the LM to do it.

You're probably familiar with function signatures, which specify the input and output arguments and their types. DSPy signatures are similar, but with a couple of differences. While typical function signatures just _describe_ things, DSPy Signatures _declare and initialize the behavior_ of modules. Moreover, the field names matter in DSPy Signatures. You express semantic roles in plain English: a `question` is different from an `answer`, a `sql_query` is different from `python_code`.

## Why should I use a DSPy Signature?

For modular and clean code, in which LM calls can be optimized into high-quality prompts (or automatic finetunes). Most people coerce LMs to do tasks by hacking long, brittle prompts. Or by collecting/generating data for fine-tuning. Writing signatures is far more modular, adaptive, and reproducible than hacking at prompts or finetunes. The DSPy compiler will figure out how to build a highly-optimized prompt for your LM (or finetune your small LM) for your signature, on your data, and within your pipeline. In many cases, we found that compiling leads to better prompts than humans write. Not because DSPy optimizers are more creative than humans, but simply because they can try more things and tune the metrics directly.

## **Inline** DSPy Signatures

Signatures can be defined as a short string, with argument names and optional types that define semantic roles for inputs/outputs.

1. Question Answering: `"question -> answer"`, which is equivalent to `"question: str -> answer: str"` as the default type is always `str`

2. Sentiment Classification: `"sentence -> sentiment: bool"`, e.g. `True` if positive

3. Summarization: `"document -> summary"`

Your signatures can also have multiple input/output fields with types:

4. Retrieval-Augmented Question Answering: `"context: list[str], question: str -> answer: str"`

5. Multiple-Choice Question Answering with Reasoning: `"question, choices: list[str] -> reasoning: str, selection: int"`

**Tip:** For fields, any valid variable names work! Field names should be semantically meaningful, but start simple and don't prematurely optimize keywords! Leave that kind of hacking to the DSPy compiler. For example, for summarization, it's probably fine to say `"document -> summary"`, `"text -> gist"`, or `"long_context -> tldr"`.

You can also add instructions to your inline signature, which can use variables at runtime. Use the `instructions` keyword argument to add instructions to your signature.

```python
toxicity = dspy.Predict(
    dspy.Signature(
        "comment -> toxic: bool",
        instructions="Mark as 'toxic' if the comment includes insults, harassment, or sarcastic derogatory remarks.",
    )
)
comment = "you are beautiful."
toxicity(comment=comment).toxic
```

**Output:**
```text
False
```


### Example A: Sentiment Classification

```python
sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.

classify = dspy.Predict('sentence -> sentiment: bool')  # we'll see an example with Literal[] later
classify(sentence=sentence).sentiment
```
**Output:**
```text
True
```

### Example B: Summarization

```python
# Example from the XSum dataset.
document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""

summarize = dspy.ChainOfThought('document -> summary')
response = summarize(document=document)

print(response.summary)
```
**Possible Output:**
```text
The 21-year-old Lee made seven appearances and scored one goal for West Ham last season. He had loan spells in League One with Blackpool and Colchester United, scoring twice for the latter. He has now signed a contract with Barnsley, but the length of the contract has not been revealed.
```

Many DSPy modules (except `dspy.Predict`) return auxiliary information by expanding your signature under the hood.

For example, `dspy.ChainOfThought` also adds a `reasoning` field that includes the LM's reasoning before it generates the output `summary`.

```python
print("Reasoning:", response.reasoning)
```
**Possible Output:**
```text
Reasoning: We need to highlight Lee's performance for West Ham, his loan spells in League One, and his new contract with Barnsley. We also need to mention that his contract length has not been disclosed.
```

## **Typed** DSPy Signatures

For some advanced tasks, you may need more verbose signatures with type annotations. DSPy supports a **typed signature API** that uses separate input and output schemas. It's useful when you want to:

1. Get autocompletions and type hints from your IDE: because the return value is a typed instance of your output class, Pylance, mypy, and PyCharm know which fields are available and what types they have.
3. Supply hints on the nature of an input field or constraints of the output field, expressed as a `desc` keyword argument using `dspy.Field(desc="...")`.
4. Clarify something about the nature of the task using `instructions`.
5. Clearly separate input and output type definitions for reuse across multiple modules.

With inline strings or class-based signatures, the return type is `Prediction`, a dynamic object that no static tool can inspect. With typed signatures, both the input *and* the return type are concrete Python classes, so your editor can auto-complete fields, flag typos, and infer types through your whole program.

### Example C: Classification

```python
import pydantic
from typing import Literal

class EmotionInput(pydantic.BaseModel):
    sentence: str

class EmotionOutput:
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

Emotion = dspy.Signature(
    input_type=EmotionInput,
    output_type=EmotionOutput,
    instructions="Classify emotion.",
)

sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"  # from dair-ai/emotion

classify = dspy.Predict(Emotion)
result = classify(EmotionInput(sentence=sentence))

assert isinstance(result, EmotionOutput)
print(result.sentiment)
```
**Possible Output:**
```text
fear
```

You can also use a `dataclass` in place of `pydantic.BaseModel` -- both work identically:

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class EmotionInput:
    sentence: str

class EmotionOutput:
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

Emotion = dspy.Signature(input_type=EmotionInput, output_type=EmotionOutput,
                          instructions="Classify emotion.")
```

Alternatively, you can express the same signature as a single class. However, this variant does not support type annotations or autocompletion. The module's return value is of type `Prediction`:

```python
from typing import Literal

class Emotion(dspy.Signature):
    """Classify emotion."""
    sentence: str = dspy.InputField()
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField()

classify = dspy.Predict(Emotion)
result = classify(sentence=sentence)

assert isinstance(result, Prediction)
```

**Tip:** There's nothing wrong with specifying your requests to the LM more clearly. Class-based Signatures help you with that. However, don't prematurely tune the keywords of your signature by hand. The DSPy optimizers will likely do a better job (and will transfer better across LMs).

### Example D: A metric that evaluates faithfulness to citations

```python
class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""

    context: str = dspy.InputField(desc="facts here are assumed to be true")
    text: str = dspy.InputField()
    faithfulness: bool = dspy.OutputField()
    evidence: dict[str, list[str]] = dspy.OutputField(desc="Supporting evidence for claims")

context = "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."

text = "Lee scored 3 goals for Colchester United."

faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
faithfulness(context=context, text=text)
```
**Possible Output:**
```text
Prediction(
    reasoning="Let's check the claims against the context. The text states Lee scored 3 goals for Colchester United, but the context clearly states 'He scored twice for the U's'. This is a direct contradiction.",
    faithfulness=False,
    evidence={'goal_count': ["scored twice for the U's"]}
)
```

### Example E: Multi-modal image classification

```python
import pydantic

class DogInput(pydantic.BaseModel):
    image_1: dspy.Image = dspy.Field(desc="An image of a dog")

class DogOutput:
    answer: str = dspy.Field(desc="The dog breed of the dog in the image")

DogPictureSignature = dspy.Signature(
    input_type=DogInput,
    output_type=DogOutput,
    instructions="Output the dog breed of the dog in the image.",
)

image_url = "https://picsum.photos/id/237/200/300"
classify = dspy.Predict(DogPictureSignature)
result = classify(DogInput(image_1=dspy.Image.from_url(image_url)))
print(result.answer)
```

**Possible Output:**

```text
Labrador Retriever
```

## Type Resolution in Signatures

DSPy signatures support various annotation types:

1. **Basic types** like `str`, `int`, `bool`
2. **Typing module types** like `list[str]`, `dict[str, int]`, `Optional[float]`. `Union[str, int]`
3. **Custom types** defined in your code
4. **Dot notation** for nested types with proper configuration
5. **Special data types** like `dspy.Image, dspy.History`

### Working with Custom Types

```python
# Simple custom type
class QueryResult(pydantic.BaseModel):
    text: str
    score: float

signature = dspy.Signature("query: str -> result: QueryResult")

class MyContainer:
    class Query(pydantic.BaseModel):
        text: str
    class Score(pydantic.BaseModel):
        score: float

signature = dspy.Signature("query: MyContainer.Query -> score: MyContainer.Score")
```

### Type Checking for Input Fields

DSPy automatically validates that the values you pass to input fields match the types specified in your signature. This works for inline, typed, and single-class-based signatures alike.
When there's a type mismatch, DSPy will log a warning.

**Example: Type Mismatch Warning**

```python
from dataclasses import dataclass

@dataclass
class MathInput:
    number: int

class MathOutput:
    result: str

MathSignature = dspy.Signature(
    input_type=MathInput,
    output_type=MathOutput,
    instructions="Perform a mathematical operation.",
)

predictor = dspy.Predict(MathSignature)

# This will trigger a warning because we're passing a string instead of an int
predictor(MathInput(number="42"))  # Warning: Type mismatch for field 'number': expected int, but provided value is incompatible
```

The same type checking applies to the single-class-based style:

```python
class MathSignature(dspy.Signature):
    """Perform a mathematical operation."""
    number: int = dspy.InputField()
    result: str = dspy.OutputField()
```
**Disabling Type Checking**

If you want to disable type mismatch warnings, you can turn off this feature:

```python
# Disable type mismatch warnings globally
dspy.configure(warn_on_type_mismatch=False)

predictor = dspy.Predict("number: int -> result: str")
predictor(number="42")  # No warning
```

## Using signatures to build modules & compiling them

While signatures are convenient for prototyping with structured inputs/outputs, that's not the only reason to use them!

You should compose multiple signatures into bigger [DSPy modules](modules.md) and [compile these modules into optimized prompts](../optimization/optimizers.md) and finetunes.
