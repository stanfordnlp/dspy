---
sidebar_position: 1
---

# Understanding Signatures

A Signature is the most basic form of task description, all it needs is the inputs and outputs to the signature, optionally a small description about them and the task too.

There are 2 ways to define a Signature: **Inline** and **Class-Based**. But before diving into the creation of signature, let's understand what is a signature and why we need it.

## What is a Signature?

In a usual LLM pipeline you have 2 key components at work i.e. an LLM and a prompt. In DSPy, we have an LLM at work via the LM(Language Model) that we configure at the start of any DSPy script, we'll see how to do this in the next blog, and we have a prompt that work that we define via **Signatures**.

A Signature is usually composed of 2 essential components: **Input Fields** and **Output Fields**. Optionally you can pass an instruction defining your task too but it's not necessary. An **Input Field** is a attribute of Signature that defines an input to the prompt and an **Output Field** is a attribute of Signature that defines an output of the prompt received from an LLM call. Let's understand this by an example.

![DSPy Signatures](./img/dspy_signatures.png)

Let's think of a basic Question-Answer task, you ask a question to the LLM and you get get the answer to it. In this the question is the input to the LLM and hence it'll be the **Input Field** in the Signature, and the answer to the question is the output you get from LLM and hence it'll be the **Output Field** in the Signature.

Now that we a grasp on the components of a Signature, let's see how we can declare a signature and what a prompt for that signature looks like.

## Inline Method

If I have to tell you one of the most intuitive and simple way to define any task is to just tell the inputs and outputs for the task. This way you can convey the gist of the task in the most simple form, for example the for the above task if you told me the input is **question** and output is **answer** I'd be able to understand the gist of the task being a Question-Answer task. If you say inputs are **context** and **question**, and outputs are **answer** and **reason** I'll be able to get the idea that the task could be a RAG pipeline with Chain-Of-Thought prompting.

Inspired by this idea, DSPy allows you to define you task as DSPy Signatures in an Einops like abstract manner, like:

```text
input_field_1,input_field_2,input_field_3...->output_field_1,output_field_2,output_field_3...
```

Then names on the right side of the `->` would be the **Input Fields** of the Signature and Then names on the right side of the `->` would be the **Output Fields** of the Signature. So let's go ahead with the QA and RAG task we talked about in the above section and see how there signature would look like:

```text
QA Task: question->answer
RAG Task: context,question->answer,rationale
```

The naming of the fields is import for the LLM to understand the nature of inputs and outputs. But how does the prompt for the signature `question->answer` look like? Note that we didn't pass any instruction yet so DSPy would prepare that based on the fields. Let's take a deeper look at the prompt constructed by DSPy to understand it better:

```
Given the fields `question`, produce the fields `answer`.

---

Follow the following format.

Question: ${question}
Answer: ${answer}

---

Question:
```

As you can see based on the fields DSPy uses the instruction `Given the fields ``question``, produce the fields ``answer``.` to define the task. It provides instruction for the format and inputs/outputs, all based on the fields you define. And this format is pretty standard for any Signature you create, let's see how it happens for RAG:

![Prompt Creation for Inline](./img/prompt_creation.png)

The prompt that you create is gonna be dependant on the fields and their that you define in inline signature format. But wouldn't it be nice to have more control ove the prompt and fields? Luckily class-based signatures help us with that!!

## Class Based Method

A Signature class comprises of three things:

* **Task Description/Instruction:** We define in the signature class docstring.
* **Inputs Field:** We define these as `dspy.InputField()`.
* **Outputs Field:** We define these as `dspy.OutputField()`.

```python
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words", prefix="Question's Answer:")
```

The I/O Fields take 3 inputs: ``desc`, `prefix` and `format`. `desc` is the description to the input, `prefix` is the placeholder text of the field in the prompt(one that has been ${field_name} until now) and `format` which is a method that'll define how to handle non-string inputs. If the input to field is a list rather than a string we'll define how to format the content of that list to a string.

Not so surprisingly both the fields are similar in implimentation as well:

```python
class InputField(Field):
    def __init__(self, *, prefix=None, desc=None, format=None):
        super().__init__(prefix=prefix, desc=desc, input=True, format=format)

class OutputField(Field):
    def __init__(self, *, prefix=None, desc=None, format=None):
        super().__init__(prefix=prefix, desc=desc, input=False, format=format)
```

But how does the prompt for the class based signature look like, let's see:

```text
Answer questions with short factoid answers.

---

Follow the following format.

Question: ${question}
Question's Answer: often between 1 and 5 words

---

Question:
```

As you can see the prefix for `answer` field has changed and been replaced with the `prefix` defined for it and the description is now the `desc` we defined for it. Where as in `question` field both `prefix` and `desc` were not defined so it remains same as the inline one. The instruction too is same as the docstring we pass to the class. This tells us that the essential prompt structure is the same for signature regardless of the way you define it, the only thing taht changes is the control we have to modify the content of that prompt.

![Class Based Prompt Creation](./img/class_based_prompt_creation.png)