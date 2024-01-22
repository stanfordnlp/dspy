---
sidebar_position: 1
---

# Signature Optimizer

`SignatureOptimizer` which aims to improve the output prefixes and instruction of the signatures in a module in a zero/few shot setting. This teleprompter is especially beneficial for fine-tuning the prompt for language models and ensure they perform tasks more effectively, all from a vague and un refine prompt.

## Using `SignatureOptimizer`

To demonstrate how `SignatureOptimizer` works, let's optimize the signatures within the following DSPy module:

```python
class CoTSignature(dspy.Signature):
    """Answer the question and give the reasoning for the same."""

    question = dspy.InputField(desc="question about something")
    reasoning = dspy.OutputField(desc="reasoning for the answer")
    answer = dspy.OutputField(desc="often between 1 and 5 words")

class CoTPipeline(dspy.Module):
    def __init__(self):
        super().__init__()

        self.signature = CoTSignature
        self.predictor = dspy.Predict(self.signature)

    def forward(self, question):
        result = self.predictor(question=question)
        return dspy.Prediction(
            answer=result.answer,
            reasoning=result.reasoning,
        )
```

Next, let's dive into the process of using `SignatureOptimizer`:

```python
from dspy.teleprompt import SignatureOptimizer

def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    return answer_EM

teleprompter = SignatureOptimizer(metric=validate_context_and_answer, verbose=True)
kwargs = dict(num_threads=64, display_progress=True, display_table=0)

compiled_prompt_opt = teleprompter.compile(CoTPipeline(), devset=devset, eval_kwargs=kwargs)
```

Once that is we

## How `SignatureOptimizer` works?

It is interesting that to get optimal prefixes and instruction, `SignatureOptimizer` uses Signatures. Basically `SignatureOptimizer` uses Signature to optimize Signature!!

```python
class BasicGenerateInstruction(Signature):
    """You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Your task is to propose an instruction that will lead a good language model to perform the task well. Don't be afraid to be creative."""

    basic_instruction = dspy.InputField(desc="The initial instructions before optimization")
    proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = dspy.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")

class GenerateInstructionGivenAttempts(dspy.Signature):
    """You are an instruction optimizer for large language models. I will give some task instructions I've tried, along with their corresponding validation scores. The instructions are arranged in increasing order based on their scores, where higher scores indicate better quality.

Your task is to propose a new instruction that will lead a good language model to perform the task even better. Don't be afraid to be creative."""

    attempted_instructions = dspy.InputField(format=dsp.passages2text)
    proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = dspy.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")
```

These two signatures are what give use the optimal instruction and prefixes. Now, the `BasicGenerateInstruction` will generate `n` instruction and prefixes based on the `breadth` parameter, basically `n=breadth`. It evaluates these instructions and based on all these pass them to `GenerateInstructionGivenAttempts` which outputs hopefully a more optimal instruction. This happens for `m` iterations which is the `depth` parameter in DSPy.
