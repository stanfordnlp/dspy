# Building metrics for evaluation and optimization

## Why optimizers need metrics

Before DSPy can improve our program automatically, we need to tell it what “better” means. That starts with a metric and a baseline evaluation. In this section we’ll see why optimization beats hand-tuning, design a quantifiable metric for our haiku task, and run a baseline score so we have something to beat.

So far we’ve written DSPy programs by hand. Our signatures and instructions are only as good as what we type, with no examples to learn from. The quality of our haikus depends on the base knowledge a given model has about locations and writing haikus.

*Optimizers* close that gap. An optimizer is an algorithm that improves your program automatically, and [DSPy ships several](../diving-deeper/choosing-an-optimizer.md). Some select small input/output pairs (few-shot examples) to include in the prompt as demonstrations. Others rewrite the natural-language instructions in your signatures. A few go further and fine-tune the underlying model. What they share is a loop: run your program many times, keep the version that scores best.

To decide what “better” means, an optimizer needs a *metric*: a Python function that scores a single prediction.

Outside DSPy, evaluation code often spends real effort on structural checks — did the model return valid JSON, are the right fields filled, does the output parse. Signatures already handle structure for us, so our metrics focus on the goal of the task: did the haiku evoke the place, did the answer reach the right conclusion, did the agent do what we asked. Rubrics, judges, and dataset comparisons all aim at that.

There are many ways to define your metric, but most use one or more of these patterns:

- **Labeled data comparisons:** Compare the prediction against a “gold” output, usually labeled by a human expert. For example, a dataset of five-haiku arrays each paired with one ideal pick would let us evaluate and optimize the ensemble judge.  
- **Rule-based checks:** Evaluate with code that checks verifiable properties. For our haiku program, this might involve counting lines and syllables in a haiku.  
- **[LLM judges](../diving-deeper/metrics-and-evaluation.md):** Evaluate with another, usually larger, model. We might ask a frontier model how well a smaller model’s haiku evokes a place and season, similar to our ensemble’s judge. But here the optimizer bakes those judgments into the smaller model’s instructions once, instead of calling the judge with every inference.

You can get creative and clever with these patterns. For example, we could measure our program’s ability to evoke a specific place by generating sets of haikus from varied location inputs, presenting an LLM judge with the array, and asking it to pick the verse matching a given location.

Today, however, we’re going to keep it simple. We’re going to use a *quantifiable metric* that checks a haiku’s syllable count, line count, tense, first-person usage, and whether it echoes our input terms verbatim.

## Preparing examples from the haiku dataset

We’ve created a dataset of example inputs by randomly grouping locations, seasons, and mood strings. [Click here to download the JSONL file](https://gist.github.com/dbreunig/b64412e6103d41889f3a87615008408d), containing 800 rows.

To prepare them for an evaluation or optimization, we need to convert each record into a `dspy.Example` object, like so:

```py
import json

examples = []
with open("haiku_examples.jsonl") as f:
    for line in f:
        row = json.loads(line)
        examples.append(
            dspy.Example(
                location=row["location"],
                season=row["season"],
                mood=row["mood"],
            ).with_inputs("location", "season", "mood")
        ) 

n = len(examples)
train_end = int(n * 0.75)
val_end = int(n * 0.875)
train, val, test = examples[:train_end], examples[train_end:val_end], examples[val_end:]
```

`.with_inputs` is how you tell DSPy which fields of an `Example` should be passed to the program at call time. In this dataset every field is an input, because our metric will score the haiku against its own rules rather than against a labeled answer. A question-answering dataset is the more typical shape: each row carries `(question, answer)`, you’d call `.with\_inputs(“question”)`, and DSPy would pass only the question to the program while holding `answer` back for the metric.

## Building our evaluation metric

A metric function for `dspy.Evaluate` accepts the original `example` and the program’s `prediction`, and returns a single float. By convention, scores fall in between 0.0 and 1.0 (higher is better).

For our evaluation metrics, let’s keep it super simple and only check if our season or mood inputs are being used verbatim in the haiku:

```py
def haiku_score(example, prediction) -> float:
    """
    Penalize verbatim use of the input season string.
    A haiku should evoke the season through imagery, not name it
    directly.
    """
    text = prediction.haiku.lower()
    if example.season.strip().lower() in text:
        return 0.0
    return 1.0
```

Then we call `Evaluate`:

```py
evaluate = dspy.Evaluate(devset=val, metric=haiku_score)
baseline_score = evaluate(haiku_bot)
print(baseline_score)
```

Our `dspy.ReAct` haiku writer, powered by `gpt-5.4-nano`, scores only 30% on this metric. And that makes sense. Our rule isn’t a hard and fast rule of haikus, so our model won’t adhere to it driven only by its weights. And our signatures gave no clues this was our objective.

See [Metrics and evaluation](../diving-deeper/metrics-and-evaluation.md) for composite scoring, LLM judges, and test-set hygiene.

---

**Next:** [GEPA optimization →](gepa-optimization.md)
