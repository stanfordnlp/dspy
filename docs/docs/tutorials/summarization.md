---
sidebar_position: 3
---

# [03] Summarization

Summarization is a fundamental task in natural language processing that involves condensing a longer piece of text into a shorter version while retaining its key information and main ideas. It's a crucial skill for both humans and machines, with applications ranging from creating article abstracts to generating concise reports from lengthy documents.

While summarization is valuable, evaluating the quality of summaries produced by language models presents significant challenges. Unlike tasks with clear right or wrong answers, summarization quality is often subjective and context-dependent. Some key difficulties in evaluating summarized outputs include:

1. Balancing information retention with conciseness
2. Preserving the original text's tone and intent
3. Ensuring factual accuracy without introducing errors
4. Adapting to different types of source material and summary purposes
5. Accounting for varying reader preferences and background knowledge

These challenges make it difficult to create universal metrics for summary quality, often requiring a combination of automated measures and human judgment for comprehensive evaluation.

In this example, we show how you can build a summarization system using DSPy. The technique involves the following:

1. We build a DSPy program for doing the actual summarization task which takes a `passage` as input and gives a `summary` as output.
2. We would ideally want to optimize the summarization program over a metric in order to build an effective program. But since writing a metric function for summarization tasks is subjective and ambiguous in nature, we can use a LM to do this task which in turn means we can write another DSPy program to do the task of scoring a summary based on the variable we care about.
3. Finally, we compose the system by using the Metric program for defining the metric function in the Summarization program.

## Metric program

Define two signatures for scoring the summary.

1. The first signature breaks down the summary and assigns defined labels as grades.
```
import dspy

class Breakdown(dspy.Signature):
    """
    Given a passage, break down the passage into key ideas.
    Enumerate every key idea in the passage and
    assign it an importance grade
    (High, Medium, or Low).
    """

    passage = dspy.InputField()
    key_ideas: str = dspy.OutputField(
        desc="numbered list of one key idea per line,"
        "followed by its importance grade, "
             "e.g. 1. <Idea here>. High.")
    importance_grades: list[str] = dspy.OutputField(
        desc='list of importance grades, '
             'e.g. ["High", "Medium", "Low"].')
```

2. The second signature assigns a binary score using the summary and the break down from the previous signature as inputs
```
class SummaryCorrectness(dspy.Signature):
    """
    Compare a system generated summary to the key ideas in the passage.
    For every key idea supplied,
    assign a binary score based on whether the summary contains it.
    And compute an overall score based on the binary scores.
    """

    key_ideas: str = dspy.InputField(
        desc="key ideas in the passage "
             "for evaluating the summary")
    summary: str = dspy.InputField()
    binary_scores: list[bool] = dspy.OutputField(
        desc="list of binary scores for each key idea, "
             "e.g. [True, False, True]")
    overall_score: float = dspy.OutputField(
        desc="overall score for the summary out of 1.0")
```

3. Now write the actual program that computes a weighted score between 0 to 1 and falls back to the overall_score that's loosely scored by the model

```
class Metric(dspy.Module):
    """
    Compute a score for the correctness of a summary.
    """

    def __init__(self):
        self.breakdown = dspy.ChainOfThought(Breakdown)
        self.assess = dspy.ChainOfThought(SummaryCorrectness)

    def forward(self, example, pred, trace=None):
        breakdown = self.breakdown(
            passage=example.passage
        )
        key_ideas = breakdown.key_ideas
        importance_grades = breakdown.importance_grades

        scores = self.assess(
            key_ideas=key_ideas,
            summary=pred.summary,
        )

        try:
            weight_map = {'High': 1.0, 'Medium': 0.7}
            score = sum(
                weight_map.get(g, 0.2) * int(b)
                for g, b in zip(importance_grades, scores.binary_scores)
            )
            score /= sum(weight_map.get(g, 0.2) for g in importance_grades)

        # pylint: disable=broad-except
        except Exception:
            score = float(scores.overall_score)

        return score if trace is None else score >= 0.75
```

## Summarization Program

First define it's signature

```
class SummarizeSignature(dspy.Signature):
    """
    Given a passage, generate a summary.
    """

    passage = dspy.InputField(desc="a passage to summarize")
    summary: str = dspy.OutputField(
        desc="a concise summary of the passage")
```

Next, define the program

```
class Summarize(dspy.Module):
    def __init__(self):
        self.summarize = dspy.ChainOfThought(SummarizeSignature)

    def forward(self, passage: str):
        summary = self.summarize(
            passage=passage
        )
        return summary
```

Let's jump into the evaluation process now. First, define the language model.

```
lm = dspy.LM("openai/gpt-4o-mini", max_tokens=250)
dspy.settings.configure(lm=lm)
```

Load a dataset. The dataset is a `jsonl` file where each line has a passage, summary and a score.

```
dataset = []
with open('src/summarization/programs/summarize/dataset.jsonl', 'r',
          encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)

        passage = data.get("passage", "")
        summary = data.get("summary", "")
        score = data.get("score", 0)

        example = dspy.Example(passage=passage, summary=summary, score=score)
        dataset.append(example)

trainset = [x.with_inputs("passage") for x in dataset]
```

Now, define the metric. This is where we use the `Metric` program to compute the score for the summary generated by the `Summarization` program.

```
def metric(gold, pred, trace=None):
    metric_program = Metric()
    examp = dspy.Example(passage=gold.passage)
    predicted = dspy.Example(summary=pred)
    pred_score = metric_program(example=examp, pred=predicted)
    gold_score = gold.score
    # check if they are almost equal
    return abs(float(gold_score) - float(pred_score)) < 0.2
```

Create the program and evaluate it.
```
program = Summarize()

evaluate = dspy.Evaluate(devset=trainset, metric=metric,
                    display_progress=True,
                    display_table=True, provide_traceback=True)

res = evaluate(program, devset=trainset)
print(res)
```

That's it! Hopefully this example was helpful in understanding how you can compose DSPy programs to build a truly compound AI system that's optimized for isolated tasks.