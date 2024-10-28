---
sidebar_position: 2
---

# ReAct

## Background

DSPy supports ReAct, an LLM agent designed to tackle complex tasks in an interactive fashion. ReAct is composed of an iterative loop of interpretation, decision and action-based activities ("Thought, Action, and Observation") based on an evolving set of input and output fields. Through this real-time iterative approach, the ReAct agent can both analyze and adapt to its responses over time as new information becomes available.

## Instantiating ReAct 

To instantiate the ReAct module, define and pass in a DSPy Signature. 

```python
# Define a simple signature for basic question answering
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# Pass signature to ReAct module
react_module = dspy.ReAct(BasicQA)
```

## Under the Hood

### ReAct Cycle

ReAct operates under a dynamic signature integration process, accounting for the signature inputs and performing the Thoughts, Action, Observation cycle to respond with the signature outputs. Thoughts (or reasoning) lead to Actions (such as queries or activities). These Actions then result in Observations (like results or responses), which subsequently feedback into the next Thought.

This cycle is maintained for a predefined number of iterations, specified by `max_iters`. The default value for the Thought-Action-Observation cycle is 5 iterations. Once the maximum iterations are reached, React will return the final output if the Action has finished `(Finish[answer])` or an empty string to indicate the agent could not determine a final output.

!!! caution
    Currently, ReAct supports only one output field in its signature. We plan to expand this in future developments.

### ReAct Tools

Tools in ReAct can shape the agent's interaction and response mechanisms, and DSPy ensures this customizability by allowing users to pass in their toolsets tailored for their task scenarios. The default tool is the `dspy.Retrieve`
module (serving to retrieve information from Retrieval Models during the Action step) with default `num_results=3`, and these can be passed as arguments to the initialization of the ReAct module.



## Tying It All Together
Using ReAct mirrors the simplicity of the base `Predict` and `ChainOfThought` modules. Here is an example call:

```python
# Call the ReAct module on a particular input
question = 'Aside from the Apple Remote, what other devices can control the program Apple Remote was originally designed to interact with?'
result = react_module(question=question)

print(f"Question: {question}")
print(f"Final Predicted Answer (after ReAct process): {result.answer}")
```
```
Question: Aside from the Apple Remote, what other devices can control the program Apple Remote was originally designed to interact with?
Final Predicted Answer (after ReAct process): The Apple Remote and the Siri Remote can control the Front Row media program.
```

Let's take a peek at how ReAct functioned internally by inspecting its history, up to maximum iterations. (This assumes the initial DSPy setup and configurations of LMs and RMs). 

`lm.inspect_history(n=3)`

```
-------------------------Step 1---------------------------------
You will be given `question` and you will respond with `answer`.

To do this, you will interleave Thought, Action, and Observation steps.

Thought can reason about the current situation, and Action can be the following types:

(1) Search[query], which takes a search query and returns one or more potentially relevant passages from a corpus
(2) Finish[answer], which returns the final `answer` and finishes the task

---

Follow the following format.

Question: ${question}
Thought 1: next steps to take based on last observation
Action 1: always either Search[query] or, when done, Finish[answer]

---

Question: Aside from the Apple Remote, what other devices can control the program Apple Remote was originally designed to interact with?
Thought 1: I need to find out what program the Apple Remote was originally designed to interact with.
Action 1: Search["program Apple Remote was originally designed to interact with"]

-------------------------Step 2---------------------------------
You will be given `question` and you will respond with `answer`.

To do this, you will interleave Thought, Action, and Observation steps.

Thought can reason about the current situation, and Action can be the following types:

(1) Search[query], which takes a search query and returns one or more potentially relevant passages from a corpus
(2) Finish[answer], which returns the final `answer` and finishes the task

---

Follow the following format.

Question: ${question}

Thought 1: next steps to take based on last observation

Action 1: always either Search[query] or, when done, Finish[answer]

Observation 1: observations based on action

Thought 2: next steps to take based on last observation

Action 2: always either Search[query] or, when done, Finish[answer]

---

Question: Aside from the Apple Remote, what other devices can control the program Apple Remote was originally designed to interact with?

[Thought 1, Action 1]

Observation 1:
[1] «Apple Remote | The Apple Remote is a remote control device released [...].»
[2] «ITunes Remote | iTunes Remote (also known simply as Remote) is a software [...].»
[3] «Siri Remote | The Siri Remote is a remote control device released in tandem [...]»

Thought 2: I have found that the Apple Remote was originally designed to interact with the Front Row media program on the iSight iMac G5. I should search for other devices that can control this program.

Action 2: Search["devices that can control Front Row media program"]

Observation 2:
[1] «Apple Remote» - The Apple Remote is the primary control mechanism for the Front Row media program [...].
[2] «Siri Remote» - The Siri Remote is the successor of the Apple Remote and is designed [...].

-------------------------Step 3---------------------------------

You will be given `question` and you will respond with `answer`.

To do this, you will interleave Thought, Action, and Observation steps.

Thought can reason about the current situation, and Action can be the following types:

(1) Search[query], which takes a search query and returns one or more potentially relevant passages from a corpus
(2) Finish[answer], which returns the final `answer` and finishes the task

---

Follow the following format.

Question: ${question}

Thought 1: next steps to take based on last observation

Action 1: always either Search[query] or, when done, Finish[answer]

Observation 1: observations based on action

Thought 2: next steps to take based on last observation

Action 2: always either Search[query] or, when done, Finish[answer]

Observation 2: observations based on action

Thought 3: next steps to take based on last observation

Action 3: always either Search[query] or, when done, Finish[answer]

---

Question: Aside from the Apple Remote, what other devices can control the program Apple Remote was originally designed to interact with?

[Thought 1, Action 1, Observation 1, Thought 2, Action 2]

Observation 2:
[1] «Front Row (software) | Front Row is a discontinued media center [...].»
[2] «Apple Remote | The Apple Remote is a remote control device [...].»
[3] «Media player (software) | A media player is a computer program for [...].»

Thought 3: I have found that the Apple Remote and the Siri Remote can control the Front Row media program. I should provide this information as the answer.

Action 3: Finish[The Apple Remote and the Siri Remote can control the Front Row media program.]

Answer: The Apple Remote and the Siri Remote can control the Front Row media program.
```
