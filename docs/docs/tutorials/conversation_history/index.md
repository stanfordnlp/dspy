# Managing Conversation History

Maintaining conversation history is a fundamental feature when building AI applications such as chatbots. While DSPy does not provide automatic conversation history management within `dspy.Module`, it offers the `dspy.History` utility to help you manage conversation history effectively.

## Using `dspy.History` to Manage Conversation History

The `dspy.History` class can be used as an input field type, containing a `messages: list[dict[str, Any]]` attribute that stores the conversation history. Each entry in this list is a dictionary with keys corresponding to the fields defined in your signature. See the example below:

```python
import dspy
import os

os.environ["OPENAI_API_KEY"] = "{your_openai_api_key}"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class QA(dspy.Signature):
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()

predict = dspy.Predict(QA)
history = dspy.History(messages=[])

while True:
    question = input("Type your question, end conversation by typing 'finish': ")
    if question == "finish":
        break
    outputs = predict(question=question, history=history)
    print(f"\n{outputs.answer}\n")
    history.messages.append({"question": question, **outputs})

dspy.inspect_history()
```

There are two key steps when using the conversation history:

- **Include a field of type `dspy.History` in your Signature.**
- **Maintain a history instance at runtime, appending new conversation turns to it.** Each entry should include all relevant input and output field information.

A sample run might look like this:

```
Type your question, end conversation by typing 'finish': do you know the competition between pytorch and tensorflow?

Yes, there is a notable competition between PyTorch and TensorFlow, which are two of the most popular deep learning frameworks. PyTorch, developed by Facebook, is known for its dynamic computation graph, which allows for more flexibility and ease of use, especially in research settings. TensorFlow, developed by Google, initially used a static computation graph but has since introduced eager execution to improve usability. TensorFlow is often favored in production environments due to its scalability and deployment capabilities. Both frameworks have strong communities and extensive libraries, and the choice between them often depends on specific project requirements and personal preference.

Type your question, end conversation by typing 'finish': which one won the battle? just tell me the result, don't include any reasoning, thanks!

There is no definitive winner; both PyTorch and TensorFlow are widely used and have their own strengths.
Type your question, end conversation by typing 'finish': finish




[2025-07-11T16:35:57.592762]

System message:

Your input fields are:
1. `question` (str): 
2. `history` (History):
Your output fields are:
1. `answer` (str):
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}

[[ ## history ## ]]
{history}

[[ ## answer ## ]]
{answer}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Given the fields `question`, `history`, produce the fields `answer`.


User message:

[[ ## question ## ]]
do you know the competition between pytorch and tensorflow?


Assistant message:

[[ ## answer ## ]]
Yes, there is a notable competition between PyTorch and TensorFlow, which are two of the most popular deep learning frameworks. PyTorch, developed by Facebook, is known for its dynamic computation graph, which allows for more flexibility and ease of use, especially in research settings. TensorFlow, developed by Google, initially used a static computation graph but has since introduced eager execution to improve usability. TensorFlow is often favored in production environments due to its scalability and deployment capabilities. Both frameworks have strong communities and extensive libraries, and the choice between them often depends on specific project requirements and personal preference.

[[ ## completed ## ]]


User message:

[[ ## question ## ]]
which one won the battle? just tell me the result, don't include any reasoning, thanks!

Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## answer ## ]]
There is no definitive winner; both PyTorch and TensorFlow are widely used and have their own strengths.

[[ ## completed ## ]]
```

Notice how each user input and assistant response is appended to the history, allowing the model to maintain context across turns.

The actual prompt sent to the language model is a multi-turn message, as shown by the output of `dspy.inspect_history`. Each conversation turn is represented as a user message followed by an assistant message.

## History in Few-shot Examples

You may notice that `history` does not appear in the input fields section of the prompt, even though it is listed as an input field (e.g., "2. `history` (History):" in the system message). This is intentional: when formatting few-shot examples that include conversation history, DSPy does not expand the history into multiple turns. Instead, to remain compatible with the OpenAI standard format, each few-shot example is represented as a single turn.

For example:

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


class QA(dspy.Signature):
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()


predict = dspy.Predict(QA)
history = dspy.History(messages=[])

predict.demos.append(
    dspy.Example(
        question="What is the capital of France?",
        history=dspy.History(
            messages=[{"question": "What is the capital of Germany?", "answer": "The capital of Germany is Berlin."}]
        ),
        answer="The capital of France is Paris.",
    )
)

predict(question="What is the capital of America?", history=dspy.History(messages=[]))
dspy.inspect_history()
```

The resulting history will look like this:

```
[2025-07-11T16:53:10.994111]

System message:

Your input fields are:
1. `question` (str): 
2. `history` (History):
Your output fields are:
1. `answer` (str):
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}

[[ ## history ## ]]
{history}

[[ ## answer ## ]]
{answer}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Given the fields `question`, `history`, produce the fields `answer`.


User message:

[[ ## question ## ]]
What is the capital of France?

[[ ## history ## ]]
{"messages": [{"question": "What is the capital of Germany?", "answer": "The capital of Germany is Berlin."}]}


Assistant message:

[[ ## answer ## ]]
The capital of France is Paris.

[[ ## completed ## ]]


User message:

[[ ## question ## ]]
What is the capital of America?

Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## answer ## ]]
The capital of the United States of America is Washington, D.C.

[[ ## completed ## ]]
```

As you can see, the few-shot example does not expand the conversation history into multiple turns. Instead, it represents the history as JSON data within its section:

```
[[ ## history ## ]]
{"messages": [{"question": "What is the capital of Germany?", "answer": "The capital of Germany is Berlin."}]}
```

This approach ensures compatibility with standard prompt formats while still providing the model with relevant conversational context.

