---
sidebar_position: 2
---

# LabeledFewShot

### Constructor

The constructor initializes the `LabeledFewShot` class and sets up its attributes, particularly defining `k` number of samples to be used by the predictor.

```python
class LabeledFewShot(Teleprompter):
    def __init__(self, k=16):
        self.k = k
```

**Parameters:**
- `k` (_int_): Number of samples to be used for each predictor. Defaults to 16.

### Method

#### `compile(self, student, *, trainset)`

This method compiles the `LabeledFewShot` instance by configuring the `student` predictor. It assigns subsets of the `trainset` in each student's predictor's `demos` attribute. If the `trainset` is empty, the method returns the original `student`.

**Parameters:**
- `student` (_Teleprompter_): Student predictor to be compiled.
- `trainset` (_list_): Training dataset for compiling with student predictor.

**Returns:**
- The compiled `student` predictor with assigned training samples for each predictor or the original `student` if the `trainset` is empty.

### Example

```python
import dspy

#Assume defined trainset
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        #declare retrieval and predictor modules
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    #flow for answering questions using predictor and retrieval modules
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

#Define teleprompter
teleprompter = LabeledFewShot()

# Compile!
compiled_rag = teleprompter.compile(student=RAG(), trainset=trainset)
```