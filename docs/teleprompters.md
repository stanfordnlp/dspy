# dspy.Teleprompters Documentation

This documentation provides an overview of the DSPy Teleprompters.

## dspy.teleprompt.LabeledFewShot

### Constructor

The constructor initializes the `LabeledFewShot` class and sets up its attributes, particularly defining `k` number of samples to be used by the predictor.

```python
class LabeledFewShot(Teleprompter):
    def __init__(self, k=16):
        self.k = k
```

**Arguments:**
- `k` (int): Number of samples to be used for each predictor. Defaults to 16.

### Method

#### `compile(self, student, *, trainset)`

This method compiles the `LabeledFewShot` instance by configuring the `student` predictor. It assigns subsets of the `trainset` in each student's predictor's `demos` attribute. If the `trainset` is empty, the method returns the original `student`.

**Arguments:**
- `student` (Teleprompter): Student predictor to be compiled.
- `trainset` (list): Training dataset for compiling with student predictor.

**Returns:**
- The compiled `student` predictor with assigned training samples for each predictor or the original `student` if the `trainset` is empty.

### Examples

```python
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

## dspy.teleprompt.BootstrapFewShot

### Constructor

The constructor initializes the `BootstrapFewShot` class and sets up parameters for bootstrapping.

```python
class BootstrapFewShot(Teleprompter):
    def __init__(self, metric=None, teacher_settings={}, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1):
        self.metric = metric
        self.teacher_settings = teacher_settings

        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max_rounds
```

**Arguments:**
- `metric` (callable, optional): Metric function to evaluate examples during bootstrapping. Defaults to `None`.
- `teacher_settings` (dict, optional): Settings for teacher predictor. Defaults to empty dictionary.
- `max_bootstrapped_demos` (int, optional): Maximum number of bootstrapped demonstrations per predictor. Defaults to 4.
- `max_labeled_demos` (int, optional): Maximum number of labeled demonstrations per predictor. Defaults to 16.
- `max_rounds` (int, optional): Maximum number of bootstrapping rounds. Defaults to 1.

### Method

#### `compile(self, student, *, teacher=None, trainset, valset=None)`

This method compiles the BootstrapFewShot instance by performing bootstrapping to refine the student predictor.

This process includes preparing the student and teacher predictors, which involves creating predictor copies, verifying the student predictor is uncompiled, and compiling the teacher predictor with labeled demonstrations via LabeledFewShot if the teacher predictor hasn't been compiled.

The next stage involves preparing predictor mappings by validating that both the student and teacher predictors have the same program structure and the same signatures but are different objects.

The final stage is performing the bootstrapping iterations.

**Arguments:**
- `student` (Teleprompter): Student predictor to be compiled.
- `teacher` (Teleprompter, optional): Teacher predictor used for bootstrapping. Defaults to `None`.
- `trainset` (list): Training dataset used in bootstrapping.
- `valset` (list, optional): Validation dataset used in compilation. Defaults to `None`.

**Returns:**
- The compiled `student` predictor after bootstrapping with refined demonstrations.

### Examples

```python
#Assume defined trainset
#Assume defined RAG class
...

#Define teleprompter and include teacher
teacher = dspy.OpenAI(model='gpt-3.5-turbo', api_key = openai.api_key, api_provider = "openai", model_type = "chat")
teleprompter = BootstrapFewShot(teacher_settings=dict({'lm': teacher}))

# Compile!
compiled_rag = teleprompter.compile(student=RAG(), trainset=trainset)
```

## dspy.teleprompt.BootstrapFewShotWithRandomSearch

### Constructor

The constructor initializes the `BootstrapFewShotWithRandomSearch` class and sets up its attributes. It inherits from the `BootstrapFewShot` class and introduces additional attributes for the random search process.

```python
class BootstrapFewShotWithRandomSearch(BootstrapFewShot):
    def __init__(self, metric, teacher_settings={}, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, num_candidate_programs=16, num_threads=6):
        self.metric = metric
        self.teacher_settings = teacher_settings
        self.max_rounds = max_rounds

        self.num_threads = num_threads

        self.min_num_samples = 1
        self.max_num_samples = max_bootstrapped_demos
        self.num_candidate_sets = num_candidate_programs
        self.max_num_traces = 1 + int(max_bootstrapped_demos / 2.0 * self.num_candidate_sets)

        self.max_bootstrapped_demos = self.max_num_traces
        self.max_labeled_demos = max_labeled_demos

        print("Going to sample between", self.min_num_samples, "and", self.max_num_samples, "traces per predictor.")
        print("Going to sample", self.max_num_traces, "traces in total.")
        print("Will attempt to train", self.num_candidate_sets, "candidate sets.")
```

**Arguments:**
- `metric` (callable, optional): Metric function to evaluate examples during bootstrapping. Defaults to `None`.
- `teacher_settings` (dict, optional): Settings for teacher predictor. Defaults to empty dictionary.
- `max_bootstrapped_demos` (int, optional): Maximum number of bootstrapped demonstrations per predictor. Defaults to 4.
- `max_labeled_demos` (int, optional): Maximum number of labeled demonstrations per predictor. Defaults to 16.
- `max_rounds` (int, optional): Maximum number of bootstrapping rounds. Defaults to 1.
- `num_candidate_programs` (int): Number of candidate programs to generate during random search.
- `num_threads` (int): Number of threads used for evaluation during random search.

### Method

Refer to dspy.teleprompt.BootstrapFewShot documentation.

### Examples

```python
#Assume defined trainset
#Assume defined RAG class
...

#Define teleprompter and include teacher
teacher = dspy.OpenAI(model='gpt-3.5-turbo', api_key = openai.api_key, api_provider = "openai", model_type = "chat")
teleprompter = BootstrapFewShotWithRandomSearch(teacher_settings=dict({'lm': teacher}))

# Compile!
compiled_rag = teleprompter.compile(student=RAG(), trainset=trainset)
```

## dspy.teleprompt.BootstrapFinetune

### Constructor

### `__init__(self, metric=None, teacher_settings={}, multitask=True)`

The constructor initializes a `BootstrapFinetune` instance and sets up its attributes. It defines the teleprompter as a `BootstrapFewShot` instance for the finetuning compilation.

```python
class BootstrapFinetune(Teleprompter):
    def __init__(self, metric=None, teacher_settings={}, multitask=True):
```

**Arguments:**
- `metric` (callable, optional): Metric function to evaluate examples during bootstrapping. Defaults to `None`.
- `teacher_settings` (dict, optional): Settings for teacher predictor. Defaults to empty dictionary.
- `multitask` (bool, optional): Enable multitask fine-tuning. Defaults to `True`.

### Method

#### `compile(self, student, *, teacher=None, trainset, valset=None, target='t5-large', bsize=12, accumsteps=1, lr=5e-5, epochs=1, bf16=False)`

This method first compiles for bootstrapping with the `BootstrapFewShot` teleprompter. It then prepares fine-tuning data by generating prompt-completion pairs for training and performs finetuning. After compilation, the LMs are set to the finetuned models and the method returns a compiled and fine-tuned predictor.

**Arguments:**
- `student` (Predict): Student predictor to be fine-tuned.
- `teacher` (Predict, optional): Teacher predictor to help with fine-tuning. Defaults to `None`.
- `trainset` (list): Training dataset for fine-tuning.
- `valset` (list, optional): Validation dataset for fine-tuning. Defaults to `None`.
- `target` (str, optional): Target model for fine-tuning. Defaults to `'t5-large'`.
- `bsize` (int, optional): Batch size for training. Defaults to `12`.
- `accumsteps` (int, optional): Gradient accumulation steps. Defaults to `1`.
- `lr` (float, optional): Learning rate for fine-tuning. Defaults to `5e-5`.
- `epochs` (int, optional): Number of training epochs. Defaults to `1`.
- `bf16` (bool, optional): Enable mixed-precision training with BF16. Defaults to `False`.

**Returns:**
- `compiled2` (Predict): A compiled and fine-tuned `Predict` instance.

### Examples

```python
#Assume defined trainset
#Assume defined RAG class
...

#Define teleprompter
teleprompter = BootstrapFinetune(teacher_settings=dict({'lm': teacher}))

# Compile!
compiled_rag = teleprompter.compile(student=RAG(), trainset=trainset, target='google/flan-t5-base')
```