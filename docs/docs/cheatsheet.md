---
sidebar_position: 999
---


# DSPy Cheatsheet

This page will contain snippets for frequent usage patterns.

## DSPy DataLoaders

Import and initializing a DataLoader Object:

```python
import dspy
from dspy.datasets import DataLoader

dl = DataLoader()
```

### Loading from HuggingFace Datasets

```python
code_alpaca = dl.from_huggingface("HuggingFaceH4/CodeAlpaca_20K")
```

You can access the dataset of the splits by calling key of the corresponding split:

```python
train_dataset = code_alpaca['train']
test_dataset = code_alpaca['test']
```

### Loading specific splits from HuggingFace

You can also manually specify splits you want to include as a parameters and it'll return a dictionary where keys are splits that you specified:

```python
code_alpaca = dl.from_huggingface(
    "HuggingFaceH4/CodeAlpaca_20K",
    split = ["train", "test"],
)

print(f"Splits in dataset: {code_alpaca.keys()}")
```

If you specify a single split then dataloader will return a List of `dspy.Example` instead of dictionary:

```python
code_alpaca = dl.from_huggingface(
    "HuggingFaceH4/CodeAlpaca_20K",
    split = "train",
)

print(f"Number of examples in split: {len(code_alpaca)}")
```

You can slice the split just like you do with HuggingFace Dataset too:

```python
code_alpaca_80 = dl.from_huggingface(
    "HuggingFaceH4/CodeAlpaca_20K",
    split = "train[:80%]",
)

print(f"Number of examples in split: {len(code_alpaca_80)}")

code_alpaca_20_80 = dl.from_huggingface(
    "HuggingFaceH4/CodeAlpaca_20K",
    split = "train[20%:80%]",
)

print(f"Number of examples in split: {len(code_alpaca_20_80)}")
```

### Loading specific subset from HuggingFace

If a dataset has a subset you can pass it as an arg like you do with `load_dataset` in HuggingFace:

```python
gms8k = dl.from_huggingface(
    "gsm8k",
    "main",
    input_keys = ("question",),
)

print(f"Keys present in the returned dict: {list(gms8k.keys())}")

print(f"Number of examples in train set: {len(gms8k['train'])}")
print(f"Number of examples in test set: {len(gms8k['test'])}")
```

### Loading from CSV

```python
dolly_100_dataset = dl.from_csv("dolly_subset_100_rows.csv",)
```

You can choose only selected columns from the csv by specifying them in the arguments:

```python
dolly_100_dataset = dl.from_csv(
    "dolly_subset_100_rows.csv",
    fields=("instruction", "context", "response"),
    input_keys=("instruction", "context")
)
```

### Splitting a List of `dspy.Example`

```python
splits = dl.train_test_split(dataset, train_size=0.8) # `dataset` is a List of dspy.Example
train_dataset = splits['train']
test_dataset = splits['test']
```

### Sampling from List of `dspy.Example`

```python
sampled_example = dl.sample(dataset, n=5) # `dataset` is a List of dspy.Example
```

## DSPy Programs

### dspy.Signature

```python
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
```

### dspy.ChainOfThought

```python
generate_answer = dspy.ChainOfThought(BasicQA)

# Call the predictor on a particular input alongside a hint.
question='What is the color of the sky?'
pred = generate_answer(question=question)
```

### dspy.ChainOfThoughtwithHint

```python
generate_answer = dspy.ChainOfThoughtWithHint(BasicQA)

# Call the predictor on a particular input alongside a hint.
question='What is the color of the sky?'
hint = "It's what you often see during a sunny day."
pred = generate_answer(question=question, hint=hint)
```

### dspy.ProgramOfThought

```python
pot = dspy.ProgramOfThought(BasicQA)

question = 'Sarah has 5 apples. She buys 7 more apples from the store. How many apples does Sarah have now?'
result = pot(question=question)

print(f"Question: {question}")
print(f"Final Predicted Answer (after ProgramOfThought process): {result.answer}")
```

### dspy.ReACT

```python
react_module = dspy.ReAct(BasicQA)

question = 'Sarah has 5 apples. She buys 7 more apples from the store. How many apples does Sarah have now?'
result = react_module(question=question)

print(f"Question: {question}")
print(f"Final Predicted Answer (after ReAct process): {result.answer}")
```

### dspy.Retrieve

```python
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)

#Define Retrieve Module
retriever = dspy.Retrieve(k=3)

query='When was the first FIFA World Cup held?'

# Call the retriever on a particular query.
topK_passages = retriever(query).passages

for idx, passage in enumerate(topK_passages):
    print(f'{idx+1}]', passage, '\n')
```

## DSPy Metrics

### Function as Metric

To create a custom metric you can create a function that returns either a number or a boolean value:

```python
def parse_integer_answer(answer, only_first_line=True):
    try:
        if only_first_line:
            answer = answer.strip().split('\n')[0]

        # find the last token that has a number in it
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][-1]
        answer = answer.split('.')[0]
        answer = ''.join([c for c in answer if c.isdigit()])
        answer = int(answer)

    except (ValueError, IndexError):
        # print(answer)
        answer = 0
    
    return answer

# Metric Function
def gsm8k_metric(gold, pred, trace=None) -> int:
    return int(parse_integer_answer(str(gold.answer))) == int(parse_integer_answer(str(pred.answer)))
```

### LLM as Judge

```python
class FactJudge(dspy.Signature):
    """Judge if the answer is factually correct based on the context."""

    context = dspy.InputField(desc="Context for the prediciton")
    question = dspy.InputField(desc="Question to be answered")
    answer = dspy.InputField(desc="Answer for the question")
    factually_correct = dspy.OutputField(desc="Is the answer factually correct based on the context?", prefix="Factual[Yes/No]:")

judge = dspy.ChainOfThought(FactJudge)

def factuality_metric(example, pred):
    factual = judge(context=example.context, question=example.question, answer=pred.answer)
    return int(factual=="Yes")
```

## DSPy Evaluation

```python
from dspy.evaluate import Evaluate

evaluate_program = Evaluate(devset=devset, metric=your_defined_metric, num_threads=NUM_THREADS, display_progress=True, display_table=num_rows_to_display)

evaluate_program(your_dspy_program)
```

## DSPy Optimizers

### LabeledFewShot 
```python
from dspy.teleprompt import LabeledFewShot

labeled_fewshot_optimizer = LabeledFewShot(k=8)
your_dspy_program_compiled = labeled_fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset)
```

### BootstrapFewShot 
```python
from dspy.teleprompt import BootstrapFewShot

fewshot_optimizer = BootstrapFewShot(metric=your_defined_metric, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, max_errors=5)

your_dspy_program_compiled = fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset)
```

#### Using another LM for compilation, specifying in teacher_settings
```python
from dspy.teleprompt import BootstrapFewShot

fewshot_optimizer = BootstrapFewShot(metric=your_defined_metric, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, max_errors=5, teacher_settings=dict(lm=gpt4))

your_dspy_program_compiled = fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset)
```

#### Compiling a compiled program - bootstrapping a bootstrapped program

```python
your_dspy_program_compiledx2 = teleprompter.compile(
    your_dspy_program,
    teacher=your_dspy_program_compiled,
    trainset=trainset,
)
```

#### Saving/loading a compiled program

```python
save_path = './v1.json'
your_dspy_program_compiledx2.save(save_path)
```

```python
loaded_program = YourProgramClass()
loaded_program.load(path=save_path)
```

### BootstrapFewShotWithRandomSearch

Detailed documentation on BootstrapFewShotWithRandomSearch can be found [here](/docs/deep-dive/optimizers/bootstrap-fewshot.mdx).


```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

fewshot_optimizer = BootstrapFewShotWithRandomSearch(metric=your_defined_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=NUM_THREADS)

your_dspy_program_compiled = fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset, valset=devset)

```
Other custom configurations are similar to customizing the `BootstrapFewShot` optimizer. 


### Ensemble

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.ensemble import Ensemble

fewshot_optimizer = BootstrapFewShotWithRandomSearch(metric=your_defined_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=NUM_THREADS)
your_dspy_program_compiled = fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset, valset=devset)

ensemble_optimizer = Ensemble(reduce_fn=dspy.majority)
programs = [x[-1] for x in your_dspy_program_compiled.candidate_programs]
your_dspy_program_compiled_ensemble = ensemble_optimizer.compile(programs[:3])
```

### BootstrapFinetune

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFinetune

#Compile program on current dspy.settings.lm
fewshot_optimizer = BootstrapFewShotWithRandomSearch(metric=your_defined_metric, max_bootstrapped_demos=2, num_threads=NUM_THREADS)
your_dspy_program_compiled = tp.compile(your_dspy_program, trainset=trainset[:some_num], valset=trainset[some_num:])

#Configure model to finetune
config = dict(target=model_to_finetune, epochs=2, bf16=True, bsize=6, accumsteps=2, lr=5e-5)

#Compile program on BootstrapFinetune
finetune_optimizer = BootstrapFinetune(metric=your_defined_metric)
finetune_program = finetune_optimizer.compile(your_dspy_program, trainset=some_new_dataset_for_finetuning_model, **config)

finetune_program = your_dspy_program

#Load program and activate model's parameters in program before evaluation
ckpt_path = "saved_checkpoint_path_from_finetuning"
LM = dspy.HFModel(checkpoint=ckpt_path, model=model_to_finetune)

for p in finetune_program.predictors():
    p.lm = LM
    p.activated = False
```

### COPRO

Detailed documentation on COPRO can be found [here](/docs/deep-dive/optimizers/copro.mdx).


```python
from dspy.teleprompt import COPRO

eval_kwargs = dict(num_threads=16, display_progress=True, display_table=0)

copro_teleprompter = COPRO(prompt_model=model_to_generate_prompts, metric=your_defined_metric, breadth=num_new_prompts_generated, depth=times_to_generate_prompts, init_temperature=prompt_generation_temperature, verbose=False)

compiled_program_optimized_signature = copro_teleprompter.compile(your_dspy_program, trainset=trainset, eval_kwargs=eval_kwargs)
```

### MIPRO


```python
from dspy.teleprompt import MIPRO

teleprompter = MIPRO(prompt_model=model_to_generate_prompts, task_model=model_that_solves_task, metric=your_defined_metric, num_candidates=num_new_prompts_generated, init_temperature=prompt_generation_temperature)

kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=0)

compiled_program_optimized_bayesian_signature = teleprompter.compile(your_dspy_program, trainset=trainset, num_trials=100, max_bootstrapped_demos=3, max_labeled_demos=5, eval_kwargs=kwargs)
```

### MIPROv2

Note: detailed documentation can be found [here](/docs/deep-dive/optimizers/miprov2.md). `MIPROv2` is the latest extension of `MIPRO` which includes updates such as (1) improvements to instruction proposal and (2) more efficient search with minibatching.

#### Optimizing with MIPROv2
```python
# Import the optimizer
from dspy.teleprompt import MIPROv2

# Initialize optimizer
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    num_candidates=7,
    init_temperature=0.5,
    verbose=False,
    num_threads=4,
)

# Optimize program
print(f"Optimizing program with MIPRO...")
optimized_program = teleprompter.compile(
    program.deepcopy(),
    trainset=trainset,
    max_bootstrapped_demos=3,
    max_labeled_demos=4,
    num_trials=15,
    minibatch_size=25,
    minibatch_full_eval_steps=10,
    minibatch=True, 
    requires_permission_to_run=False,
)

# Save optimize program for future use
optimized_program.save(f"mipro_optimized")

# Evaluate optimized program
print(f"Evluate optimized program...")
evaluate(optimized_program, devset=devset[:])
```

#### Optimizing instructions only with MIPROv2 (0-Shot)
```python
# Import optimizer
from dspy.teleprompt import MIPROv2

# Initialize optimizer 
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    num_candidates=15,
    init_temperature=0.5,
    verbose=False,
    num_threads=4,
)

# Perform optimization
print(f"Optimizing program with MIPRO (0-Shot)...")
zeroshot_optimized_program = teleprompter.compile(
    program.deepcopy(),
    trainset=trainset,
    max_bootstrapped_demos=0, # setting demos to 0 for 0-shot optimization
    max_labeled_demos=0,
    num_trials=15,
    minibatch=False, 
    requires_permission_to_run=False,
)


zeroshot_optimized_program.save(f"mipro_0shot_optimized")

print(f"Evaluate optimized program...")
evaluate(zeroshot_optimized_program, devset=devset[:])
```
### Signature Optimizer with Types

```python
from dspy.teleprompt.signature_opt_typed import optimize_signature
from dspy.evaluate.metrics import answer_exact_match
from dspy.functional import TypedChainOfThought

compiled_program = optimize_signature(
    student=TypedChainOfThought("question -> answer"),
    evaluator=Evaluate(devset=devset, metric=answer_exact_match, num_threads=10, display_progress=True),
    n_iterations=50,
).program
```

### KNNFewShot

```python
from dspy.predict import KNN
from dspy.teleprompt import KNNFewShot

knn_optimizer = KNNFewShot(KNN, k=3, trainset=trainset)

your_dspy_program_compiled = knn_optimizer.compile(student=your_dspy_program, trainset=trainset, valset=devset)
```

### BootstrapFewShotWithOptuna

```python
from dspy.teleprompt import BootstrapFewShotWithOptuna

fewshot_optuna_optimizer = BootstrapFewShotWithOptuna(metric=your_defined_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=NUM_THREADS)

your_dspy_program_compiled = fewshot_optuna_optimizer.compile(student=your_dspy_program, trainset=trainset, valset=devset)
```
Other custom configurations are similar to customizing the `dspy.BootstrapFewShot` optimizer. 


## DSPy Assertions

### Including `dspy.Assert` and `dspy.Suggest` statements
```python
dspy.Assert(your_validation_fn(model_outputs), "your feedback message", target_module="YourDSPyModule")

dspy.Suggest(your_validation_fn(model_outputs), "your feedback message", target_module="YourDSPyModule")
```

### Activating DSPy Program with Assertions 

**Note**: To use Assertions properly, you must **activate** a DSPy program that includes `dspy.Assert` or `dspy.Suggest` statements from either of the methods above. 

```python
#1. Using `assert_transform_module:
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

program_with_assertions = assert_transform_module(ProgramWithAssertions(), backtrack_handler)

#2. Using `activate_assertions()`
program_with_assertions = ProgramWithAssertions().activate_assertions()
```

## Compiling with DSPy Programs with Assertions

```python
program_with_assertions = assert_transform_module(ProgramWithAssertions(), backtrack_handler)
fewshot_optimizer = BootstrapFewShotWithRandomSearch(metric = your_defined_metric, max_bootstrapped_demos=2, num_candidate_programs=6)
compiled_dspy_program_with_assertions = fewshot_optimizer.compile(student=program_with_assertions, teacher = program_with_assertions, trainset=trainset, valset=devset) #student can also be program_without_assertions
```
