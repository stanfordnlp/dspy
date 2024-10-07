# Optimizer Tester

Optimizer Tester is intended to allow simple and repeatable testing of DSPy Optimizers.  This is a development tool for the creation of more optimizers, and the validation that they work across tasks.  This is still under development and subject to change.

## Usage 

To use the Optimizer Tester in code instantiate an OptimizerTester object:

```python
from optimizer_tester import OptimizerTester

tester = OptimizerTester()
```

The default verison (no parameters) expects a llama model hosted on ports [7140, 7141, 7142, 7143] and OpenAI keys stored in a .env file (OPENAI_API_KEY and OPENAI_API_BASE).

If you prefer to specify your own model parameters then you can pass models into the OptimizerTester

```python
task_model = dspy.LM(...)
prompt_model = dspy.LM(...)

tester = OptimizerTester(task_model=task_model, prompt_model=prompt_model)
```

If you just want to get baseline results for a particular task you're ready to go!

```python
tester.test_baseline(datasets=["hotpotqa", "gsm8k", "scone"])
```

If you want to test out a custom optimizer you'll have to write a quick function to call it properly:

```python
def your_optimizer_caller(default_program, trainset, devset, test_name, dataset_name, kwargs):
    # initialize your teleprompter (optimizer) here!
    teleprompter = my_custom_optimizer(metric=kwargs["metric"], ...)

    # call your optimizer on the default_program here!
    compiled_program = teleprompter.compile(default_program.deepcopy(), trainset=trainset, ...)

    # if you wish to tweak any of the outputs to the csv file you can do that here
    output = {
        "test_name": "my_optimizer-" + dataset_name + "-" + test_name
    }

    # return the compiled program and modified output (or empty dict if no changes made)
    return compiled_program, output
```

You can use these kwargs:
- breadth=self.BREADTH
- depth=self.DEPTH
- temperature=self.TEMPERATURE
- prompt_model=self.prompt_model
- view_data=False
- log_dir=log_dir
- metric=task.get_metric()
- task_model=self.task_model

These are somewhat specific to the testing for the signature optimizer teleprompter which this tool was originally built for, so you probably will not use most of these.

Here is an example of an implemented version of the function:

```python
def signature_optimizer_default(default_program, trainset, devset, test_name, dataset_name, kwargs):
    eval_kwargs = dict(num_threads=16, display_progress=True, display_table=0)

    teleprompter = SignatureOptimizer(prompt_model=kwargs["prompt_model"], task_model=kwargs["task_model"], metric=kwargs["metric"], breadth=kwargs["breadth"], depth=kwargs["depth"], init_temperature=kwargs["temperature"], verbose=False, log_dir=kwargs["log_dir"])
    compiled_program = teleprompter.compile(default_program.deepcopy(), devset=trainset, evalset=devset, eval_kwargs=eval_kwargs)

    output = {
        "meta_prompt_style": "best",
        "view_data": True,
        "test_name": dataset_name + "_" + test_name
    }

    return compiled_program, output
```

Then you can use the 'test_optimizer_default' test to run the tests on your optimizer:

```python
tester.test_optimizer_default(signature_optimizer_default, datasets=["BioDex", "Tweet", "Tweet Metric"])
```

We are working on more optimizer tests with interesting compositions of optimizers, and other hyperparameter adjustments, which will be added as they are developed.

## Important Note

The BioDex Task requires an external download.  Please see the `/tasks/biodex.py` file for a link to download details and a note about where to insert the download path in the code.