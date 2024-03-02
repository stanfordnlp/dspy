import dspy
import random
from typing import List
from tqdm import tqdm, trange
from datasets import Dataset

def format_examples(examples: List[dspy.Example]):
    if isinstance(examples, str):
        return examples

    formatted_example = ""

    for example in examples:
        input_keys = example.inputs().keys()
        label_keys = example.labels().keys()

        formatted_example += f"Inputs:\n"
        for key in input_keys:
            formatted_example += f"{key}: {example[key]}\n"

        formatted_example += f"Outputs:\n"
        for key in label_keys:
            formatted_example += f"{key}: {example[key]}\n"

    return formatted_example

class ExplainTask(dspy.Signature):
    """Analyze the provided set of datapoints carefully, and prepare a concise, comprehensible summary that captures the essence and purpose of the task these datapoints aim to address. Your summary should illuminate the general objective and the type of problem being solved, offering a clear picture of what the task entails at a high level. Avoid getting into the nuances of individual datapoints, specifics about models, examples, algorithms, or any intricate technicalities. Your explanation should serve to clarify the task's overall goal and its basic premise, without touching on methodologies or solutions."""

    examples = dspy.InputField(
        prefix="Examples Datapoints:-",
        desc="List of datapoints to analyze and explain the task.",
        format=format_examples,
    )
    explanation = dspy.OutputField(
        prefix="Task Description:",
        desc="Explanation of the task.",
    )

class GenerateFieldDescription(dspy.Signature):
    """Generate a concise and informative description for a given field based on the provided name and task description. This description should be no longer than 10 words and should be in simple english."""

    task_description = dspy.InputField(
        prefix="Task Description:",
        desc="Description of the task the field is an input to.",
    )
    field_name = dspy.InputField(
        prefix="Field Name:",
        desc="Name of the field to generate synthetic data for.",
    )
    field_description = dspy.OutputField(
        prefix="Field Description:",
        desc="Description of the field.",
    )

class GenerateInputFieldsData(dspy.Signature):
    """Generate synthetic data based on the task description and the given knowledge seed."""
    
    knowledge_seed = dspy.InputField(
        prefix="Knowledge Seed:",
        desc="Seed for the knowledge base search to base the inputs around.",
        format=lambda x: str(x),
    )
    task_description = dspy.InputField(
        prefix="Task Description:",
        desc="Description of the task the field is an input to.",
    )

class GenerateOutputFieldsData(dspy.Signature):
    pass

class Synthesizer:
    def __init__(self):
        self.explain_task = dspy.Predict(ExplainTask)
        self.generate_field_description = dspy.Predict(GenerateFieldDescription)

        self.generate_input_data = GenerateInputFieldsData
        self.generate_output_data = GenerateOutputFieldsData

    def _prepare_synthetic_data_predictors(self, input_keys: List[str], output_keys: List[str], task_description: str):
        for key in tqdm(input_keys, desc="Preparing Input Fields"):
            field_details = self.generate_field_description(
                task_description=task_description,
                field_name=key,
            )

            field_name = key
            field_description = field_details.field_description

            output_field = dspy.OutputField(
                prefix=f"{field_name}:",
                desc=field_description,
            )
            self.generate_input_data = self.generate_input_data.insert(
                -1,
                field_name,
                output_field
            )

            input_field = dspy.InputField(
                prefix=f"{field_name}:",
                desc=field_description,
            )
            self.generate_output_data = self.generate_output_data.insert(
                -1,
                field_name,
                input_field
            )

        for key in tqdm(output_keys, desc="Preparing Output Fields"):
            field_details = self.generate_field_description(
                task_description=task_description,
                field_name=key,
            )

            field_name = key
            field_description = field_details.field_description

            output_field = dspy.OutputField(
                prefix=f"{field_name}:",
                desc=field_description,
            )
            self.generate_output_data = self.generate_output_data.insert(
                -1,
                field_name,
                output_field
            )

        return dspy.ChainOfThought(self.generate_input_data), dspy.Predict(self.generate_output_data)

    def generate(self, examples: List[dspy.Example], num_data: int, task_description: str = None, input_keys: str = None, output_keys: str = None) -> List[dspy.Example]:
        task_description = task_description or self.explain_task(examples=examples).explanation
        self.generate_output_data.__doc__ = task_description

        input_keys = input_keys or [key for key in examples[0].inputs()]
        output_keys = output_keys or [key for key in examples[0].labels()]

        self.input_predictor, self.output_predictor = self._prepare_synthetic_data_predictors(
            input_keys=input_keys,
            output_keys=output_keys,
            task_description=task_description,
        )

        data = []

        for idx in trange(num_data, desc="Generating Synthetic Data"):
            inputs = self.input_predictor(task_description=task_description, knowledge_seed=random.randint(0, 1000000), config=dict(temperature=0.7+0.01*idx))

            input_kwargs = {
                key: getattr(inputs, key)
                for key in input_keys
            }

            outputs = self.output_predictor(**input_kwargs, config=dict(temperature=0.7+0.01*idx))

            output_kwargs = {
                key: getattr(outputs, key)
                for key in output_keys
            }

            data.append(dspy.Example(**input_kwargs, **output_kwargs).with_inputs(*input_keys))

        return data


    def export(self, data: List[dspy.Example], path: str, mode: str = None, **kwargs):
        extention = mode or path.split(".")[-1]

        dataset = Dataset.from_list(
            [example.toDict() for example in data]
        )

        if extention == "csv":
            dataset.to_csv(path_or_buf=path, **kwargs)

        elif extention == "json":
            dataset.to_json(path_or_buf=path, **kwargs)

        elif extention == "arrow" or extention == "hf":
            dataset.save_to_disk(path)