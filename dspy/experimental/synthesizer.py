import dspy
import random

from datasets import Dataset
from tqdm import tqdm, trange
from typing import List, Union, Mapping

def format_examples(examples: List[dspy.Example]) -> str:
    if isinstance(examples, str):
        return examples

    formatted_example = ""

    for example in examples:
        input_keys = example.inputs().keys()
        label_keys = example.labels().keys()

        formatted_example += "Inputs:\n"
        for key in input_keys:
            formatted_example += f"{key}: {example[key]}\n"

        formatted_example += "Outputs:\n"
        for key in label_keys:
            formatted_example += f"{key}: {example[key]}\n"

    return formatted_example

class UnderstandTask(dspy.Signature):
    """I'll be providing you a task description, your task is to prepare a concise, comprehensible summary that captures the broad essence and purpose of the task this description aim to address. Your summary should illuminate the general objective and the type of problem being solved, offering a clear picture of what the task entails at a high level. Avoid getting into the nuances of individual datapoints, specifics about models, examples, algorithms, or any intricate technicalities. Your explanation should serve to clarify the task's overall goal and its basic premise, without touching on methodologies or solutions."""

    task_description = dspy.InputField(
        prefix="Task Description:",
        desc="Description of the task.",
    )
    explanation = dspy.OutputField(
        prefix="Task Description:",
        desc="Explanation of the task.",
    )

class ExplainTask(dspy.Signature):
    """Analyze the provided set of datapoints carefully, and prepare a concise, comprehensible summary that captures the  broad essence and purpose of the task these datapoints aim to address. Your summary should illuminate the general objective and the type of problem being solved, offering a clear picture of what the task entails at a high level. Avoid getting into the nuances of individual datapoints, specifics about models, examples, algorithms, or any intricate technicalities. Your explanation should serve to clarify the task's overall goal and its basic premise, without touching on methodologies or solutions."""

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
    """Create synthetic data using the task description and the provided knowledge seed. Your task is to generate diverse and imaginative data that aligns with the given task description and knowledge seed. You are encouraged to be creative and not limit yourself, allowing for a wide range of synthetic data that reflects the characteristics and details provided in the task description. The data should be unique and varied, showcasing originality and creativity while maintaining relevance to the task and knowledge seed."""

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

    def _get_field_data(self, key: str, keys_dict: Mapping[str, str]):
        if key.startswith("$"):
            field_details = self.generate_field_description(
                task_description=keys_dict["task_description"],
                field_name=key,
            )

            field_name = key
            field_description = field_details.field_description

            return field_name, field_description

        else:
            field_name = key
            field_description = keys_dict[key]

            return field_name, field_description

    def _prepare_synthetic_data_predictors(self, input_keys: Mapping[str, str], output_keys: Mapping[str, str], task_description: str):
        for key in tqdm(input_keys, desc="Preparing Input Fields"):
            field_name, field_description = self._get_field_data(key, input_keys)

            output_field = dspy.OutputField(
                prefix=f"{field_name}:",
                desc=field_description,
            )
            self.generate_input_data = self.generate_input_data.insert(
                -1,
                field_name,
                output_field,
            )

            input_field = dspy.InputField(
                prefix=f"{field_name}:",
                desc=field_description,
            )
            self.generate_output_data = self.generate_output_data.insert(
                -1,
                field_name,
                input_field,
            )

        for key in tqdm(output_keys, desc="Preparing Output Fields"):
            field_name, field_description = self._get_field_data(key, output_keys)

            output_field = dspy.OutputField(
                prefix=f"{field_name}:",
                desc=field_description,
            )
            self.generate_output_data = self.generate_output_data.insert(
                -1,
                field_name,
                output_field,
            )

        return dspy.ChainOfThought(self.generate_input_data), dspy.Predict(self.generate_output_data)

    def _get_dataset_metadata(self, ground_source: Union[List[dspy.Example], dspy.Signature]):
        if isinstance(ground_source, dspy.SignatureMeta):
            task_description = self.explain_task(examples=ground_source.__doc__).explanation
            input_keys = {k:v.json_schema_extra["desc"] for k,v in ground_source.input_fields.items()}
            output_keys = {k:v.json_schema_extra["desc"] for k,v in ground_source.output_fields.items()}

            return task_description, input_keys, output_keys

        elif isinstance(ground_source, list) and isinstance(ground_source[0], dspy.Example):
            task_description = self.explain_task(examples=ground_source).explanation
            input_keys = {key:f"${{{key}}}" for key in ground_source[0].inputs()}
            output_keys = {key:f"${{{key}}}" for key in ground_source[0].labels()}

            return task_description, input_keys, output_keys

        else:
            raise ValueError("Ground source must be either a list of examples or a signature.")

    def generate(
        self,
        ground_source: Union[List[dspy.Example], dspy.Signature],
        num_data: int,
        batch_size: int = None,
    ):
        batch_size = batch_size or 1
        task_description, input_keys, output_keys = self._get_dataset_metadata(ground_source)

        self.generate_output_data.__doc__ = task_description

        self.input_predictor, self.output_predictor = self._prepare_synthetic_data_predictors(
            input_keys=input_keys,
            output_keys=output_keys,
            task_description=task_description,
        )

        data = []

        for idx in trange(0, num_data, batch_size, desc="Generating Synthetic Data"):
            iter_temperature = 0.7+0.01*idx
            iter_seed = random.randint(0, 1000000)

            inputs = self.input_predictor(task_description=task_description, knowledge_seed=iter_seed, config=dict(temperature=iter_temperature, n=batch_size))

            input_kwargs = [{
                key: getattr(completions, key)
                for key in input_keys
            } for completions in inputs.completions]

            for kwargs in input_kwargs:
                outputs = self.output_predictor(**kwargs, config=dict(temperature=iter_temperature))

                output_kwargs = {
                    key: getattr(outputs, key)
                    for key in output_keys
                }

                data.append(dspy.Example(**kwargs, **output_kwargs).with_inputs(*input_keys))

        return data


    def export(self, data: List[dspy.Example], path: str, mode: str = None, **kwargs):
        extention = mode or path.split(".")[-1]

        dataset = Dataset.from_list(
            [example.toDict() for example in data],
        )

        if extention == "csv":
            dataset.to_csv(path_or_buf=path, **kwargs)

        elif extention == "json":
            dataset.to_json(path_or_buf=path, **kwargs)

        elif extention == "arrow" or extention == "hf":
            dataset.save_to_disk(path)