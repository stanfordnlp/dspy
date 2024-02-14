import dspy
from typing import List

class ExplainTask(dspy.Signature):
    """Imagine you're a detective in a game where your mission is to unlock the mystery of a hidden treasure. The treasure map, in this case, is the task description. Your first step is to study the map carefully to understand where the treasure is hidden and how to get there. This means figuring out what the task is all about—like decoding clues. Once you've got a good grasp, your job is to explain your plan to your team in a way that's super easy to understand, as if you're telling a friend how to find the treasure without using the map. You won't be using the clues directly in your explanation but rather your understanding of them to guide your team clearly and simply to the treasure."""

    @staticmethod
    def format_examples(examples: List[dspy.Example]):
        if isinstance(examples, str):
            return examples

        formatted_example = ""

        for example in examples:
            input_keys = example.inputs().keys()
            label_keys = example.labels().keys()

            formatted_example += f"Input:\n"
            for key in input_keys:
                formatted_example += f"{key}: {example[key]}\n"

            formatted_example += f"Output:\n"
            for key in label_keys:
                formatted_example += f"{key}: {example[key]}\n"

        return formatted_example

    examples = dspy.InputField(
        prefix="Few Shot Examples:-",
        desc="List of examples to analyze and explain the task.",
        format=format_examples,
    )
    explanation = dspy.OutputField(
        prefix="Explanation:",
        desc="Explanation of the task.",
    )

class GenerateFieldDescription(dspy.Signature):
    """I'll be providing you with the name of the field and the task description. Your task is to generate a description for the field. The description should be such that it is easy to understand and gives a clear idea of what the field is about."""
    
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
    """You are an expert data generator with 30 years of experience in generating synthetic data. We want you to put these skills at work, I'll be providing you with some input fields that are columns of the csv file and the explanation of the task thse fields would be an input to. Your task is to generate synthetic for these fields."""
    pass

class GenerateOutputFieldsData(dspy.Signature):
    pass

class Synthesizer:
    def __init__(self):
        self.explain_task = ExplainTask()
        self.generate_field_description = GenerateFieldDescription()

        self.generate_input_data = GenerateInputFieldsData()
        self.generate_output_data = GenerateOutputFieldsData()

    def _prepare_synthetic_data_signature(self, signature: dspy.Signature):
        signature

    def generate(self, examples: List[dspy.Example], num_data: int) -> List[dspy.Example]:
        input_keys = examples[0].keys()

        task_description = self.explain_task(examples=examples)
        self.generate_output_data.__doc__ = task_description

        self._prepare_synthetic_data_signature()

        data = []
        for _ in range(num_data):
            synthetic_data = {field: self.generate_synthetic_data() for field in fields}
            data.append(synthetic_data)
            

    def export(self):
        pass

    def _to_csv(self):
        pass

    def _to_jsonl(self):
        pass

    def _to_pickle(self):
        pass

    def _to_sql(self):
        pass

    def _to_parquet(self):
        pass

    def _to_arrow(self):
        pass