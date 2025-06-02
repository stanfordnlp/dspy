from typing import List

import dspy


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
