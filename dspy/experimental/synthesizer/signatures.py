import dspy

from .utils import format_examples


class UnderstandTask(dspy.Signature):
    """I'll be providing you a task description. Your task is to prepare a concise, comprehensible summary that captures the broad essence and purpose of this task description. Your summary should illuminate the general objective and the type of problem being solved, offering a clear picture of what the task entails at a high level. Avoid getting into the nuances or specifics of individual datapoints, models, examples, algorithms, or any intricate technicalities. Your explanation should serve to clarify the task's overall goal and its basic premise without touching on methodologies or solutions."""

    task_description = dspy.InputField(
        prefix="Task Description:",
        desc="Description of the task.",
    )
    explanation = dspy.OutputField(
        prefix="Task Description:",
        desc="Explanation of the task.",
    )


class ExplainTask(dspy.Signature):
    """Analyze the provided set of datapoints carefully, and prepare a concise, comprehensible summary that captures the broad essence and purpose of the task these datapoints aim to address. Your summary should illuminate the general objective and the type of problem being solved, offering a clear picture of what the task entails at a high level. Avoid getting into the nuances of individual datapoints, specifics about models, examples, algorithms, or any intricate technicalities. Your explanation should serve to clarify the task's overall goal and its basic premise, without touching on methodologies or solutions."""

    examples = dspy.InputField(
        prefix="Examples Datapoints:",
        desc="List of datapoints to analyze and explain the task.",
        format=format_examples,
    )
    explanation = dspy.OutputField(
        prefix="Task Description:",
        desc="Explanation of the task.",
    )


class UpdateTaskDescriptionBasedOnFeedback(dspy.Signature):
    """Update the task description based on the feedback provided. Ensure that the revised task description incorporates the feedback to improve its overall clarity and effectiveness. Focus on enhancing the task's goal and basic premise, without delving into specific data points, models, examples, algorithms, or technical intricacies. Your explanation should aim to clarify the task's fundamental objective and purpose."""

    task_description = dspy.InputField(
        prefix="Task Description:",
        desc="Description of the task.",
    )
    feedback = dspy.InputField(
        prefix="Feedback:",
        desc="Feedback on the task description.",
    )
    updated_task_description = dspy.OutputField(
        prefix="Task Description:",
        desc="Updated description of the task.",
    )


class GetFeedbackOnGeneration(dspy.Signature):
    """Provide constructive feedback on the synthetic data generated, focusing on its quality, relevance, and diversity. Highlight any areas that require improvement and offer suggestions for enhancement. The feedback should center on the overall effectiveness of the synthetic data in aligning with the task description and knowledge seed. Avoid delving into specific data points, models, examples, algorithms, or technical intricacies. Your feedback should be critical but constructive, aiming to improve the synthetic data and the task description."""

    synthetic_data = dspy.InputField(
        prefix="Synthetic Data:",
        desc="Synthetic data generated.",
        format=format_examples,
    )
    task_description = dspy.InputField(
        prefix="Task Description:",
        desc="Description of the task the synthetic data is aligned with.",
    )
    feedback = dspy.OutputField(
        prefix="Feedback:",
        desc="Feedback on the synthetic data.",
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
    """Create synthetic data using the task description and the provided knowledge seed. Your task is to generate diverse and imaginative data that aligns with the given task description and knowledge seed. You are encouraged to be creative and not limit yourself, allowing for a wide range of synthetic data that reflects the characteristics and details provided in the task description. The data should be unique and varied, showcasing originality and creativity while maintaining relevance to the task and knowledge seed.

    A knowledge seed is the index of the knowledge base you have, each index represents a different knowledge base."""

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
