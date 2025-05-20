import logging
import random
from typing import List, Optional

from pydantic import BaseModel

import dspy


class DescriptionSignature(dspy.Signature):
    field_name = dspy.InputField(desc="name of a field")
    example = dspy.InputField(desc="an example value for the field")
    description = dspy.OutputField(desc="a short text only description of what the field contains")


class SyntheticDataGenerator:
    def __init__(self, schema_class: Optional[BaseModel] = None, examples: Optional[List[dspy.Example]] = None):
        self.schema_class = schema_class
        self.examples = examples

    def generate(self, sample_size: int) -> List[dspy.Example]:
        """Generate synthetic examples.

        Args:
            sample_size (int): number of examples to generate
        Raises:
            ValueError: either a schema_class or examples should be provided
        Returns:
            List[dspy.Example]: list of synthetic examples generated
        """
        if not self.schema_class and not self.examples:
            raise ValueError("Either a schema_class or examples must be provided.")
        if self.examples and len(self.examples) >= sample_size:
            logging.info("No additional data generation needed.")
            return self.examples[:sample_size]

        additional_samples_needed = sample_size - (len(self.examples) if self.examples else 0)
        generated_examples = self._generate_additional_examples(additional_samples_needed)

        return self.examples + generated_examples if self.examples else generated_examples

    def _define_or_infer_fields(self):
        """Define fields to generate if a schema class is provided.
        Infer fields to generate if an initial sample of examples is provided.

        Returns:
            dict: dictionary of fields to generate
        """
        if self.schema_class:
            data_schema = self.schema_class.model_json_schema()
            properties = data_schema["properties"]
        elif self.examples:
            inferred_schema = self.examples[0].__dict__["_store"]
            descriptor = dspy.Predict(DescriptionSignature)
            properties = {
                field: {
                    "description": str((descriptor(field_name=field, example=str(inferred_schema[field]))).description)
                }
                for field in inferred_schema.keys()
            }
        else:
            properties = {}
        return properties

    def _generate_additional_examples(self, additional_samples_needed: int) -> List[dspy.Example]:
        """Generate additional examples if needed.

        Args:
            additional_samples_needed (int): the difference between the desired
            number of examples and the current number of examples
        Returns:
            List[dspy.Example]: list of synthetic examples
        """
        properties = self._define_or_infer_fields()
        class_name = f"{self.schema_class.__name__ if self.schema_class else 'Inferred'}Signature"
        fields = self._prepare_fields(properties)

        signature_class = type(class_name, (dspy.Signature,), fields)
        generator = dspy.Predict(signature_class, n=additional_samples_needed)
        response = generator(sindex=str(random.randint(1, additional_samples_needed)))

        return [
            dspy.Example({field_name: getattr(completion, field_name) for field_name in properties.keys()})
            for completion in response.completions
        ]

    def _prepare_fields(self, properties) -> dict:
        """Prepare fields to generate in an appropriate format."""
        return {
            "__doc__": f"Generates the following outputs: {{{', '.join(properties.keys())}}}.",
            "sindex": dspy.InputField(desc="a random string"),
            **{
                field_name: dspy.OutputField(desc=properties[field_name].get("description", "No description"))
                for field_name in properties.keys()
            },
        }


# # Usage example
# # Generating synthetic data via a pydantic model
# generator = SyntheticDataGenerator(schema_class=SyntheticFacts)
# examples = generator.generate(sample_size=6)

# # Generating synthetic data via existing examples
# generator = SyntheticDataGenerator(examples=existing_examples)
# examples = generator.generate(sample_size=5)
