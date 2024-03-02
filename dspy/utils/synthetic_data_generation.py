from pydantic import BaseModel
import dspy
import random

def synthetic_data_generation(schema_class: BaseModel, sample_size: int):
    class_name = f"{schema_class.__name__}Signature"

    # Fetch schema information
    data_schema = schema_class.model_json_schema()
    properties = data_schema['properties']

    fields = {
        '__doc__': f"Generates the following outputs: {{{', '.join(properties.keys())}}}.",
        'sindex': dspy.InputField(desc="a random string")
    }

    for field_name, field_info in properties.items():
        fields[field_name] = dspy.OutputField(desc=field_info.get('description', 'No description'))

    signature_class = type(class_name, (dspy.Signature,), fields)

    generator = dspy.Predict(signature_class, n=sample_size)
    response = generator(sindex=str(random.randint(1, sample_size)))


    # Creation of few_shot_examples using dspy.Example
    few_shot_examples = [
        dspy.Example({
            field_name: completion[field_name] for field_name in properties.keys()
        }) for completion in response.completions
    ]

    return few_shot_examples

# Example usage:

# class SyntheticFacts(BaseModel):
#     fact: str = Field(..., description="a statement")
#     veracity: bool = Field(..., description="an assessment of the veracity of the statement")

# synthetic_examples = synthetic_data_generation(SyntheticFacts, sample_size=10)

# print(synthetic_examples)
