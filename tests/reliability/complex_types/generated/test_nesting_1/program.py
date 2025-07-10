### Input models ###


from pydantic import BaseModel, Field


class Level5(BaseModel):
    field1: str = Field(..., description="A string field at the deepest level")
    field2: float = Field(..., description="A numerical field at the deepest level")


class Level4(BaseModel):
    level5: Level5


class Level3(BaseModel):
    level4: Level4


class Level2(BaseModel):
    level3: Level3


class Level1(BaseModel):
    level2: Level2


class ProgramInputs(BaseModel):
    level1: Level1


### Output models ###


from typing import List

from pydantic import BaseModel, Field


class ResultLevel5(BaseModel):
    outputField1: bool = Field(..., description="A boolean field indicating success or failure")
    outputField2: list[str] = Field(..., description="An array of strings representing messages")


class ResultLevel4(BaseModel):
    resultLevel5: ResultLevel5


class ResultLevel3(BaseModel):
    resultLevel4: ResultLevel4


class ResultLevel2(BaseModel):
    resultLevel3: ResultLevel3


class ResultLevel1(BaseModel):
    resultLevel2: ResultLevel2


class ProgramOutputs(BaseModel):
    resultLevel1: ResultLevel1


### Program definition ###

import dspy


class BaseSignature(dspy.Signature):
    """
    The AI program is designed to process hierarchical data structures with multiple levels of nesting. The program will take a deeply nested input structure representing a complex dataset, perform specific transformations, validations, and computations, and then produce an equally complex nested output structure. The program is suitable for applications that require detailed data processing, such as multi-level data aggregation, hierarchical data validation, and nested data transformation.
    """


program_signature = BaseSignature
for input_field_name, input_field in ProgramInputs.model_fields.items():
    program_signature = program_signature.append(
        name=input_field_name,
        field=dspy.InputField(description=input_field.description),
        type_=input_field.annotation,
    )
for output_field_name, output_field in ProgramOutputs.model_fields.items():
    program_signature = program_signature.append(
        name=output_field_name,
        field=dspy.OutputField(description=input_field.description),
        type_=output_field.annotation,
    )

program = dspy.Predict(program_signature)
