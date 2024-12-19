### Input models ###


from datetime import datetime
from enum import Enum
from typing import List, Tuple

from pydantic import BaseModel, Field


class EnumField(Enum):
    option1 = "option1"
    option2 = "option2"
    option3 = "option3"


class LiteralField(Enum):
    literalValue = "literalValue"


class ObjectField(BaseModel):
    subField1: str
    subField2: float


class NestedObjectField(BaseModel):
    tupleField: Tuple[str, float]
    enumField: EnumField
    datetimeField: datetime
    literalField: LiteralField


class ProgramInputs(BaseModel):
    tupleField: Tuple[str, float]
    enumField: EnumField
    datetimeField: datetime
    literalField: LiteralField
    objectField: ObjectField
    nestedObjectField: NestedObjectField


### Output models ###


from datetime import datetime
from enum import Enum
from typing import List, Tuple, Union

from pydantic import BaseModel, Field


class ProcessedEnumField(Enum):
    option1 = "option1"
    option2 = "option2"
    option3 = "option3"


class ProcessedLiteralField(Enum):
    literalValue = "literalValue"


class ProcessedObjectField(BaseModel):
    subField1: str
    subField2: float
    additionalField: bool


class EnumField(Enum):
    option1 = "option1"
    option2 = "option2"
    option3 = "option3"


class LiteralField(Enum):
    literalValue = "literalValue"


class ProcessedNestedObjectField(BaseModel):
    tupleField: Tuple[str, float]
    enumField: EnumField
    datetimeField: datetime
    literalField: LiteralField
    additionalField: bool


class ProgramOutputs(BaseModel):
    processedTupleField: Tuple[str, float]
    processedEnumField: ProcessedEnumField
    processedDatetimeField: datetime
    processedLiteralField: ProcessedLiteralField
    processedObjectField: ProcessedObjectField
    processedNestedObjectField: ProcessedNestedObjectField


### Program definition ###

import dspy


class BaseSignature(dspy.Signature):
    """
    The program is designed to process various data types including tuples, enums, datetime values, literals, objects, and nested objects containing these types. The program will accept inputs of these types, perform specified operations on them, and return the results. The operations could include validation, transformation, and extraction of information from these inputs.
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
