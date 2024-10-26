import pytest
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import Annotated, Optional
from dspy.functional.typed_predictor_signature import TypedPredictorSignature


class PydanticInput(BaseModel):
    command: str


class PydanticOutput1(BaseModel):
    @field_validator("name", mode="wrap")
    @staticmethod
    def validate_name(name, handler):
        try:
            return handler(name)
        except ValidationError:
            return 'INVALID_NAME'

    name: Annotated[str,
                    Field(default='NOT_FOUND', max_length=15,
                          title='Name', description='The name of the person',
                          examples=['John Doe', 'Jane Doe'],
                          json_schema_extra={'invalid_value': 'INVALID_NAME'}
                         )
                   ]

class PydanticOutput2(BaseModel):
    @field_validator("age", mode="wrap")
    @staticmethod
    def validate_age(age, handler):
        try:
            return handler(age)
        except ValidationError:
            return -8888

    age: Annotated[int, 
                   Field(gt=0, lt=150, default=-999, 
                         json_schema_extra={'invalid_value': '-8888'}
                        )
                  ]

class PydanticOutput3(BaseModel):
    age: Annotated[int, 
                   Field(gt=0, lt=150,
                         json_schema_extra={'invalid_value': '-8888'}
                         )
                  ] = -999

class PydanticOutput4(BaseModel):
    age: Optional[Annotated[int, 
                   Field(gt=0, lt=150,
                         json_schema_extra={'invalid_value': '-8888'}
                         )]]

class PydanticOutput5(BaseModel):
    @field_validator("email", mode="wrap")
    @staticmethod
    def validate_email(email, handler):
        try:
            return handler(email)
        except ValidationError:
            return 'INVALID_EMAIL'

    email: Annotated[str, 
                     Field(default='NOT_FOUND', 
                           pattern=r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$',
                           json_schema_extra={'invalid_value': 'INVALID_EMAIL'}
                          )
                    ]

@pytest.mark.parametrize("pydantic_output_class", [
    # PydanticOutput1,
    # PydanticOutput2,
    # PydanticOutput3,
    PydanticOutput4,
    # PydanticOutput5
])
def test_valid_pydantic_types(pydantic_output_class: str):
    dspy_signature_class = TypedPredictorSignature.create(PydanticInput, pydantic_output_class)
    assert dspy_signature_class is not None

