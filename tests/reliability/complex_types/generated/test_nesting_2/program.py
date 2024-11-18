### Input models ###


from datetime import datetime

from pydantic import BaseModel, Field


class Details(BaseModel):
    value: str = Field(..., description="Customer's value category")
    age: int = Field(..., description="Customer's age")


class Customer(BaseModel):
    customer_id: str = Field(..., description="Unique identifier for the customer")
    customer_type: bool = Field(..., description="Indicates if the customer is a premium member")
    details: Details


class Details1(BaseModel):
    value: float = Field(..., description="Monetary value of the transaction")
    timestamp: datetime = Field(..., description="Timestamp of the transaction")


class Transaction(BaseModel):
    transaction_id: str = Field(..., description="Unique identifier for the transaction")
    amount: float = Field(..., description="Transaction amount")
    details: Details1


class ProgramInputs(BaseModel):
    customer: Customer
    transaction: Transaction


### Output models ###


from datetime import datetime

from pydantic import BaseModel, Field


class CustomerType(BaseModel):
    is_premium: bool = Field(..., description="Indicates if the customer is a premium member")
    category: str = Field(..., description="Customer's membership category")


class CustomerSummary(BaseModel):
    customer_id: str = Field(..., description="Unique identifier for the customer")
    customer_type: CustomerType
    value: str = Field(..., description="Customer's value category")


class Details(BaseModel):
    value: float = Field(..., description="Monetary value of the transaction")
    timestamp: datetime = Field(..., description="Timestamp of the transaction")


class TransactionSummary(BaseModel):
    transaction_id: str = Field(..., description="Unique identifier for the transaction")
    total_amount: float = Field(..., description="Total transaction amount")
    details: Details


class ProgramOutputs(BaseModel):
    customer_summary: CustomerSummary
    transaction_summary: TransactionSummary


### Program definition ###

import dspy


class BaseSignature(dspy.Signature):
    """
    This AI program is designed to process complex datasets with multiple nested input fields and produce structured output fields. It can handle cases where nested fields have the same name but different types, ensuring that the data is accurately processed and transformed. The program is particularly useful for applications that require detailed data analysis, integration of multiple data sources, and handling of heterogeneous data types.
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

program = dspy.ChainOfThought(program_signature)
