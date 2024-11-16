### Input models ###


from pydantic import BaseModel, Field


class ProgramInputs(BaseModel):
    markdown_content: str = Field(
        ...,
        description="The content of the markdown document from which the table of contents will be generated.",
    )


### Output models ###


from pydantic import BaseModel, Field


class ProgramOutputs(BaseModel):
    table_of_contents: str = Field(..., description="The generated table of contents in markdown format.")


### Program definition ###

import dspy


class BaseSignature(dspy.Signature):
    """
    The program is designed to generate a table of contents (TOC) from a given markdown document. It will parse the markdown content, identify headings, and create a hierarchical TOC based on the heading levels. The TOC will be presented in markdown format, with each entry linked to the corresponding section in the document.
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
