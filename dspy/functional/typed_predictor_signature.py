from typing import Annotated, Type
from typing import get_origin, get_args

from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from pydantic.fields import FieldInfo

import dspy
from dspy import InputField, OutputField, Signature


class TypedPredictorSignature:
    @classmethod
    def create(
            cls,
            pydantic_class_for_dspy_input_fields: Type[BaseModel],
            pydantic_class_for_dspy_output_fields: Type[BaseModel],
            prefix_instructions: str = "") -> Type[Signature]:
        """
        Return a DSPy Signature class that can be used to extract the output parameters.

        :param pydantic_class_for_dspy_input_fields: Pydantic class that defines the DSPy InputField's.
        :param pydantic_class_for_dspy_output_fields: Pydantic class that defines the DSPy OutputField's.
        :param prefix_instructions: Optional text that is prefixed to the instructions.
        :return: A DSPy Signature class optimizedfor use with a TypedPredictor to extract structured information.
        """
        if prefix_instructions:
            prefix_instructions += "\n\n"
        instructions = prefix_instructions + "Use only the available information to extract the output fields.\n\n"
        dspy_fields = {}
        for field_name, field in pydantic_class_for_dspy_input_fields.model_fields.items():
            if field.default and 'typing.Annotated' in str(field.default):
                raise ValueError(f"Field '{field_name}' is annotated incorrectly. See 'Constraints on compound types' in https://docs.pydantic.dev/latest/concepts/fields/")

            is_default_value_specified, is_marked_as_optional, inner_field = cls._process_field(field)
            if is_marked_as_optional:
                if not is_default_value_specified or field.default is None:
                    field.default = 'null'
                field.description = inner_field.description
                field.examples = inner_field.examples
                field.metadata = inner_field.metadata
                field.json_schema_extra = inner_field.json_schema_extra
            else:
                field.validate_default = False

            input_field = InputField(desc=field.description)
            dspy_fields[field_name] = (field.annotation, input_field)

        for field_name, field in pydantic_class_for_dspy_output_fields.model_fields.items():
            if field.default and 'typing.Annotated' in str(field.default):
                raise ValueError(f"Field '{field_name}' is annotated incorrectly. See 'Constraints on compound types' in https://docs.pydantic.dev/latest/concepts/fields/")

            is_default_value_specified, is_marked_as_optional, inner_field = cls._process_field(field)
            if is_marked_as_optional:
                if not is_default_value_specified or field.default is None:
                    field.default = 'null'
                field.description = inner_field.description
                field.examples = inner_field.examples
                field.metadata = inner_field.metadata
                field.json_schema_extra = inner_field.json_schema_extra
            else:   
                field.validate_default = False

            if field.default is PydanticUndefined:
                raise ValueError(
                    f"Field '{field_name}' has no default value. Required fields must have a default value. "
                    "Change the field to be Optional or specify a default value."
                )
            
            output_field = OutputField(desc=field.description if field.description else "")
            dspy_fields[field_name] = (field.annotation, output_field)

            instructions += f"When extracting '{field_name}':\n"
            instructions += f"If it is not mentioned in the input fields, return: '{field.default}'. "

            examples = field.examples
            if examples:
                quoted_examples = [f"'{example}'" for example in examples]
                instructions += f"Example values are: {', '.join(quoted_examples)} etc. "

            if field.metadata:
                constraints = [meta for meta in field.metadata if 'Validator' not in str(meta)]
                if field.json_schema_extra and 'invalid_value' in field.json_schema_extra:
                    instructions += f"If the extracted value does not conform to: {constraints}, return: '{field.json_schema_extra['invalid_value']}'."
                else:
                    print(f"WARNING - Field: '{field_name}' is missing an 'invalid_value' attribute. Fields with value constraints should specify an 'invalid_value'.")
                    instructions += f"If the extracted value does not conform to: {constraints}, return: '{field.default}'."

            instructions += '\n\n'

        return dspy.Signature(dspy_fields, instructions.strip())

    @classmethod
    def _process_field(cls, field: FieldInfo) -> tuple[bool, bool, FieldInfo]:
        is_default_value_specified = not field.is_required()
        is_marked_as_optional, inner_type, field_info = cls._analyze_field_annotation(field.annotation)
        if field_info:
            field_info.annotation = inner_type
            return is_default_value_specified, is_marked_as_optional, field_info

        return is_default_value_specified, is_marked_as_optional, field

    @classmethod
    def _analyze_field_annotation(cls, annotation):
        is_optional = False
        inner_type = annotation
        field_info = None

        # Check if it's Optional
        if hasattr(annotation, '_name') and annotation._name == 'Optional':
            is_optional = True
            inner_type = get_args(annotation)[0]

        # Check if it's Annotated
        if get_origin(inner_type) is Annotated:
            args = get_args(inner_type)
            inner_type = args[0]
            for arg in args[1:]:
                if isinstance(arg, FieldInfo):
                    field_info = arg
                    break

        return is_optional, inner_type, field_info
