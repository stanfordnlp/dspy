import importlib.util
import json
import os
import pathlib
import random
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import pydantic
from datamodel_code_generator import InputFileType, generate

import dspy
from tests.reliability.utils import assert_program_output_correct, judge_dspy_configuration


def _retry(retries):
    """
    A decorator to retry a function a specified number of times.

    Args:
        retries (int): The number of retries before failing.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    print(f"Retrying {func.__name__} (attempt {attempt} of {retries})." f" Exception: {e}")
                    if attempt >= retries:
                        raise e

        return wrapper

    return decorator


@_retry(retries=5)
def generate_test_program(dst_path: str, additional_instructions: Optional[str] = None) -> dspy.Module:
    """
    Generate a DSPy program for a reliability test case and save it to a destination path.

    Args:
        dst_path: The directory path to which to save the generated program.
        additional_instructions: Additional instructions for generating the program signature.
    Return:
        A dspy.Module object representing the generated program.
    """

    def generate_models(schema: dict[str, Any], class_name: str) -> str:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_schema_path = os.path.join(tmp_dir, "schema.json")
            tmp_model_path = os.path.join(tmp_dir, "model.py")
            with open(tmp_schema_path, "w") as f:
                json.dump(schema, f)

            generate(
                input_=pathlib.Path(tmp_schema_path),
                input_file_type=InputFileType.JsonSchema,
                output=pathlib.Path(tmp_model_path),
                class_name=class_name,
                # For enums with only one value, use the value as a literal instead of an enum
                # in order to test literals
                enum_field_as_literal="one",
                # Don't use con* field types, which are deprecated in recent pydantic versions
                field_constraints=True,
                use_annotated=False,
            )
            # Remove annotation imports from __future__, which break compatibility with Python's
            # built-in type hints
            _remove_line_from_file(tmp_model_path, "from __future__ import annotations")
            # Remove comments inserted by datamodel-code-generator from the generated model file
            _remove_comments_from_file(tmp_model_path)
            with open(tmp_model_path, "r") as f:
                return f.read()

    def rename_conflicting_fields(
        input_schema: dict[str, Any],
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        input_fields = set(input_schema.get("properties", {}))
        output_schema["properties"] = {
            (f"{field}_output" if field in input_fields else field): properties
            for field, properties in output_schema.get("properties", {}).items()
        }
        # Update required fields, if they exist
        if "required" in output_schema:
            output_schema["required"] = [
                f"{field}_output" if field in input_fields else field for field in output_schema["required"]
            ]
        return output_schema

    # Disable caching and use a nonzero temperature to ensure that new programs are generated
    # upon retry if there's an error in the generation process (e.g. the program has an
    # invalid signature)
    with judge_dspy_configuration(cache=False, temperature=0.5), tempfile.TemporaryDirectory() as tmp_dir:
        generated_signature = _get_test_program_generation_program()(
            additional_instructions=additional_instructions or ""
        )
        input_schema = json.loads(generated_signature.program_input_fields)
        output_schema = json.loads(generated_signature.program_output_fields)
        # If there are conflicting field names between input and output schemas, rename the output
        # fields to avoid conflicts
        output_schema = rename_conflicting_fields(input_schema, output_schema)

        # Generate input and output models
        input_models = generate_models(schema=input_schema, class_name="ProgramInputs")
        output_models = generate_models(schema=output_schema, class_name="ProgramOutputs")

        # Write program code
        program_code = (
            "### Input models ###\n"
            + input_models
            + "\n"
            + "### Output models ###\n"
            + output_models
            + "\n"
            + "### Program definition ###\n"
            + _get_test_program_signature_and_module_definition(
                program_description=generated_signature.program_description
            )
        )
        program_path = os.path.join(tmp_dir, "program.py")
        with open(program_path, "w") as f:
            f.write(program_code)

        # Validate the generated program by loading it before copying it to the destination path
        loaded_program, _ = load_generated_program(program_path)

        # Write schema
        _write_pretty_json(
            data=_clean_schema(_get_json_schema(loaded_program.signature)),
            path=os.path.join(tmp_dir, "schema.json"),
        )

        # Copy all generated files to the destination path
        os.makedirs(dst_path, exist_ok=True)
        shutil.copytree(tmp_dir, dst_path, dirs_exist_ok=True)

        return loaded_program


@_retry(retries=5)
def generate_test_inputs(
    dst_path: str,
    program_path: str,
    num_inputs: int,
    additional_instructions: Optional[str] = None,
):
    """
    Generate test inputs for a reliability test case and save them to a destination path.

    Args:
        dst_path: The directory path to which to save the generated test inputs.
        program_path: The path to the program for which to generate test inputs.
        num_inputs: The number of test inputs to generate.
        additional_instructions: Additional instructions for generating the test inputs.
    """
    # Disable caching and use a nonzero temperature to ensure that new inputs are generated
    # upon retry if there's an error in the generation process (e.g. the input doesn't match the
    # program signature)
    with judge_dspy_configuration(cache=False, temperature=0.5), tempfile.TemporaryDirectory() as tmp_dir:
        program: dspy.Module
        program_input_schema: pydantic.BaseModel
        program, program_input_schema = load_generated_program(program_path)
        signature_json_schema = _get_json_schema(program.signature)
        inputs, outputs = _split_schema(signature_json_schema)
        generated_test_inputs = _get_test_inputs_generation_program()(
            program_description=program.signature.__doc__ or "",
            program_input_signature=_write_pretty_json({"properties": _clean_schema(inputs)}),
            program_output_signature=_write_pretty_json({"properties": _clean_schema(outputs)}),
            additional_instructions=additional_instructions or "",
            num_inputs=num_inputs,
        ).test_inputs[:num_inputs]

        def find_max_input_number(directory):
            if not os.path.exists(directory):
                return 0

            max_number = 0
            pattern = re.compile(r"input(\d+)\.json")

            for filename in os.listdir(directory):
                match = pattern.match(filename)
                if match:
                    number = int(match.group(1))
                    max_number = max(max_number, number)
            return max_number

        base_input_number = find_max_input_number(dst_path) + 1
        for idx, test_input in enumerate(generated_test_inputs):
            output_assertions = _get_assertions_generation_program()(
                program_description=program.signature.__doc__ or "",
                program_input=test_input.program_input,
                program_output_signature=_write_pretty_json({"properties": _clean_schema(outputs)}),
            ).output_assertions

            # Verify that the generated input is valid JSON and matches the input signature of the
            # program before saving it to the destination path
            _json_input_to_program_input(
                input_schema=program_input_schema,
                json_input=test_input.program_input,
            )

            test_input_file_path = os.path.join(tmp_dir, f"input{base_input_number + idx}.json")
            json_program_input = json.loads(test_input.program_input)
            _write_pretty_json(
                data={
                    "input": json_program_input,
                    "assertions": output_assertions,
                },
                path=test_input_file_path,
            )

        os.makedirs(dst_path, exist_ok=True)
        shutil.copytree(tmp_dir, dst_path, dirs_exist_ok=True)


def load_generated_program(path) -> Tuple[dspy.Module, pydantic.BaseModel]:
    """
    Loads a generated program from the specified file.

    Args:
        path: The path to the file containing the generated program.
    Returns:
        A tuple containing: 1. a dspy.Module object representing the generated program
        and 2. a pydantic.BaseModel object representing the program's input schema.
    """
    if os.path.isdir(path):
        path = os.path.join(path, "program.py")
    if not os.path.exists(path):
        raise ValueError(f"DSPy test program file not found: {path}")

    program_module = _import_program_module_from_path(module_name="program", file_path=path)
    return program_module.program, program_module.ProgramInputs


@dataclass
class GeneratedTestCase:
    """
    Represents a DSPy reliability test case that has been generated with the help of a
    DSPy program generator and program input generator.
    """

    # The name of the test case for identification / debugging with pytest
    name: str
    # The local filesystem path to the program that the test case is testing.
    program_path: str
    # A JSON  representation of the input to the program that the test case is testing.
    program_input: str
    # The assertions that the output of the program must satisfy for the test case to pass.
    output_assertions: list[str]


def load_generated_cases(dir_path) -> list[GeneratedTestCase]:
    """
    Recursively loads generated test cases from the specified directory and its subdirectories.

    Args:
        dir_path: The path to the directory containing the generated test cases.
    Returns:
        A list of GeneratedTestCase objects.
    """
    test_cases = []

    # Walk through all directories and subdirectories in dir_path
    for root, dirs, files in os.walk(dir_path):
        # Check if the directory contains a program.py and an inputs directory
        if "program.py" in files and "inputs" in dirs:
            program_path = os.path.join(root, "program.py")
            inputs_path = os.path.join(root, "inputs")

            # Load each JSON test input file in the inputs directory
            for input_file in os.listdir(inputs_path):
                if input_file.endswith(".json"):
                    with open(os.path.join(inputs_path, input_file), "r") as f:
                        # Best effort to extract a meaningful enclosing directory name
                        # from the test path that can be used as part of the test case name
                        readable_dir_name = os.path.basename(os.path.dirname(os.path.dirname(root)))
                        test_case_name = (
                            f"{readable_dir_name}-" f"{os.path.basename(root)}-" f"{os.path.splitext(input_file)[0]}"
                        )
                        program_input_and_assertions = json.load(f)
                        program_input = program_input_and_assertions["input"]
                        assertions = program_input_and_assertions["assertions"]

                        # Create a GeneratedTestCase object and add it to the list
                        test_cases.append(
                            GeneratedTestCase(
                                name=test_case_name,
                                program_path=program_path,
                                program_input=json.dumps(program_input),
                                output_assertions=assertions,
                            )
                        )

    return test_cases


def run_generated_case(generated_case: GeneratedTestCase):
    """
    Runs a generated reliability test case by 1. running the test case program on the test case
    input using the global DSPy configuration and 2. verifying that the output of the program
    satisfies the assertions specified in the test case.

    Args:
        generated_case: The generated test case to run.
    """
    program, program_input_schema = load_generated_program(generated_case.program_path)
    program_input = _json_input_to_program_input(
        input_schema=program_input_schema,
        json_input=generated_case.program_input,
    )
    program_output = program(**program_input)
    for assertion in generated_case.output_assertions:
        assert_program_output_correct(
            program_input=program_input,
            program_output=program_output,
            grading_guidelines=assertion,
        )


def _get_test_program_signature_and_module_definition(program_description: str) -> str:
    """
    Generate the signature and model definition for a test DSPy program.

    Args:
        program_description: A description of the generated program.
    """
    use_cot = random.choice([True, False])
    if use_cot:
        program_var_definition = "program = dspy.ChainOfThought(program_signature)"
    else:
        program_var_definition = "program = dspy.Predict(program_signature)"

    return '''
import dspy

class BaseSignature(dspy.Signature):
    """
    {program_description}
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

{program_var_definition}
'''.format(program_description=program_description, program_var_definition=program_var_definition)


def _get_test_program_generation_program() -> dspy.Module:
    """
    Create a DSPy program for generating other DSPy test programs.

    Returns:
        A dspy.Module object representing the program generation program.
    """

    class ProgramGeneration(dspy.Signature):
        """
        Creates an AI program definition, including the AI program's description, input fields, and output fields.
        The AI program should be designed to solve a real problem for its users and produce correct outputs for a variety of inputs.

        The input fields and the output fields must be represented in JSON Schema format, including field names, types, and descriptions.
        The JSON schema definitions themselves MUST be valid JSON without any extra text (no backticks, no explanatory text, etc.).

        It's very important to be sure that the additional instructions, if specified, are obeyed
        precisely in absolutely all cases.
        """

        additional_instructions: str = dspy.InputField(
            description="Additional instructions for what kind of program to generate and how to generate it"
        )
        program_description: str = dspy.OutputField(
            description="A description of the generated AI program, including its purpose and expected behavior"
        )
        program_input_fields: str = dspy.OutputField(
            description="The input fields of the generated program in JSON Schema format, including input field names, types, and descriptions."
        )
        program_output_fields: str = dspy.OutputField(
            description="The output fields of the generated program in JSON Schema format, including input field names, types, and descriptions."
        )

    return dspy.ChainOfThought(ProgramGeneration)


def _get_test_inputs_generation_program() -> dspy.Module:
    """
    Create a DSPy program for generating test inputs for a given DSPy test program.

    Returns:
        A dspy.Module object representing the test input generation program.
    """

    class _TestInputsGeneration(dspy.Signature):
        """
        Given the description and input / output signature (format) of an AI program that is designed to produce correct outputs for a variety
        of inputs while adhering to the input / output signature, generate test inputs used to verify that the program
        indeed produces correct outputs. The AI program uses LLM prompting with carefully crafted prompt templates to generate
        responses.

        When generating an input, do not think about how the program will respond. Instead, focus on creating
        valid and interesting inputs that are likely to test the program's capabilities.

        It's very important to be sure that the additional instructions, if specified, are obeyed
        precisely in absolutely all cases.
        """

        program_description: str = dspy.InputField(
            description="A description of the AI program being tested, including its purpose and expected behavior"
        )
        program_input_signature: str = dspy.InputField(
            description="The input signature of the program in JSON Schema format, including input field names, types, and descriptions. The outermost fields in the JSON schema definition represent the top-level input fields of the program."
        )
        program_output_signature: str = dspy.InputField(
            description="The output signature of the program in JSON Schema format, including output field names, types, and descriptions. The outermost fields in the JSON schema definition represent the top-level output fields of the program."
        )
        additional_instructions: str = dspy.InputField(description="Additional instructions for generating test inputs")
        test_inputs: list[_TestInput] = dspy.OutputField(
            description="Generated test inputs for the program, used to verify the correctness of the program outputs for a variety of inputs"
        )

    return dspy.ChainOfThought(_TestInputsGeneration)


class _TestInput(pydantic.BaseModel):
    """
    Represents a generated test input for a DSPy program.
    """

    program_input: str = pydantic.Field(
        "Generated input matching the program signature that will be used to test the program, represented as a JSON string."
        " The schema of the JSON string must match the input signature of the program precisely, including any wrapper objects."
        " Be very careful to ensure that the input is valid JSON and matches the input signature of the program, with correct"
        " field nesting."
    )


def _get_assertions_generation_program() -> dspy.Module:
    """
    Create a DSPy program for generating assertions that verify the correctness of outputs
    from other DSPy programs.
    """

    class _TestInputsGeneration(dspy.Signature):
        """
        Given 1. the description and input / output signature (format) of an AI program that is designed to produce correct outputs for a variety
        of inputs while adhering to the input / output signature and 2. an example input to the AI program, generate assertions that can be used
        to verify the correctness of the program output.

        Assertions should be expressed in natural language where possible, rather than code. Only
        include code if necessary to clarify the assertion. Assertions should be objective and verifiable,
        with minimal subjectivity only where absolutely necessary.

        There should be a limited number of assertions, ideally about 5, that are sufficient to
        verify the correctness of the program output.

        If it's too difficult to generate accurate assertions, leave them blank.
        """

        program_description: str = dspy.InputField(
            description="A description of the AI program being tested, including its purpose and expected behavior"
        )
        program_input: str = dspy.InputField(
            description="An example input to the AI program, represented as a JSON string"
        )
        program_output_signature: str = dspy.InputField(
            description="The output signature of the program in JSON Schema format, including output field names, types, and descriptions. The outermost fields in the JSON schema definition represent the top-level output fields of the program."
        )
        output_assertions: list[str] = dspy.OutputField(
            description="Assertions used to verify the correctness of the program output after running the program on the specified input"
        )

    return dspy.ChainOfThought(_TestInputsGeneration)


def _clean_json_schema_property(prop: dict[str, Any]) -> dict[str, Any]:
    """
    Remove unnecessary keys from a JSON schema property dictionary, as well as
    all of its child properties.

    Args:
        prop: The JSON schema property dictionary to clean.
    Returns:
        The cleaned JSON schema property dictionary.
    """
    cleaned_prop = {
        k: v for k, v in prop.items() if k not in {"desc", "__dspy_field_type", "title", "prefix", "required"}
    }

    # Recursively clean nested properties
    if "properties" in cleaned_prop:
        cleaned_prop["properties"] = {k: _clean_json_schema_property(v) for k, v in cleaned_prop["properties"].items()}

    return cleaned_prop


def _get_json_schema(signature: dspy.Signature) -> dict[str, Any]:
    """
    Obtain the JSON schema representation of a DSPy signature.

    Args:
        signature: The DSPy signature for which to generate a JSON schema.
    Returns:
        A JSON schema representation of the signature.
    """

    def expand_refs(schema: dict[str, Any], definitions: dict[str, Any]) -> dict[str, Any]:
        """
        Expand $ref fields in a JSON schema, inlining the referenced schema definitions
        directly into the $ref field locations.
        """
        if isinstance(schema, dict):
            if "$ref" in schema:
                ref_path = schema["$ref"].replace("#/$defs/", "")
                ref_schema = definitions.get(ref_path, {})
                if "__dspy_field_type" in schema:
                    ref_schema["__dspy_field_type"] = schema["__dspy_field_type"]
                # Recursively expand the reference schema as well
                return expand_refs(ref_schema, definitions)
            else:
                # Recursively expand properties in the schema
                return {key: expand_refs(value, definitions) for key, value in schema.items()}
        elif isinstance(schema, list):
            return [expand_refs(item, definitions) for item in schema]
        return schema

    signature_schema_with_refs = signature.schema()
    definitions = signature_schema_with_refs.pop("$defs", {})
    return expand_refs(signature_schema_with_refs, definitions)


def _split_schema(schema: dict[str, Any]) -> Tuple[dict[str, Any], dict[str, Any]]:
    """
    Split a JSON schema into input and output components based on DSPy field types.

    Args:
        schema: The JSON schema to split.
    Returns:
        A tuple containing the input and output components of the schema.
    """
    inputs = {}
    outputs = {}

    # Traverse the properties to categorize inputs and outputs
    for key, prop in schema.get("properties", {}).items():
        # Clean the property
        cleaned_prop = _clean_schema(prop)

        # Determine if the property is input or output based on __dspy_field_type
        field_type = prop.get("__dspy_field_type")
        if field_type == "input":
            inputs[key] = cleaned_prop
        elif field_type == "output" or field_type is None:
            outputs[key] = cleaned_prop

        # Handle nested properties for complex models
        if "properties" in prop:
            nested_inputs, nested_outputs = _split_schema(prop)
            if nested_inputs and field_type == "input":
                inputs[key] = {"properties": nested_inputs, **cleaned_prop}
            elif nested_outputs and (field_type == "output" or field_type is None):
                outputs[key] = {"properties": nested_outputs, **cleaned_prop}

    return inputs, outputs


def _clean_schema(prop: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively clean a JSON schema property by removing unnecessary keys.

    Args:
        prop: The JSON schema property to clean.
    Returns:
        A cleaned version of the property.
    """
    keys_to_remove = ["__dspy_field_type", "title"]  # Add any other keys to be removed here

    # Iterate through the dictionary, applying cleaning recursively if value is a nested dict
    cleaned_prop = {
        k: (_clean_schema(v) if isinstance(v, dict) else v)  # Recurse if value is a dict
        for k, v in prop.items()
        if k not in keys_to_remove
    }
    return cleaned_prop


def _json_input_to_program_input(input_schema: pydantic.BaseModel, json_input: str) -> dict[str, Any]:
    """
    Convert a JSON input string to a DSPy program input dictionary, validating it against the
    provided program signature.

    Args:
        input_schema: A pydantic model representing the program input schema.
        json_input: The JSON input string to convert to a DSPy program input.
    Returns:
        The converted DSPy program input dictionary.
    """
    json_input = json.loads(json_input)
    program_input: pydantic.BaseModel = input_schema.model_validate(json_input)
    return {field: getattr(program_input, field) for field in program_input.__fields__}


@contextmanager
def _temporarily_prepend_to_system_path(path):
    """
    Temporarily prepend a path to the system path for the duration of a context.

    Args:
        path: The path to prepend to the system path.
    """
    original_sys_path = sys.path.copy()
    try:
        sys.path.insert(0, path)
        yield
    finally:
        sys.path = original_sys_path


def _import_program_module_from_path(module_name: str, file_path: str):
    """
    Import a Python module containing a DSPy program from a specified file path.

    Args:
        module_name: The name of the module containing the DSPy program to import.
        file_path: The path to the file containing the module definition.
    """
    program_dir = os.path.dirname(file_path)

    with _temporarily_prepend_to_system_path(program_dir):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _remove_line_from_file(file_path: str, line_to_remove: str):
    """
    Remove all instances of a specific line from a file.

    Args:
        file_path: The path to the file from which to remove all instances of the line.
        line_to_remove: The line to remove from the file.
    """
    # Read all lines from the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Write all lines back except the one to remove
    with open(file_path, "w") as file:
        for line in lines:
            if line.strip() != line_to_remove:
                file.write(line)


def _remove_comments_from_file(file_path: str) -> None:
    """
    Removes all lines with comments from the specified file.

    Args:
        file_path: Path to the file where comments should be removed.
    """
    # Read the file contents
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Filter out lines that start with '#'
    cleaned_lines = [line for line in lines if not line.strip().startswith("#")]

    # Write the cleaned lines back to the file
    with open(file_path, "w") as file:
        file.writelines(cleaned_lines)


def _write_pretty_json(data: dict[str, Any], path: Optional[str] = None) -> Optional[str]:
    """
    Format JSON data with indentation, and write it to a file if specified.

    Args:
        data: The JSON data to format.
        path: The optional path to which to write the formatted JSON data.
    Returns:
        The formatted JSON data as a string, if no path is specified.
    """
    formatted_json = json.dumps(data, indent=4)
    if path:
        with open(path, "w") as f:
            f.write(formatted_json)
        return None
    else:
        return formatted_json
