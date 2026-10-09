import ast
import inspect
import json
import re
import textwrap
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import get_args

from pydantic import BaseModel

import dspy

try:
    from IPython.core.magics.code import extract_symbols
except ImportError:
    # Won't be able to read code from jupyter notebooks
    extract_symbols = None

from dspy.predict.parameter import Parameter
from dspy.teleprompt.utils import get_signature, new_getfile


def strip_prefix(text):
    pattern = r"^[\*\s]*(([\w\'\-]+\s+){0,4}[\w\'\-]+):\s*"
    modified_text = re.sub(pattern, "", text)
    return modified_text.strip('"')

def create_instruction_set_history_string(base_program, trial_logs, top_n):
    program_history = []
    for trial_num in trial_logs:
        trial = trial_logs[trial_num]
        if "program_path" in trial:
            trial_program = base_program.deepcopy()
            trial_program.load(trial["program_path"])
            program_history.append({
                "program": trial_program,
                "score": trial["score"],
            })

    # Deduplicate program history based on the program's instruction set
    seen_programs = set()
    unique_program_history = []
    for entry in program_history:
        program = entry["program"]
        instruction_set = get_program_instruction_set_string(program)
        if instruction_set not in seen_programs:
            seen_programs.add(instruction_set)
            unique_program_history.append(entry)

    # Get the top n programs from program history
    top_n_program_history = sorted(unique_program_history, key=lambda x: x["score"], reverse=True)[:top_n]
    top_n_program_history.reverse()

    # Create formatted string
    instruction_set_history_string = ""
    for entry in top_n_program_history:
        program = entry["program"]
        score = entry["score"]
        instruction_set = get_program_instruction_set_string(program)
        instruction_set_history_string += instruction_set + f" | Score: {score}\n\n"

    return instruction_set_history_string

def parse_list_of_instructions(instruction_string):
    # Try to convert the string representation of a list to an actual list using JSON
    try:
        instructions = json.loads(instruction_string)
        return instructions
    except json.JSONDecodeError:
        pass

    # If JSON decoding fails, extract strings within quotes
    instructions = re.findall(r'"([^"]*)"', instruction_string)
    return instructions

def get_program_instruction_set_string(program):
    instruction_list = []
    for _, pred in enumerate(program.predictors()):
        pred_instructions = get_signature(pred).instructions
        instruction_list.append(f'"{pred_instructions}"')
    # Joining the list into a single string that looks like a list
    return f"[{', '.join(instruction_list)}]"

def create_predictor_level_history_string(base_program, predictor_i, trial_logs, top_n):
    instruction_aggregate = {}
    instruction_history = []

    # Load trial programs
    for trial_num in trial_logs:
        trial = trial_logs[trial_num]
        if "program_path" in trial:
            trial_program = base_program.deepcopy()
            trial_program.load(trial["program_path"])
            instruction_history.append({
                "program": trial_program,
                "score": trial["score"],
            })

    # Aggregate scores for each instruction
    for history_item in instruction_history:
        predictor = history_item["program"].predictors()[predictor_i]
        instruction = get_signature(predictor).instructions
        score = history_item["score"]

        if instruction in instruction_aggregate:
            instruction_aggregate[instruction]["total_score"] += score
            instruction_aggregate[instruction]["count"] += 1
        else:
            instruction_aggregate[instruction] = {"total_score": score, "count": 1}

    # Calculate average score for each instruction and prepare for sorting
    predictor_history = []
    for instruction, data in instruction_aggregate.items():
        average_score = data["total_score"] / data["count"]
        predictor_history.append((instruction, average_score))

    # Deduplicate and sort by average score, then select top N
    seen_instructions = set()
    unique_predictor_history = []
    for instruction, score in predictor_history:
        if instruction not in seen_instructions:
            seen_instructions.add(instruction)
            unique_predictor_history.append((instruction, score))

    top_instructions = sorted(unique_predictor_history, key=lambda x: x[1], reverse=True)[:top_n]
    top_instructions.reverse()

    # Create formatted history string
    predictor_history_string = ""
    for instruction, score in top_instructions:
        predictor_history_string += instruction + f" | Score: {score}\n\n"

    return predictor_history_string

def create_example_string(fields, example):

    # Building the output string
    output = []
    for field_name, field_values in fields.items():
        name = field_values.json_schema_extra["prefix"]

        # Determine the value from input_data or prediction_data
        value = example.get(field_name)

        # Construct the string for the current field
        field_str = f"{name} {value}"
        output.append(field_str)

    # Joining all the field strings
    return "\n".join(output)


def _referenced_annotation_declarations(signature):
    declarations = []
    seen = set()

    def is_user_class(annotation):
        if not isinstance(annotation, type):
            return False
        return annotation.__module__.split(".", 1)[0] not in {
            "builtins",
            "dataclasses",
            "dspy",
            "enum",
            "pydantic",
            "typing",
        }

    def namespace(annotation):
        module = inspect.getmodule(annotation)
        values = dict(vars(module)) if module else {}
        for name, value in (getattr(annotation, "__pydantic_parent_namespace__", None) or {}).items():
            if callable(value) and type(value).__name__ == "_PydanticWeakRef":
                value = value()
            values[name] = value
        return values

    def alias_source(owner, alias_name):
        module = inspect.getmodule(owner)
        if module is None:
            return None
        try:
            source = inspect.getsource(module)
            tree = ast.parse(source)
        except (OSError, SyntaxError, TypeError):
            return None
        for node in tree.body:
            target = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
            elif isinstance(node, ast.AnnAssign):
                target = node.target
            elif type(node).__name__ == "TypeAlias":
                target = node.name
            if isinstance(target, ast.Name) and target.id == alias_name:
                return ast.get_source_segment(source, node)
        return None

    def visit_source_dependencies(annotation):
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(annotation)))
        except (IndentationError, OSError, SyntaxError, TypeError):
            return
        class_node = next((node for node in tree.body if isinstance(node, ast.ClassDef)), None)
        if class_node is None:
            return
        annotation_nodes = list(class_node.bases)
        annotation_nodes.extend(
            node.annotation for node in class_node.body if isinstance(node, ast.AnnAssign)
        )
        values = namespace(annotation)
        for annotation_node in annotation_nodes:
            for node in ast.walk(annotation_node):
                if not isinstance(node, ast.Name) or node.id not in values:
                    continue
                value = values[node.id]
                source = alias_source(annotation, node.id)
                if is_user_class(value):
                    visit(value)
                elif source is not None:
                    visit(value, alias_name=node.id, alias_source_text=source, alias_owner=annotation)

    def visit(annotation, alias_name=None, alias_source_text=None, alias_owner=None):
        if alias_name is not None:
            key = ("alias", alias_name, id(annotation))
            if key in seen:
                return
            seen.add(key)
            values = namespace(alias_owner)
            try:
                tree = ast.parse(alias_source_text)
            except SyntaxError:
                tree = None
            if tree is not None:
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Name) or node.id == alias_name or node.id not in values:
                        continue
                    value = values[node.id]
                    source = alias_source(alias_owner, node.id)
                    if is_user_class(value):
                        visit(value)
                    elif source is not None:
                        visit(value, alias_name=node.id, alias_source_text=source, alias_owner=alias_owner)
            declarations.append((alias_name, annotation, alias_source_text))
            return

        try:
            is_pydantic_model = isinstance(annotation, type) and issubclass(annotation, BaseModel)
        except TypeError:
            is_pydantic_model = False

        is_enum = isinstance(annotation, type) and issubclass(annotation, Enum)
        if is_pydantic_model or is_enum or (is_user_class(annotation) and is_dataclass(annotation)):
            key = ("class", annotation)
            if key in seen or not is_user_class(annotation):
                return
            seen.add(key)
            for base in annotation.__bases__:
                visit(base)
            if is_pydantic_model:
                for field in annotation.model_fields.values():
                    visit(field.annotation)
            else:
                for field_annotation in getattr(annotation, "__annotations__", {}).values():
                    visit(field_annotation)
            visit_source_dependencies(annotation)
            declarations.append(annotation)
            return

        if isinstance(annotation, (list, tuple)):
            for argument in annotation:
                visit(argument)
            return

        alias_value = getattr(annotation, "__value__", None)
        if alias_value is not None and type(annotation).__name__ == "TypeAliasType":
            key = ("alias_value", id(annotation))
            if key in seen:
                return
            seen.add(key)
            visit(alias_value)
            return

        for argument in get_args(annotation):
            visit(argument)

    for field in signature.fields.values():
        visit(field.annotation)

    return declarations


def _pydantic_model_sources(signature):
    sources = []
    for declaration in _referenced_annotation_declarations(signature):
        if isinstance(declaration, tuple):
            name, annotation, source = declaration
            sources.append(source or f"{name} = {inspect.formatannotation(annotation)}")
            continue
        try:
            sources.append(inspect.getsource(declaration))
        except (TypeError, OSError):
            if issubclass(declaration, BaseModel):
                schema = json.dumps(declaration.model_json_schema(), indent=2, sort_keys=True)
                sources.append(f"# JSON Schema for {declaration.__name__}\n{schema}")
            elif issubclass(declaration, Enum):
                members = {name: member.value for name, member in declaration.__members__.items()}
                sources.append(f"{declaration.__name__} = Enum({declaration.__name__!r}, {members!r})")
            elif is_dataclass(declaration):
                field_sources = "\n".join(
                    f"    {field.name}: {inspect.formatannotation(field.type, base_module=declaration.__module__)}"
                    for field in fields(declaration)
                )
                sources.append(f"@dataclass\nclass {declaration.__name__}:\n{field_sources}")
    return sources


def get_dspy_source_code(module):
    header = []
    base_code = ""

    # Don't get source code for Predict or ChainOfThought modules (NOTE we will need to extend this list as more DSPy.modules are added)
    # TODO: if type(module).__name__ not in ["Predict", "ChainOfThought", "ReAct"]:
    if not type(module).__name__ == "Predict" and not type(module).__name__ == "ChainOfThought":
        try:
            base_code = inspect.getsource(type(module))
        except TypeError:
            obj = type(module)
            cell_code = "".join(inspect.linecache.getlines(new_getfile(obj)))
            class_code = extract_symbols(cell_code, obj.__name__)[0][0]
            base_code = str(class_code)

    completed_set = set()
    for attribute in module.__dict__.keys():
        try:
            iterable = iter(getattr(module, attribute))
        except TypeError:
            iterable = [getattr(module, attribute)]

        for item in iterable:
            # Skip items that are unhashable (like module history)
            try:
                hash(item)
            except TypeError:
                continue
            if isinstance(item, Parameter):
                if (
                    hasattr(item, "signature")
                    and item.signature is not None
                    and item.signature.__pydantic_parent_namespace__["signature_name"] + "_sig" not in completed_set
                ):
                    for model_source in _pydantic_model_sources(item.signature):
                        if model_source not in completed_set:
                            header.append(model_source)
                            completed_set.add(model_source)
                    try:
                        header.append(inspect.getsource(item.signature))
                        print(inspect.getsource(item.signature))
                    except (TypeError, OSError):
                        header.append(str(item.signature))
                    completed_set.add(item.signature.__pydantic_parent_namespace__["signature_name"] + "_sig")
            if isinstance(item, dspy.Module):
                code = get_dspy_source_code(item).strip()
                if code not in completed_set:
                    header.append(code)
                    completed_set.add(code)
            completed_set.add(item)

    return "\n\n".join(header) + "\n\n" + base_code
