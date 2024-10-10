import json
import re
import dspy
import inspect
try:
    from IPython.core.magics.code import extract_symbols
except ImportError:
    # Won't be able to read code from juptyer notebooks
    extract_symbols = None

from dspy.predict.parameter import Parameter

from dspy.teleprompt.utils import get_signature

def strip_prefix(text):
    pattern = r'^[\*\s]*(([\w\'\-]+\s+){0,4}[\w\'\-]+):\s*'
    modified_text = re.sub(pattern, '', text)
    return modified_text.strip("\"")

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
    top_n_program_history = sorted(unique_program_history, key=lambda x: x['score'], reverse=True)[:top_n]
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
        instruction_list.append(f"\"{pred_instructions}\"")
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
            instruction_aggregate[instruction]['total_score'] += score
            instruction_aggregate[instruction]['count'] += 1
        else:
            instruction_aggregate[instruction] = {'total_score': score, 'count': 1}
    
    # Calculate average score for each instruction and prepare for sorting
    predictor_history = []
    for instruction, data in instruction_aggregate.items():
        average_score = data['total_score'] / data['count']
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
    return '\n'.join(output)

def get_dspy_source_code(module):
    header = []
    base_code = ""

    # Don't get source code for Predict or ChainOfThought modules (NOTE we will need to extend this list as more DSPy.modules are added)
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
            if item in completed_set:
                continue
            if isinstance(item, Parameter):
                if hasattr(item, 'signature') and item.signature is not None and item.signature.__pydantic_parent_namespace__['signature_name'] + "_sig" not in completed_set:
                    try:
                        header.append(inspect.getsource(item.signature))
                        print(inspect.getsource(item.signature))
                    except (TypeError, OSError):
                        header.append(str(item.signature))
                    completed_set.add(item.signature.__pydantic_parent_namespace__['signature_name'] + "_sig")
            if isinstance(item, dspy.Module):
                code = get_dspy_source_code(item).strip()
                if code not in completed_set:
                    header.append(code)
                    completed_set.add(code)
            completed_set.add(item)
        
    return '\n\n'.join(header) + '\n\n' + base_code