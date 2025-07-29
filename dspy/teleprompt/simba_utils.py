import dspy
import ujson
import inspect
import logging
import textwrap
import re

from dspy.adapters.chat_adapter import enumerate_fields
from dspy.signatures import InputField, OutputField
from dspy.adapters.types.base_type import Type
from typing import Callable, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

def prepare_models_for_resampling(program: dspy.Module, n: int, teacher_settings: Optional[Dict] = None):

    models = []
    if teacher_settings:
        with dspy.settings.context(trace=[], **teacher_settings):
            lm = dspy.settings.lm
            models.append(lm)

    lm = program.get_lm() or dspy.settings.lm

    # Check to see if our model is a reasoning model, which means temp must stay as 1.0
    model_family = lm.model.split("/")[-1].lower() if "/" in lm.model else lm.model.lower()
    model_pattern = re.match(r"^o([13])(?:-mini)?", model_family)

    if model_pattern: # Vary the seed
        start_seed = 0 if "seed" not in lm.kwargs else lm.kwargs["seed"]
        seeds = [start_seed + 1 + i for i in range(n-len(models))]
        seeds = list(dict.fromkeys(seeds))[:(n-len(models))]
        models.extend([lm.copy(seed=seed) for seed in seeds])
    else: # Vary the temperature
        start_temp = 0 if "temperature" not in lm.kwargs else lm.kwargs["temperature"]
        temps = [start_temp + 0.5 + i * (0.5 / n) for i in range(n-len(models))]
        temps = list(dict.fromkeys(temps))[:(n-len(models))]
        models.extend([lm.copy(temperature=t) for t in temps])
    
    return models

def wrap_program(program: dspy.Module, metric: Callable):
    def wrapped_program(example):
        with dspy.context(trace=[]):
            prediction, trace, score = None, None, 0.0
            try:
                prediction = program(**example.inputs())
            except Exception as e:
                logger.info(e)
            trace = dspy.settings.trace.copy()

        output = None
        score = 0.0
        output_metadata = {}

        try:
            output = metric(example, prediction)
            if isinstance(output, (int, float)):
                score = output
            elif isinstance(output, dspy.Prediction):
                if not hasattr(output, 'score'):
                    raise ValueError("dspy.Prediction must contain a 'score' attribute")
                score = output.score
                # Just extract fields from _store, excluding 'score'
                output_metadata = {
                    k: v for k, v in output._store.items() if k != "score"
                }
        except Exception as e:
            logger.info(e)

        return {
            "prediction": prediction,
            "trace": trace,
            "score": score,
            "example": example,
            "output_metadata": output_metadata
        }

    return wrapped_program

def append_a_demo(demo_input_field_maxlen):
    def append_a_demo_(bucket, system, **kwargs):
        predictor2name, name2predictor = kwargs["predictor2name"], kwargs["name2predictor"]
        batch_10p_score = kwargs["batch_10p_score"]

        logger.info(f"Appending a demo with max length {demo_input_field_maxlen}")
        
        good = bucket[0]
        trace = good["trace"]
        name2demo = {}

        # if good["score"] < batch_10p_score:
        #     logger.info(f"Skipping appending a demo as good score {good['score']} is below the 10th percentile.")
        #     return False
        if good["score"] <= batch_10p_score:
            logger.info(f"Skipping appending a demo as good score {good['score']} is at or below the 10th percentile.")
            return False

        for step in trace:
            predictor, _inputs, _outputs = step

            for k, v in _inputs.items():
                if demo_input_field_maxlen and len(str(v)) > demo_input_field_maxlen:
                    _inputs[k] = f"{str(v)[:demo_input_field_maxlen]}\n\t\t... <TRUNCATED FOR BREVITY>"

            demo = dspy.Example(augmented=True, **_inputs, **_outputs)
            name = predictor2name[id(predictor)]
            name2demo[name] = demo  # keep the last demo for each predictor
        for name, demo in name2demo.items():
            predictor = name2predictor[name]
            predictor.demos.append(demo)

        logger.info(f"Added {len(name2demo)} demos (one each) across all predictors. Each predictor now has {len(predictor.demos)} demos total.")
        return True
    
    return append_a_demo_

def get_good_and_bad_examples(bucket):
    """Get good and bad examples from bucket
    """
    good, bad = bucket[0], bucket[-1]
    return good["example"], bad["example"]

def get_good_and_bad_trajectories(good_example, bad_example, predictor2name):
    """Get good and bad trajectories from examples
    """
    good_trajectory = [
        dict(module_name=predictor2name[id(p)], inputs=i, outputs=dict(o))
        for p, i, o in good_example["trace"]
    ]
    bad_trajectory = [
        dict(module_name=predictor2name[id(p)], inputs=i, outputs=dict(o))
        for p, i, o in bad_example["trace"]
    ]
    return good_trajectory, bad_trajectory

def append_a_rule(bucket, system, **kwargs):
    # Read in kwargs
    predictor2name = kwargs["predictor2name"]
    batch_10p_score, batch_90p_score = kwargs["batch_10p_score"], kwargs["batch_90p_score"]
    prompt_model = kwargs["prompt_model"] or dspy.settings.lm

    module_names = [name for name, _ in system.named_predictors()]
    good, bad = bucket[0], bucket[-1]
    example = good["example"]

    if good["score"] <= batch_10p_score or bad["score"] >= batch_90p_score:
        logger.info(f"Skipping rule generation as good score {good['score']} is at or below the 10th percentile "
                    f"*or* bad score {bad['score']} is at or above the 90th percentile.")
        return False

    if good["score"] <= bad["score"]:
        if good["score"] > batch_90p_score:
            bad["trace"] = []
            bad["score"] = "N/A"
            bad["prediction"] = {"N/A": "Prediction not available"}
        else:
            good["trace"] = []
            good["score"] = "N/A"
            good["prediction"] = {"N/A": "Prediction not available"}

    better_trajectory = [
        dict(module_name=predictor2name[id(p)], inputs=i, outputs=dict(o))
        for p, i, o in good["trace"]
    ]
    worse_trajectory = [
        dict(module_name=predictor2name[id(p)], inputs=i, outputs=dict(o))
        for p, i, o in bad["trace"]
    ]

    kwargs = dict(
        program_code=inspect.getsource(system.__class__),
        modules_defn=inspect_modules(system),
        program_inputs={**example.inputs()},
        oracle_metadata={**example.labels()},
        better_program_trajectory=better_trajectory,
        better_program_outputs=dict(good["prediction"]),
        worse_program_trajectory=worse_trajectory,
        worse_program_outputs=dict(bad["prediction"] or {}),
        worse_reward_value=bad["score"],
        better_reward_value=good["score"],
        worse_reward_info=bad["output_metadata"],
        better_reward_info=good["output_metadata"],
        module_names=module_names,
    )

    kwargs = {k: v if isinstance(v, str) else ujson.dumps(recursive_mask(v), indent=2)
              for k, v in kwargs.items()}

    with dspy.settings.context(trace=[], lm=prompt_model):
        advice_program = dspy.Predict(OfferFeedback)
        advice = advice_program(**kwargs).module_advice

    for name, predictor in system.named_predictors():
        if name in advice:
            logger.info(f"Advice for {name}: {advice[name]}")
            instructions = predictor.signature.instructions + "\n\n" + advice[name]
            predictor.signature = predictor.signature.with_instructions(instructions)

    return True

def update_fields(bucket, system, **kwargs):
    predictor2name = kwargs["predictor2name"]
    batch_10p_score, batch_90p_score = kwargs["batch_10p_score"], kwargs["batch_90p_score"]
    prompt_model = kwargs["prompt_model"] or dspy.settings.lm

    module_names = [name for name, _ in system.named_predictors()]
    good, bad = bucket[0], bucket[-1]
    example = good["example"]

    if good["score"] <= batch_10p_score or bad["score"] >= batch_90p_score:
        logger.info(f"Skipping rule generation as good score {good['score']} is at or below the 10th percentile "
                    f"*or* bad score {bad['score']} is at or above the 90th percentile.")
        return False

    if good["score"] <= bad["score"]:
        if good["score"] > batch_90p_score:
            bad["trace"] = []
            bad["score"] = "N/A"
            bad["prediction"] = {"N/A": "Prediction not available"}
        else:
            good["trace"] = []
            good["score"] = "N/A"
            good["prediction"] = {"N/A": "Prediction not available"}

    better_trajectory = [
        dict(module_name=predictor2name[id(p)], inputs=i, outputs=dict(o))
        for p, i, o in good["trace"]
    ]
    worse_trajectory = [
        dict(module_name=predictor2name[id(p)], inputs=i, outputs=dict(o))
        for p, i, o in bad["trace"]
    ]

    def extract_model_fields(model, prefix, current_fields_dict, module_name):
        if hasattr(model, 'model_fields'):
            for nested_field_name, nested_field in model.model_fields.items():
                nested_key = f"{prefix}.{nested_field_name}"
                current_fields_dict[module_name][nested_key] = {}
                current_fields_dict[module_name][nested_key]["name"] = nested_field_name
                current_fields_dict[module_name][nested_key]["desc"] = nested_field.description or ""

    def process_field(field, field_name, current_fields_dict, module_name):
        if hasattr(field.annotation, 'model_fields'):
            extract_model_fields(field.annotation, field_name, current_fields_dict, module_name)
        elif hasattr(field.annotation, '__origin__'):
            origin = field.annotation.__origin__
            args = field.annotation.__args__
            
            if origin is list:
                extract_model_fields(args[0], field_name, current_fields_dict, module_name)
            elif origin is Union:
                for arg in args:
                    if arg != type(None):
                        extract_model_fields(arg, field_name, current_fields_dict, module_name)
            elif origin is tuple:
                for i, arg in enumerate(args):
                    extract_model_fields(arg, f"{field_name}.{i}", current_fields_dict, module_name)
            elif origin is dict:
                extract_model_fields(args[1], f"{field_name}.value", current_fields_dict, module_name)
        elif hasattr(field.annotation, '__bases__') and any(issubclass(base, Type) for base in field.annotation.__bases__):
            extract_model_fields(field.annotation, field_name, current_fields_dict, module_name)

    # Get the current fields
    current_fields = {}
    for name, predictor in system.named_predictors():
        current_fields[name] = {}
        for field_name, field in predictor.signature.input_fields.items():
            current_fields[name][field_name] = {}
            current_fields[name][field_name]["name"] = field.json_schema_extra["prefix"]
            current_fields[name][field_name]["desc"] = field.json_schema_extra["desc"]
            process_field(field, field_name, current_fields, name)
        for field_name, field in predictor.signature.output_fields.items():
            current_fields[name][field_name] = {}
            current_fields[name][field_name]["name"] = field.json_schema_extra["prefix"]
            current_fields[name][field_name]["desc"] = field.json_schema_extra["desc"]
            process_field(field, field_name, current_fields, name)

    kwargs = dict(
        program_code=inspect.getsource(system.__class__),
        modules_defn=inspect_modules(system),
        program_inputs={**example.inputs()},
        oracle_metadata={**example.labels()},
        current_fields=current_fields,
        better_program_trajectory=better_trajectory,
        better_program_outputs=dict(good["prediction"]),
        worse_program_trajectory=worse_trajectory,
        worse_program_outputs=dict(bad["prediction"] or {}),
        worse_reward_value=bad["score"],
        better_reward_value=good["score"],
        worse_reward_info=bad["output_metadata"],
        better_reward_info=good["output_metadata"],
    )

    kwargs = {k: v if isinstance(v, str) else ujson.dumps(recursive_mask(v), indent=2)
              for k, v in kwargs.items()}

    # Get new prefixes and descriptions for each field
    with dspy.settings.context(trace=[], lm=prompt_model):
        update_fields_program = dspy.Predict(UpdateFields)
        updated_fields = update_fields_program(**kwargs).updated_fields
    
    # Set the prefix and description of the fields
    for name, predictor in system.named_predictors():
        if name in updated_fields:
            for field_name, field in predictor.signature.input_fields.items():
                if field_name in updated_fields[name]:
                    if "name" in updated_fields[name][field_name]:
                        field.json_schema_extra["prefix"] = updated_fields[name][field_name]["name"]
                    if "desc" in updated_fields[name][field_name]:
                        field.json_schema_extra["desc"] = updated_fields[name][field_name]["desc"]
            for field_name, field in predictor.signature.output_fields.items():
                if field_name in updated_fields[name]:
                    if "name" in updated_fields[name][field_name]:
                        field.json_schema_extra["prefix"] = updated_fields[name][field_name]["name"]
                    if "desc" in updated_fields[name][field_name]:
                        field.json_schema_extra["desc"] = updated_fields[name][field_name]["desc"]

    prompt_model.inspect_history(n=1)
    print(f"Current fields: {current_fields}")
    print(f"Updated fields: {updated_fields}")

    return True

class UpdateFields(dspy.Signature):
    """
    You will be given two trajectories of an LLM-driven program's execution: one that is successful and one that is not.
    You will also be provided with the current fields of the program, which are being used to describe the desired inputs and outputs of the program to the LLM.
    Your goal is to update the fields of the program to be more accurate and informative to ensure that the program
    is able to learn from the successful trajectory and avoid the mistakes of the unsuccessful trajectory. You can update both the name and the description of the fields.

    These fields are important because they are used to provide the LLM with a description of the inputs it will receive,
    and the outputs it will produce.

    """
    program_code: str = InputField(desc="The code of the program that we are analyzing")
    modules_defn: str = InputField(desc="The definition of each module in the program, including its I/O")
    program_inputs: str = InputField(desc="The inputs to the program that we are analyzing")
    oracle_metadata: str = InputField(desc="Any (hidden) metadata about the training set instance we're analyzing")
    worse_program_trajectory: str = InputField(
        desc="The trajectory of the program's execution, showing each module's I/O"
    )
    worse_program_outputs: str = InputField(desc="The outputs of the program that we are analyzing")
    worse_reward_value: float = InputField(desc="The reward value assigned to the program's outputs")
    worse_reward_info: str = InputField(desc="Additional information that might be helpful to understanding the assigned reward value.")
    better_program_trajectory: str = InputField(
        desc="The trajectory of the program's execution, showing each module's I/O"
    )
    better_program_outputs: str = InputField(desc="The outputs of the program that we are analyzing")
    better_reward_value: float = InputField(desc="The reward value assigned to the program's outputs")
    better_reward_info: str = InputField(desc="Additional information that might be helpful to understanding the assigned reward value.")
    current_fields: dict[str, dict[str, dict[str, str]]] = InputField(desc="A dictionary of current field names and descriptions for the program.")
    discussion: str = OutputField(desc="Discussing blame of where each module went wrong, if it did.")
    field_discussion: str = OutputField(desc="Discussing the changes to the fields that should be made for each model in the program.")
    updated_fields: dict[str, dict[str, dict[str, str]]] = OutputField(desc="A dictionary of new field names and descriptions for each module in the program. These will be used to update the fields of the program to better clarify expected inputs & outputs to the LLM.")

class OfferFeedback(dspy.Signature):
    """
    You will be given two trajectories of an LLM-driven program's execution. Your goal is to help the program's modules
    build up experience on how to maximize the reward value assigned to the program's outputs if it were to receive
    similar inputs in the future.

    The module won't see its own history. It will rely on your advice balancing being concrete and being generalizable.

    In your advice:
    - Avoid boilerplate. Offer advice that would change the module's behavior for the better in the future.
    - Ensure that advice offered to a module M is specific to that M's specific sub-task, not the overall program.
    - Rely on contrasting the behavior of the worse trajectory against the better trajectory in making recommendations.
    - Ensure each unique module name appears exactly once as a key in the advice dictionary.
    """

    program_code: str = InputField(desc="The code of the program that we are analyzing")
    modules_defn: str = InputField(desc="The definition of each module in the program, including its I/O")
    program_inputs: str = InputField(desc="The inputs to the program that we are analyzing")
    oracle_metadata: str = InputField(desc="Any (hidden) metadata about the training set instance we're analyzing")
    worse_program_trajectory: str = InputField(
        desc="The trajectory of the program's execution, showing each module's I/O"
    )
    worse_program_outputs: str = InputField(desc="The outputs of the program that we are analyzing")
    worse_reward_value: float = InputField(desc="The reward value assigned to the program's outputs")
    worse_reward_info: str = InputField(desc="Additional information that might be helpful to understanding the assigned reward value.")
    better_program_trajectory: str = InputField(
        desc="The trajectory of the program's execution, showing each module's I/O"
    )
    better_program_outputs: str = InputField(desc="The outputs of the program that we are analyzing")
    better_reward_value: float = InputField(desc="The reward value assigned to the program's outputs")
    better_reward_info: str = InputField(desc="Additional information that might be helpful to understanding the assigned reward value.")
    module_names: list[str] = InputField(desc="The names of the modules in the program, for which we seek advice")
    discussion: str = OutputField(desc="Discussing blame of where each module went wrong, if it did")
    module_advice: dict[str, str] = OutputField(
        desc="For each module, describe very concretely: If the module receives ${description of input or patterns "
        "therein}, then it should ${description of content, behavior, or strategies to adopt and/or others to avoid}. "
        "Basically, your advice be such that if the module has access to your tip, it would be much more likely to act "
        "like the successful trajectory rather than the lower-scoring trajectory."
    )

def inspect_modules(program):
    separator = "-" * 80
    output = [separator]

    for idx, (name, predictor) in enumerate(program.named_predictors()):
        signature = predictor.signature
        instructions = textwrap.dedent(signature.instructions)
        instructions = ("\n" + "\t" * 2).join([""] + instructions.splitlines())

        output.append(f"Module {name}")
        output.append("\n\tInput Fields:")
        output.append(("\n" + "\t" * 2).join([""] + enumerate_fields(signature.input_fields).splitlines()))
        output.append("\tOutput Fields:")
        output.append(("\n" + "\t" * 2).join([""] + enumerate_fields(signature.output_fields).splitlines()))
        output.append(f"\tOriginal Instructions: {instructions}")
        output.append(separator)

    return "\n".join([o.strip("\n") for o in output])


def recursive_mask(o):
    # If the object is already serializable, return it.
    try:
        ujson.dumps(o)
        return o
    except TypeError:
        pass

    # If it's a dictionary, apply recursively to its values.
    if isinstance(o, dict):
        return {k: recursive_mask(v) for k, v in o.items()}
    # If it's a list, apply recursively.
    elif isinstance(o, list):
        return [recursive_mask(v) for v in o]
    # If it's a tuple, apply recursively.
    elif isinstance(o, tuple):
        return tuple(recursive_mask(v) for v in o)
    # Otherwise, replace it with a placeholder string (or use repr(o)).
    else:
        return f"<non-serializable: {type(o).__name__}>"