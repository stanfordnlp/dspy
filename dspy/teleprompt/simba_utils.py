import dspy
import ujson
import inspect
import logging
import textwrap

from dspy.adapters.chat_adapter import enumerate_fields
from dspy.signatures import InputField, OutputField
from typing import Callable

logger = logging.getLogger(__name__)


def prepare_models_for_resampling(program: dspy.Module, n: int):
    lm = program.get_lm() or dspy.settings.lm
    temps = [lm.kwargs["temperature"]] + [0.5 + i * (0.5 / n) for i in range(n)]
    temps = list(dict.fromkeys(temps))[:n]
    return [lm.copy(temperature=t) for t in temps]


def wrap_program(program: dspy.Module, metric: Callable):
    def wrapped_program(example):
        with dspy.context(trace=[]):
            prediction, trace, score = None, None, 0.0
            try:
                prediction = program(**example.inputs())
            except Exception as e:
                print(e)
            trace = dspy.settings.trace.copy()

        try:
            score = metric(example, prediction)
        except Exception as e:
            print(e)

        # Include the `example` in the output for subsequent usage in buckets/strategies.
        return {
            "prediction": prediction,
            "trace": trace,
            "score": score,
            "example": example
        }

    return wrapped_program



def append_a_demo(demo_input_field_maxlen):
    def append_a_demo_(bucket, system, **kwargs):
        predictor2name, name2predictor = kwargs["predictor2name"], kwargs["name2predictor"]

        trace = bucket[0]["trace"]
        name2demo = {}

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

        logger.info(f"Added {len(name2demo)} demos (one each) across all predictors.")
        return True
    
    return append_a_demo_


def append_a_rule(bucket, system, **kwargs):
    predictor2name = kwargs["predictor2name"]
    batch_10p_score, batch_90p_score = kwargs["batch_10p_score"], kwargs["batch_90p_score"]

    module_names = [name for name, _ in system.named_predictors()]
    good, bad = bucket[0], bucket[-1]
    example = good["example"]

    if good["score"] < batch_10p_score or bad["score"] > batch_90p_score:
        logger.info(f"Skipping rule generation as good score {good['score']} is below the 10th percentile "
                    f"*or* bad score {bad['score']} is above the 90th percentile.")
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
        module_names=module_names,
    )

    kwargs = {k: v if isinstance(v, str) else ujson.dumps(recursive_mask(v), indent=2)
              for k, v in kwargs.items()}
    advice = dspy.Predict(OfferFeedback)(**kwargs).module_advice

    for name, predictor in system.named_predictors():
        if name in advice:
            logger.info(f"Advice for {name}: {advice[name]}")
            instructions = predictor.signature.instructions + "\n\n" + advice[name]
            predictor.signature = predictor.signature.with_instructions(instructions)

    return True

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
    better_program_trajectory: str = InputField(
        desc="The trajectory of the program's execution, showing each module's I/O"
    )
    better_program_outputs: str = InputField(desc="The outputs of the program that we are analyzing")
    better_reward_value: float = InputField(desc="The reward value assigned to the program's outputs")
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
