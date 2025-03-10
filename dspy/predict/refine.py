import inspect
import textwrap
from typing import Callable, Optional

import ujson

import dspy
from dspy.adapters.chat_adapter import enumerate_fields
from dspy.predict.predict import Prediction
from dspy.signatures import InputField, OutputField, Signature

from .predict import Module


class OfferFeedback(Signature):
    """
    In the discussion, assign blame to each module that contributed to the final reward being below the threshold, if
    any. Then, prescribe concrete advice of how the module should act on its future input when we retry the process, if
    it were to receive the same or similar inputs. If a module is not to blame, the advice should be N/A.
    The module will not see its own history, so it needs to rely on entirely concrete and actionable advice from you
    to avoid the same mistake on the same or similar inputs.
    """

    program_code: str = InputField(desc="The code of the program that we are analyzing")
    modules_defn: str = InputField(desc="The definition of each module in the program, including its I/O")
    program_inputs: str = InputField(desc="The inputs to the program that we are analyzing")
    program_trajectory: str = InputField(desc="The trajectory of the program's execution, showing each module's I/O")
    program_outputs: str = InputField(desc="The outputs of the program that we are analyzing")
    reward_code: str = InputField(desc="The code of the reward function that we are analyzing")
    target_threshold: float = InputField(desc="The target threshold for the reward function")
    reward_value: float = InputField(desc="The reward value assigned to the program's outputs")
    module_names: list[str] = InputField(desc="The names of the modules in the program, for which we seek advice")
    discussion: str = OutputField(desc="Discussing blame of where each module went wrong, if it did")
    advice: dict[str, str] = OutputField(
        desc="For each module, describe very concretely, in this order: the specific scenarios in which it has made "
        "mistakes in the past and what each mistake was, followed by what it should do differently in that kind of"
        "scenario in the future. If the module is not to blame, write N/A."
    )


class Refine(Module):
    def __init__(
        self,
        module: Module,
        N: int,
        reward_fn: Callable[[dict, Prediction], float],
        threshold: float,
        fail_count: Optional[int] = None,
    ):
        """
        Refines a module by running it up to N times with different temperatures and returns the best prediction.

        This module runs the provided module multiple times with varying temperature settings and selects
        either the first prediction that exceeds the specified threshold or the one with the highest reward.
        If no prediction meets the threshold, it automatically generates feedback to improve future predictions.


        Args:
            module (Module): The module to refine.
            N (int): The number of times to run the module. must
            reward_fn (Callable): The reward function.
            threshold (float): The threshold for the reward function.
            fail_count (Optional[int], optional): The number of times the module can fail before raising an error

        Example:
            >>> import dspy
            >>>
            >>> qa = dspy.ChainOfThought("question -> answer")
            >>> def one_word_answer(args, pred):
            >>>     return 1.0 if len(pred.answer) == 1 else 0.0
            >>> best_of_3 = dspy.Refine(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0)
            >>> best_of_3(question="What is the capital of Belgium?").answer
            >>> # Brussels
        """
        self.module = module
        self.reward_fn = lambda *args: reward_fn(*args)  # to prevent this from becoming a parameter
        self.threshold = threshold
        self.N = N
        self.fail_count = fail_count or N  # default to N if fail_count is not provided
        self.module_code = inspect.getsource(module.__class__)
        try:
            self.reward_fn_code = inspect.getsource(reward_fn)
        except TypeError:
            self.reward_fn_code = inspect.getsource(reward_fn.__class__)

    def forward(self, **kwargs):
        lm = self.module.get_lm() or dspy.settings.lm
        temps = [lm.kwargs["temperature"]] + [0.5 + i * (0.5 / self.N) for i in range(self.N)]
        temps = list(dict.fromkeys(temps))[: self.N]
        best_pred, best_trace, best_reward = None, None, -float("inf")
        advice = None
        adapter = dspy.settings.adapter or dspy.ChatAdapter()

        for idx, t in enumerate(temps):
            lm_ = lm.copy(temperature=t)
            mod = self.module.deepcopy()
            mod.set_lm(lm_)

            predictor2name = {predictor: name for name, predictor in mod.named_predictors()}
            signature2name = {predictor.signature: name for name, predictor in mod.named_predictors()}
            module_names = [name for name, _ in mod.named_predictors()]

            try:
                with dspy.context(trace=[]):
                    if not advice:
                        outputs = mod(**kwargs)
                    else:

                        class WrapperAdapter(adapter.__class__):
                            def __call__(self, lm, lm_kwargs, signature, demos, inputs):
                                inputs["hint_"] = advice.get(signature2name[signature], "N/A")
                                signature = signature.append(
                                    "hint_", InputField(desc="A hint to the module from an earlier run")
                                )
                                return adapter(lm, lm_kwargs, signature, demos, inputs)

                        with dspy.context(adapter=WrapperAdapter()):
                            outputs = mod(**kwargs)

                    trace = dspy.settings.trace.copy()

                    # TODO: Remove the hint from the trace, if it's there.

                    # NOTE: Not including the trace of reward_fn.
                    reward = self.reward_fn(kwargs, outputs)

                if reward > best_reward:
                    best_reward, best_pred, best_trace = reward, outputs, trace

                if self.threshold is not None and reward >= self.threshold:
                    break

                if idx == self.N - 1:
                    break

                modules = dict(program_code=self.module_code, modules_defn=inspect_modules(mod))
                trajectory = [dict(module_name=predictor2name[p], inputs=i, outputs=dict(o)) for p, i, o in trace]
                trajectory = dict(program_inputs=kwargs, program_trajectory=trajectory, program_outputs=dict(outputs))
                reward = dict(reward_code=self.reward_fn_code, target_threshold=self.threshold, reward_value=reward)

                advise_kwargs = dict(**modules, **trajectory, **reward, module_names=module_names)
                # advise_kwargs = {k: ujson.dumps(recursive_mask(v), indent=2) for k, v in advise_kwargs.items()}
                # only dumps if it's a list or dict
                advise_kwargs = {
                    k: v if isinstance(v, str) else ujson.dumps(recursive_mask(v), indent=2)
                    for k, v in advise_kwargs.items()
                }
                advice = dspy.Predict(OfferFeedback)(**advise_kwargs).advice
                # print(f"Advice for each module: {advice}")

            except Exception as e:
                print(f"Refine: Attempt failed with temperature {t}: {e}")
                if idx > self.fail_count:
                    raise e
                self.fail_count -= 1
        if best_trace:
            dspy.settings.trace.extend(best_trace)
        return best_pred


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
