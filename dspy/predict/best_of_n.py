import dspy

from .predict import Module
from typing import Callable


class BestOfN(Module):
    def __init__(self, module, N: int, reward_fn: Callable, threshold: float):
        """
        Runs a module up to `N` times with different temperatures and returns the best prediction or the first prediction that passes the `threshold`.

        Args:
            module: The module to run.
            N: The number of times to run the module.
            reward_fn: The reward function.
            threshold: The threshold for the reward function.

        Example:
            >>> import dspy
            >>> qa = dspy.ChainOfThought("question -> answer")
            >>> def one_word_answer(args, pred):
            >>>     return 1.0 if len(pred.answer) == 1 else 0.0
            >>> best_of_3 = dspy.BestOfN(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0)
            >>> best_of_3(question="What is the capital of Belgium?").answer
            >>> # Brussels
        """
        self.module = module
        self.reward_fn = lambda *args: reward_fn(
            *args)  # to prevent this from becoming a parameter
        self.threshold = threshold
        self.N = N

    def forward(self, **kwargs):
        lm = self.module.get_lm() or dspy.settings.lm
        temps = [lm.kwargs['temperature']
                 ] + [0.5 + i * (0.5 / self.N) for i in range(self.N)]
        temps = list(dict.fromkeys(temps))[:self.N]
        best_pred, best_trace, best_reward = None, None, -float("inf")

        for t in temps:
            lm_ = lm.copy(temperature=t)
            mod = self.module.deepcopy()
            mod.set_lm(lm_)

            try:
                with dspy.context(trace=[]):
                    pred = mod(**kwargs)
                    trace = dspy.settings.trace.copy()

                    # NOTE: Not including the trace of reward_fn.
                    reward = self.reward_fn(kwargs, pred)

                if reward > best_reward:
                    best_reward, best_pred, best_trace = reward, pred, trace

                if self.threshold is not None and reward >= self.threshold:
                    break

            except Exception as e:
                print(f"Attempt failed with temperature {t}: {e}")

        dspy.settings.trace.extend(best_trace)
        return best_pred
