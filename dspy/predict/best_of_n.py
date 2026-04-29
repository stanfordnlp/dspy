from typing import Callable

import dspy
from dspy.predict.predict import Module, Prediction


class BestOfN(Module):
    def __init__(
        self,
        module: Module,
        N: int,  # noqa: N803
        reward_fn: Callable[[dict, Prediction], float],
        threshold: float,
        fail_count: int | None = None,
    ):
        """
        Runs a module up to `N` times with different rollout IDs at `temperature=1.0` and
        returns the best prediction out of `N` attempts or the first prediction that passes the
        `threshold`.

        Args:
            module (Module): The module to run.
            N (int): The number of times to run the module.
            reward_fn (Callable[[dict, Prediction], float]): The reward function which takes in the args passed to the module, the resulting prediction, and returns a scalar reward.
            threshold (float): The threshold for the reward function.
            fail_count (Optional[int], optional): The number of times the module can fail before raising an error. Defaults to N if not provided.

        Examples:
            ```python
            import dspy

            dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

            # Define a QA module with chain of thought
            qa = dspy.ChainOfThought("question -> answer")

            # Define a reward function that checks for one-word answers
            def one_word_answer(args, pred):
                return 1.0 if len(pred.answer.split()) == 1 else 0.0

            # Create a refined module that tries up to 3 times
            best_of_3 = dspy.BestOfN(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0)

            # Use the refined module
            result = best_of_3(question="What is the capital of Belgium?").answer
            # Returns: Brussels
            ```
        """
        self.module = module
        self.reward_fn = lambda *args: reward_fn(*args)  # to prevent this from becoming a parameter
        self.threshold = threshold
        self.N = N
        self.fail_count = fail_count or N  # default to N if fail_count is not provided

    def forward(self, **kwargs):
        """Execute the best-of-N selection loop, running the module up to ``N`` times.

        Each iteration runs the module with a unique rollout ID at ``temperature=1.0`` and scores
        the result using the reward function. If a prediction meets or exceeds the threshold, it is
        returned immediately. Otherwise, the prediction with the highest reward across all attempts
        is returned.

        Unlike :class:`~dspy.Refine`, this method does not generate feedback between attempts —
        each attempt is independent.

        Args:
            **kwargs: Keyword arguments passed directly to the underlying module's forward method.
                These should match the input fields defined in the module's signature.

        Returns:
            Prediction: The best prediction found across all attempts, determined by the highest
                reward score. Returns ``None`` if no attempt succeeded and the ``fail_count``
                re-raise threshold was not reached (possible when ``N <= 2`` or
                ``fail_count >= N``).

        Raises:
            Exception: Re-raises the last encountered exception when the current attempt
                index exceeds the remaining ``fail_count`` (decremented after each failure).
        """
        lm = self.module.get_lm() or dspy.settings.lm
        start = lm.kwargs.get("rollout_id", 0)
        rollout_ids = [start + i for i in range(self.N)]
        best_pred, best_trace, best_reward = None, None, -float("inf")

        for idx, rid in enumerate(rollout_ids):
            lm_ = lm.copy(rollout_id=rid, temperature=1.0)
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

                if reward >= self.threshold:
                    break

            except Exception as e:
                print(f"BestOfN: Attempt {idx + 1} failed with rollout id {rid}: {e}")
                if idx > self.fail_count:
                    raise e
                self.fail_count -= 1

        if best_trace:
            dspy.settings.trace.extend(best_trace)
        return best_pred
