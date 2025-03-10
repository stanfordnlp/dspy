from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import numpy as np

import optuna
from optuna._experimental import experimental_class
from optuna.pruners import BasePruner
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial


if TYPE_CHECKING:
    import scipy.stats as ss
else:
    from optuna._imports import _LazyImport

    ss = _LazyImport("scipy.stats")


@experimental_class("3.6.0")
class WilcoxonPruner(BasePruner):
    """Pruner based on the `Wilcoxon signed-rank test <https://en.wikipedia.org/w/index.php?title=Wilcoxon_signed-rank_test&oldid=1195011212>`__.

    This pruner performs the Wilcoxon signed-rank test between the current trial and the current best trial,
    and stops whenever the pruner is sure up to a given p-value that the current trial is worse than the best one.

    This pruner is effective for optimizing the mean/median of some (costly-to-evaluate) performance scores over a set of problem instances.
    Example applications include the optimization of:

    * the mean performance of a heuristic method (simulated annealing, genetic algorithm, SAT solver, etc.) on a set of problem instances,
    * the k-fold cross-validation score of a machine learning model, and
    * the accuracy of outputs of a large language model (LLM) on a set of questions.

    There can be "easy" or "hard" instances (the pruner handles correspondence of the instances between different trials).
    In each trial, it is recommended to shuffle the evaluation order, so that the optimization doesn't overfit to the instances in the beginning.

    When you use this pruner, you must call ``Trial.report(value, step)`` method for each step (instance id) with
    the evaluated value. The instance id may not be in ascending order.
    This is different from other pruners in that the reported value need not converge
    to the real value. To use pruners such as :class:`~optuna.pruners.SuccessiveHalvingPruner`
    in the same setting, you must provide e.g., the historical average of the evaluated values.

    .. seealso::
        Please refer to :meth:`~optuna.trial.Trial.report`.

    Example:

        .. testcode::

            import optuna
            import numpy as np


            # We minimize the mean evaluation loss over all the problem instances.
            def evaluate(param, instance):
                # A toy loss function for demonstrative purpose.
                return (param - instance) ** 2


            problem_instances = np.linspace(-1, 1, 100)


            def objective(trial):
                # Sample a parameter.
                param = trial.suggest_float("param", -1, 1)

                # Evaluate performance of the parameter.
                results = []

                # For best results, shuffle the evaluation order in each trial.
                instance_ids = np.random.permutation(len(problem_instances))
                for instance_id in instance_ids:
                    loss = evaluate(param, problem_instances[instance_id])
                    results.append(loss)

                    # Report loss together with the instance id.
                    # CAVEAT: You need to pass the same id for the same instance,
                    # otherwise WilcoxonPruner cannot correctly pair the losses across trials and
                    # the pruning performance will degrade.
                    trial.report(loss, instance_id)

                    if trial.should_prune():
                        # Return the current predicted value instead of raising `TrialPruned`.
                        # This is a workaround to tell the Optuna about the evaluation
                        # results in pruned trials. (See the note below.)
                        return sum(results) / len(results)

                return sum(results) / len(results)


            study = optuna.create_study(pruner=optuna.pruners.WilcoxonPruner(p_threshold=0.1))
            study.optimize(objective, n_trials=100)



    .. note::
        This pruner cannot handle ``infinity`` or ``nan`` values.
        Trials containing those values are never pruned.

    .. note::
        If :func:`~optuna.trial.FrozenTrial.should_prune` returns :obj:`True`, you can return an
        estimation of the final value (e.g., the average of all evaluated
        values) instead of ``raise optuna.TrialPruned()``.
        This is a workaround for the problem that currently there is no way
        to tell Optuna the predicted objective value for trials raising
        :class:`optuna.TrialPruned`.

    Args:
        p_threshold:
            The p-value threshold for pruning. This value should be between 0 and 1.
            A trial will be pruned whenever the pruner is sure up to the given p-value
            that the current trial is worse than the best trial.
            The larger this value is, the more aggressive pruning will be performed.
            Defaults to 0.1.

            .. note::
                This pruner repeatedly performs statistical tests between the
                current trial and the current best trial with increasing samples.
                The false-positive rate of such a sequential test is different from
                performing the test only once. To get the nominal false-positive rate,
                please specify the Pocock-corrected p-value.

        n_startup_steps:
            The number of steps before which no trials are pruned.
            Pruning starts only after you have ``n_startup_steps`` steps of
            available observations for comparison between the current trial
            and the best trial.
            Defaults to 2. Note that the trial is not pruned at the first and second steps even if
            the `n_startup_steps` is set to 0 or 1 due to the lack of enough data for comparison.
    """  # NOQA: E501

    def __init__(
        self,
        *,
        p_threshold: float = 0.1,
        n_startup_steps: int = 2,
    ) -> None:
        if n_startup_steps < 0:  # TODO: Consider changing the RHS to 2.
            raise ValueError(f"n_startup_steps must be nonnegative but got {n_startup_steps}.")
        if not 0.0 <= p_threshold <= 1.0:
            raise ValueError(f"p_threshold must be between 0 and 1 but got {p_threshold}.")

        self._n_startup_steps = n_startup_steps
        self._p_threshold = p_threshold

    def prune(self, study: "optuna.study.Study", trial: FrozenTrial) -> bool:
        if len(trial.intermediate_values) == 0:
            return False

        steps, step_values = np.array(list(trial.intermediate_values.items())).T

        if np.any(~np.isfinite(step_values)):
            warnings.warn(
                f"The intermediate values of the current trial (trial {trial.number}) "
                f"contain infinity/NaNs. WilcoxonPruner will not prune this trial."
            )
            return False

        try:
            best_trial = study.best_trial
        except ValueError:
            return False

        if len(best_trial.intermediate_values) == 0:
            warnings.warn(
                "The best trial has no intermediate values so WilcoxonPruner cannot prune trials. "
                "If you have added the best trial with Study.add_trial, please consider setting "
                "intermediate_values argument."
            )
            return False

        best_steps, best_step_values = np.array(list(best_trial.intermediate_values.items())).T

        if np.any(~np.isfinite(best_step_values)):
            warnings.warn(
                f"The intermediate values of the best trial (trial {best_trial.number}) "
                f"contain infinity/NaNs. WilcoxonPruner will not prune the current trial."
            )
            return False

        _, idx1, idx2 = np.intersect1d(steps, best_steps, return_indices=True)

        if len(idx1) < len(step_values):
            # This if-statement is never satisfied if following "average_is_best" safety works,
            # because the safety ensures that the best trial always has the all steps.
            warnings.warn(
                "WilcoxonPruner finds steps existing in the current trial "
                "but does not exist in the best trial. "
                "Those values are ignored."
            )

        diff_values = step_values[idx1] - best_step_values[idx2]

        if len(diff_values) < max(2, self._n_startup_steps):
            return False

        if study.direction == StudyDirection.MAXIMIZE:
            alt = "less"
            average_is_best = sum(best_step_values) / len(best_step_values) <= sum(
                step_values
            ) / len(step_values)
        else:
            alt = "greater"
            average_is_best = sum(best_step_values) / len(best_step_values) >= sum(
                step_values
            ) / len(step_values)

        # We use zsplit to avoid the problem when all values are zero.
        p = ss.wilcoxon(diff_values, alternative=alt, zero_method="zsplit").pvalue

        if p < self._p_threshold and average_is_best:
            # ss.wilcoxon found the current trial is probably worse than the best trial,
            # but the value of the best trial was not better than
            # the average of the current trial's intermediate values.
            # For safety, WilcoxonPruner concludes not to prune it for now.
            return False
        return p < self._p_threshold
