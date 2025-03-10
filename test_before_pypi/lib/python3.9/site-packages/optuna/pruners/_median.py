from optuna.pruners._percentile import PercentilePruner


class MedianPruner(PercentilePruner):
    """Pruner using the median stopping rule.

    Prune if the trial's best intermediate result is worse than median of intermediate results of
    previous trials at the same step.

    Example:

        We minimize an objective function with the median stopping rule.

        .. testcode::

            import numpy as np
            from sklearn.datasets import load_iris
            from sklearn.linear_model import SGDClassifier
            from sklearn.model_selection import train_test_split

            import optuna

            X, y = load_iris(return_X_y=True)
            X_train, X_valid, y_train, y_valid = train_test_split(X, y)
            classes = np.unique(y)


            def objective(trial):
                alpha = trial.suggest_float("alpha", 0.0, 1.0)
                clf = SGDClassifier(alpha=alpha)
                n_train_iter = 100

                for step in range(n_train_iter):
                    clf.partial_fit(X_train, y_train, classes=classes)

                    intermediate_value = clf.score(X_valid, y_valid)
                    trial.report(intermediate_value, step)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

                return clf.score(X_valid, y_valid)


            study = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5, n_warmup_steps=30, interval_steps=10
                ),
            )
            study.optimize(objective, n_trials=20)

    Args:
        n_startup_trials:
            Pruning is disabled until the given number of trials finish in the same study.
        n_warmup_steps:
            Pruning is disabled until the trial exceeds the given number of step. Note that
            this feature assumes that ``step`` starts at zero.
        interval_steps:
            Interval in number of steps between the pruning checks, offset by the warmup steps.
            If no value has been reported at the time of a pruning check, that particular check
            will be postponed until a value is reported.
        n_min_trials:
            Minimum number of reported trial results at a step to judge whether to prune.
            If the number of reported intermediate values from all trials at the current step
            is less than ``n_min_trials``, the trial will not be pruned. This can be used to ensure
            that a minimum number of trials are run to completion without being pruned.
    """

    def __init__(
        self,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
        interval_steps: int = 1,
        *,
        n_min_trials: int = 1,
    ) -> None:
        super().__init__(
            50.0, n_startup_trials, n_warmup_steps, interval_steps, n_min_trials=n_min_trials
        )
