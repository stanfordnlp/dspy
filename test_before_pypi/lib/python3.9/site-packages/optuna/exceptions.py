class OptunaError(Exception):
    """Base class for Optuna specific errors."""

    pass


class TrialPruned(OptunaError):
    """Exception for pruned trials.

    This error tells a trainer that the current :class:`~optuna.trial.Trial` was pruned. It is
    supposed to be raised after :func:`optuna.trial.Trial.should_prune` as shown in the following
    example.

    See also:
        :class:`optuna.TrialPruned` is an alias of :class:`optuna.exceptions.TrialPruned`.

    Example:

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


            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)
    """

    pass


class CLIUsageError(OptunaError):
    """Exception for CLI.

    CLI raises this exception when it receives invalid configuration.
    """

    pass


class StorageInternalError(OptunaError):
    """Exception for storage operation.

    This error is raised when an operation failed in backend DB of storage.
    """

    pass


class DuplicatedStudyError(OptunaError):
    """Exception for a duplicated study name.

    This error is raised when a specified study name already exists in the storage.
    """

    pass


class ExperimentalWarning(Warning):
    """Experimental Warning class.

    This implementation exists here because the policy of `FutureWarning` has been changed
    since Python 3.7 was released. See the details in
    https://docs.python.org/3/library/warnings.html#warning-categories.
    """

    pass
